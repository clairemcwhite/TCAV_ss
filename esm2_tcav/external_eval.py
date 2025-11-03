#!/usr/bin/env python3
"""
External evaluation for TCAV + ESM2 (NO PCA).

What this script does:
1) Collect a fresh test set (leakage-safe, ESM-fit):
   - Positives: PF00096 (C2H2 ZnF) from InterPro, centered 41-aa windows.
   - Negatives: reviewed UniProt, NOT (Zinc-finger or Metal-binding), random 41-aa windows.
   - All windows guaranteed to fit ESM token limit (1022 incl. BOS).

2) Embed 41-aa windows with ESM2 from local checkpoint (HuggingFace folder OR fair-esm .pt).

3) Score & evaluate:
   - Use saved scaler + CAV per layer (no PCA).
   - Window-level AUROC/AUPRC, FPR@TPR90; thresholds at target FPRs.
   - Protein-level sliding-window scan (max score per protein); AUROC/AUPRC, Recall@FPR targets.
   - Compare against random CAVs.

Outputs (in outputs/external_eval/<model_name>/):
   - test_pos.jsonl / test_neg.jsonl
   - window_scores.csv
   - protein_scores.csv
   - metrics_window.json
   - metrics_protein.json
   - thresholds_operational.json
   - random_baseline.json
"""

import os
import re
import sys
import json
import math
import time
import yaml
import glob
import argparse
import random
import logging
import pickle
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import requests
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

import torch

# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

# =========================
# Constants & Regex
# =========================
ESM_MAX_TOKENS = 1022                 # includes BOS
WINDOW_LEN_DEFAULT = 41
FASTA_HEADER_RE = re.compile(r"^>(\S+)\s*(.*)$")
CYS_HIS_PATTERN = re.compile(r"C.{2,6}C.{8,20}H.{2,6}H", re.IGNORECASE)

INTERPRO_PFAM_PROTEIN = "https://www.ebi.ac.uk/interpro/api/protein/uniprot/entry/pfam/PF00096/"
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{}.fasta"

# =========================
# HTTP utils
# =========================
def http_get(url: str, params: Dict[str, Any] = None, headers: Dict[str, str] = None,
             retries: int = 4, sleep: float = 0.8, timeout: int = 25):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 502, 503, 504):
                time.sleep(sleep * (attempt + 1))
            else:
                time.sleep(sleep * (attempt + 1))
        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                time.sleep(sleep * (attempt + 1))
                continue
            raise
    r.raise_for_status()
    return r

# =========================
# FASTA parsing
# =========================
def parse_fasta(text: str) -> List[Tuple[str, str, str]]:
    records = []
    acc, header, seq_lines = None, None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if acc is not None:
                records.append((acc, header, "".join(seq_lines)))
            m = FASTA_HEADER_RE.match(line)
            if m:
                acc = m.group(1)
                header = m.group(2)
            else:
                acc = line[1:].split()[0]
                header = line[1:].strip()
            seq_lines = []
        else:
            seq_lines.append(line.strip())
    if acc is not None:
        records.append((acc, header, "".join(seq_lines)))
    return records

def fasta_for_accession(accession: str) -> Tuple[str, str]:
    r = http_get(UNIPROT_FASTA.format(accession))
    recs = parse_fasta(r.text)
    for acc, header, seq in recs:
        if accession in acc:
            return header, seq
    if recs:
        return recs[0][1], recs[0][2]
    raise RuntimeError(f"FASTA not found for {accession}")

# =========================
# Data collection helpers
# =========================
def load_accession_blacklist(*jsonl_paths):
    """
    Load accessions from training JSONL files to create exclusion list.
    
    Args:
        *jsonl_paths: Variable number of JSONL file paths
        
    Returns:
        set: Set of accessions to exclude from external test set
    """
    import json
    bl = set()
    for p in jsonl_paths:
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    if line.strip():
                        acc = json.loads(line).get("accession")
                        if acc:
                            bl.add(acc)
    return bl
def fits_esm_limit(w_end: int) -> bool:
    # raw window end index (0-based, half-open) must satisfy e+1 <= 1022
    return (w_end + 1) <= ESM_MAX_TOKENS

def center_window(i0: int, i1: int, L: int, win: int) -> Tuple[int, int]:
    start0 = max(0, i0 - 1)
    end0 = min(L - 1, i1 - 1)
    center = (start0 + end0) // 2
    half = win // 2
    s = max(0, center - half)
    e = s + win
    if e > L:
        e = L
        s = max(0, L - win)
    return s, e

def sample_neg_window(L: int, win: int) -> Optional[Tuple[int, int]]:
    max_residue_end = min(L, ESM_MAX_TOKENS - 1)
    if max_residue_end < win:
        return None
    s_max = max_residue_end - win
    if s_max < 0:
        return None
    s = random.randint(0, s_max)
    e = s + win
    return (s, e)

def iter_interpro_pf00096():
    url = INTERPRO_PFAM_PROTEIN
    params = {"page_size": 200}
    while url:
        r = http_get(url, params=params, headers={"Accept": "application/json"})
        data = r.json()
        for item in data.get("results", []):
            yield item
        url = data.get("next")
        params = None

def stream_positives(n: int, window_len: int, exclude: set = None) -> List[Dict[str, Any]]:
    exclude = exclude or set()
    pos, seen = [], set()
    pbar = tqdm(total=n, desc="Collecting test PF00096 positives", unit="seq")
    for item in iter_interpro_pf00096():
        if len(pos) >= n:
            break
        acc = (item.get("metadata") or {}).get("accession")
        if not acc or acc in seen or acc in exclude:
            continue

        spans = []
        for entry in item.get("entries", []):
            if entry.get("source_database", "").lower() == "pfam" and entry.get("accession") == "PF00096":
                for loc in entry.get("entry_protein_locations", []):
                    for frag in loc.get("fragments", []):
                        s = frag.get("start"); e = frag.get("end")
                        if isinstance(s, int) and isinstance(e, int):
                            spans.append((s, e))
        if not spans:
            for loc in item.get("entry_protein_locations", []) or []:
                entry = loc.get("entry") or {}
                if entry.get("source_database", "").lower() == "pfam" and entry.get("accession") == "PF00096":
                    for frag in loc.get("fragments", []):
                        s = frag.get("start"); e = frag.get("end")
                        if isinstance(s, int) and isinstance(e, int):
                            spans.append((s, e))
        if not spans:
            continue

        try:
            header, seq = fasta_for_accession(acc)
        except Exception:
            seen.add(acc); continue
        L = len(seq)

        random.shuffle(spans)
        kept = False
        for (s1, e1) in spans[:2]:
            w_s, w_e = center_window(s1, e1, L, window_len)
            if not (0 <= w_s < w_e <= L):
                continue
            if not fits_esm_limit(w_e):
                continue
            rec = {
                "set": "positive",
                "accession": acc,
                "header": header,
                "sequence_length": L,
                "sequence": seq,
                "domain_source": "PF00096",
                "domain_span_1based_inclusive": [s1, e1],
                "window_span_0based_halfopen": [w_s, w_e],
                "window_length": window_len,
            }
            pos.append(rec)
            kept = True
            pbar.update(1)
            break
        seen.add(acc)
    pbar.close()
    if len(pos) < n:
        raise RuntimeError(f"Only collected {len(pos)} positives.")
    return pos

def iter_uniprot_negatives():
    q = 'reviewed:true AND length:[50 TO 5000] AND NOT keyword:Zinc-finger AND NOT keyword:Metal-binding'
    params = {
        "query": q,
        "format": "tsv",
        "fields": "accession,id,organism_name,length,protein_name",
        "size": 500
    }
    url = UNIPROT_SEARCH
    while True:
        r = http_get(url, params=params)
        text = r.text
        lines = text.strip().splitlines()
        if not lines:
            break
        header = lines[0].split("\t")
        idx = {k: i for i, k in enumerate(header)}
        for row in lines[1:]:
            cols = row.split("\t")
            yield {
                "accession": cols[idx["Entry"]],
                "id": cols[idx.get("Entry Name", "Entry")],
                "organism": cols[idx.get("Organism", "Organism")],
                "length": int(cols[idx.get("Length", "Length")]),
                "protein_name": cols[idx.get("Protein names", "Protein name")] if "Protein names" in idx or "Protein name" in idx else ""
            }
        link = r.headers.get("Link")
        next_url = None
        if link:
            for p in [p.strip() for p in link.split(",")]:
                if 'rel="next"' in p:
                    m = re.search(r'<([^>]+)>', p)
                    if m:
                        next_url = m.group(1)
                        break
        if not next_url:
            break
        url = next_url
        params = None

def stream_negatives(n: int, window_len: int, exclude: set = None) -> List[Dict[str, Any]]:
    exclude = exclude or set()
    neg, seen = [], set()
    pbar = tqdm(total=n, desc="Collecting test negatives", unit="seq")
    for item in iter_uniprot_negatives():
        if len(neg) >= n:
            break
        acc = item["accession"]
        if acc in seen or acc in exclude:
            continue
        seen.add(acc)
        try:
            header, seq = fasta_for_accession(acc)
        except Exception:
            continue
        L = len(seq)
        if L < window_len or L > 5000:
            continue
        ok = False
        for _ in range(5):
            w = sample_neg_window(L, window_len)
            if w is None:
                break
            w_s, w_e = w
            if fits_esm_limit(w_e):
                ok = True; break
        if not ok:
            continue
        rec = {
            "set": "negative",
            "accession": acc,
            "header": header,
            "sequence_length": L,
            "sequence": seq,
            "domain_source": None,
            "domain_span_1based_inclusive": None,
            "window_span_0based_halfopen": [w_s, w_e],
            "window_length": window_len,
        }
        neg.append(rec)
        pbar.update(1)
    pbar.close()
    if len(neg) < n:
        raise RuntimeError(f"Only collected {len(neg)} negatives.")
    return neg

def iter_uniprot_hard_negatives():
    """Iterate over UniProt proteins that are metal-binding but NOT zinc-finger."""
    q = 'reviewed:true AND length:[50 TO 5000] AND keyword:Metal-binding AND NOT keyword:Zinc-finger'
    params = {
        "query": q,
        "format": "tsv",
        "fields": "accession,id,organism_name,length,protein_name",
        "size": 500
    }
    url = UNIPROT_SEARCH
    while True:
        r = http_get(url, params=params)
        text = r.text
        lines = text.strip().splitlines()
        if not lines:
            break
        header = lines[0].split("\t")
        idx = {k: i for i, k in enumerate(header)}
        for row in lines[1:]:
            cols = row.split("\t")
            yield {
                "accession": cols[idx["Entry"]],
                "id": cols[idx.get("Entry Name", "Entry")],
                "organism": cols[idx.get("Organism", "Organism")],
                "length": int(cols[idx.get("Length", "Length")]),
                "protein_name": cols[idx.get("Protein names", "Protein name")] if "Protein names" in idx or "Protein name" in idx else ""
            }
        link = r.headers.get("Link")
        next_url = None
        if link:
            for p in [p.strip() for p in link.split(",")]:
                if 'rel="next"' in p:
                    m = re.search(r'<([^>]+)>', p)
                    if m:
                        next_url = m.group(1)
                        break
        if not next_url:
            break
        url = next_url
        params = None

def stream_hard_negatives(n: int, window_len: int, exclude: set = None) -> List[Dict[str, Any]]:
    """Collect hard negatives: metal-binding proteins that are NOT zinc-finger."""
    exclude = exclude or set()
    neg, seen = [], set()
    pbar = tqdm(total=n, desc="Collecting hard negatives (metal-binding, no ZnF)", unit="seq")
    for item in iter_uniprot_hard_negatives():
        if len(neg) >= n:
            break
        acc = item["accession"]
        if acc in seen or acc in exclude:
            continue
        seen.add(acc)
        try:
            header, seq = fasta_for_accession(acc)
        except Exception:
            continue
        L = len(seq)
        if L < window_len or L > 5000:
            continue
        ok = False
        for _ in range(5):
            w = sample_neg_window(L, window_len)
            if w is None:
                break
            w_s, w_e = w
            if fits_esm_limit(w_e):
                ok = True; break
        if not ok:
            continue
        rec = {
            "set": "negative",
            "accession": acc,
            "header": header,
            "sequence_length": L,
            "sequence": seq,
            "domain_source": None,
            "domain_span_1based_inclusive": None,
            "window_span_0based_halfopen": [w_s, w_e],
            "window_length": window_len,
            "negative_type": "hard_metal_binding"
        }
        neg.append(rec)
        pbar.update(1)
    pbar.close()
    if len(neg) < n:
        raise RuntimeError(f"Only collected {len(neg)} hard negatives.")
    return neg

def sample_intra_positive_background(pos_records: List[Dict[str, Any]], n: int, window_len: int) -> List[Dict[str, Any]]:
    """
    Sample off-domain windows from positive proteins as harder negatives.
    For each positive protein, take 1-2 windows far from the annotated span.
    
    Args:
        pos_records: List of positive records with domain spans
        n: Number of background negatives to generate
        window_len: Window length
        
    Returns:
        List of negative records from off-domain regions
    """
    neg = []
    pbar = tqdm(total=n, desc="Sampling intra-positive background", unit="seq")
    
    # Group by accession to avoid sampling multiple windows from same protein
    by_acc = {}
    for rec in pos_records:
        acc = rec["accession"]
        if acc not in by_acc:
            by_acc[acc] = []
        by_acc[acc].append(rec)
    
    accs = list(by_acc.keys())
    random.shuffle(accs)
    
    for acc in accs:
        if len(neg) >= n:
            break
            
        # Get all domain spans for this accession
        all_spans = []
        for rec in by_acc[acc]:
            span = rec.get("domain_span_1based_inclusive")
            if span:
                all_spans.append((span[0] - 1, span[1]))  # Convert to 0-based
        
        if not all_spans:
            continue
            
        # Get sequence length
        seq_len = by_acc[acc][0]["sequence_length"]
        seq = by_acc[acc][0]["sequence"]
        header = by_acc[acc][0]["header"]
        
        # Find regions far from any domain span
        valid_starts = []
        for start in range(0, seq_len - window_len + 1):
            end = start + window_len
            # Check if this window overlaps significantly with any domain span
            overlaps = False
            for span_start, span_end in all_spans:
                # Check for significant overlap (more than 50% of window)
                overlap_start = max(start, span_start)
                overlap_end = min(end, span_end)
                if overlap_end > overlap_start:
                    overlap_len = overlap_end - overlap_start
                    if overlap_len > window_len * 0.5:
                        overlaps = True
                        break
            
            if not overlaps and fits_esm_limit(end):
                valid_starts.append(start)
        
        # Sample 1-2 windows from valid regions
        if valid_starts:
            n_windows = min(2, len(valid_starts), n - len(neg))
            selected_starts = random.sample(valid_starts, n_windows)
            
            for start in selected_starts:
                if len(neg) >= n:
                    break
                end = start + window_len
                
                rec = {
                    "set": "negative",
                    "accession": acc,
                    "header": header,
                    "sequence_length": seq_len,
                    "sequence": seq,
                    "domain_source": None,
                    "domain_span_1based_inclusive": None,
                    "window_span_0based_halfopen": [start, end],
                    "window_length": window_len,
                    "negative_type": "intra_positive_background"
                }
                neg.append(rec)
                pbar.update(1)
    
    pbar.close()
    return neg[:n]

# =========================
# Registry & Model loading
# =========================
def _extract_path_from_registry_entry(entry):
    """
    Accept string or dict entries. For dicts, prefer:
      - checkpoint_path (model weights / HF dir)
      - local_path / path / checkpoint / ckpt (fallback synonyms)
    Returns (checkpoint_path, tokenizer_path, layers_to_extract or None)
    """
    if isinstance(entry, str):
        return entry, None, None

    if isinstance(entry, dict):
        ckpt = (
            entry.get("checkpoint_path")
            or entry.get("local_path")
            or entry.get("path")
            or entry.get("checkpoint")
            or entry.get("ckpt")
        )
        tok = entry.get("tokenizer_path")
        layers = entry.get("layers_to_extract")  # e.g., [2,4,6]
        return ckpt, tok, layers

    return None, None, None


def resolve_model_info(cfg: dict) -> dict:
    """
    Returns a dict:
      {
        'checkpoint_path': <str>,    # HF dir (config.json+pytorch_model.bin) or fair-esm .pt
        'tokenizer_path': <str|None>,# HF dir for tokenizer (optional; default=checkpoint_path)
        'layers': <List[int]>,       # taken from config.model.layers or registry layers_to_extract
        'device': <'cuda'|'cpu'>,
      }
    """
    model_name = cfg["model"]["name"]
    device = cfg["model"].get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Priority 1: explicit overrides in config (like --model_path -> cfg.model.local_path)
    explicit_ckpt = cfg["model"].get("local_path")
    if explicit_ckpt and os.path.exists(explicit_ckpt):
        layers = cfg["model"].get("layers")
        if not layers:
            raise KeyError("Please specify model.layers in config.yaml or via --layers (e.g., 2 4 6).")
        return {
            "checkpoint_path": explicit_ckpt,
            "tokenizer_path": cfg["model"].get("tokenizer_path", explicit_ckpt),
            "layers": layers,
            "device": device,
        }

    # Priority 2: registry lookup
    reg_path = cfg["model"].get("registry_path")
    if not reg_path or not os.path.exists(reg_path):
        raise FileNotFoundError(f"Model registry not found: {reg_path}")

    with open(reg_path, "r") as f:
        reg = yaml.safe_load(f) or {}

    # Accept both top-level and 'models' nesting
    entry = None
    if model_name in reg:
        entry = reg[model_name]
    elif "models" in reg and model_name in reg["models"]:
        entry = reg["models"][model_name]

    if entry is None:
        raise KeyError(f"Model '{model_name}' not found in {reg_path}")

    ckpt, tok, reg_layers = _extract_path_from_registry_entry(entry)
    if not ckpt or not os.path.exists(ckpt):
        raise FileNotFoundError(f"checkpoint_path missing or does not exist for model '{model_name}': {ckpt}")

    # Layers: config override > registry > error
    layers = cfg["model"].get("layers") or reg_layers
    if not layers:
        raise KeyError("No layers specified. Provide model.layers in config or layers_to_extract in registry.")

    if tok is None:
        tok = ckpt  # default tokenizer path same as checkpoint

    return {
        "checkpoint_path": ckpt,
        "tokenizer_path": tok,
        "layers": layers,
        "device": device,
    }


def load_esm_from_config(cfg: dict):
    """
    Load ESM2 either from a fair-esm .pt or a Hugging Face directory.
    Uses the registry-aware resolve_model_info().
    Returns: {'flavor','model','device','tokenizer' or 'alphabet'}
    """
    info = resolve_model_info(cfg)
    ckpt = info["checkpoint_path"]
    tok_path = info["tokenizer_path"]
    device = info["device"]

    # HF if directory with config.json; fair-esm if file endswith .pt
    if os.path.isdir(ckpt) and os.path.exists(os.path.join(ckpt, "config.json")):
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        log.info(f"Loading HF ESM2 from {ckpt} (tokenizer: {tok_path}) on {device} …")
        cfg_hf = AutoConfig.from_pretrained(ckpt, output_hidden_states=True)
        tok = AutoTokenizer.from_pretrained(tok_path, use_fast=False)
        mdl = AutoModel.from_pretrained(ckpt, config=cfg_hf)
        mdl.eval().to(device)
        return {"flavor": "hf", "model": mdl, "tokenizer": tok, "device": device}

    if os.path.isfile(ckpt) and ckpt.endswith(".pt"):
        from esm import pretrained
        log.info(f"Loading fair-esm ESM2 from {ckpt} on {device} …")
        mdl, alphabet = pretrained.load_model_and_alphabet_local(ckpt)
        mdl.eval().to(device)
        return {"flavor": "fair-esm", "model": mdl, "alphabet": alphabet, "device": device}

    # Fallback: try HF if it looks like a dir but no config.json (symlinks etc.)
    if os.path.isdir(ckpt):
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        log.warning(f"Directory without config.json; attempting HF load anyway: {ckpt}")
        cfg_hf = AutoConfig.from_pretrained(ckpt, output_hidden_states=True)
        tok = AutoTokenizer.from_pretrained(tok_path, use_fast=False)
        mdl = AutoModel.from_pretrained(ckpt, config=cfg_hf)
        mdl.eval().to(device)
        return {"flavor": "hf", "model": mdl, "tokenizer": tok, "device": device}

    raise FileNotFoundError(f"Unrecognized model path format for '{ckpt}'. Expect HF dir or fair-esm .pt.")

def tokenize_batch(loader: Dict[str, Any], seqs: List[str]):
    device = loader["device"]
    if loader["flavor"] == "hf":
        tok = loader["tokenizer"]
        enc = tok(seqs, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True)
        return {k: v.to(device) for k, v in enc.items()}
    else:
        alphabet = loader["alphabet"]
        batch_converter = alphabet.get_batch_converter()
        labels = [("seq", s) for s in seqs]
        _, _, toks = batch_converter(labels)
        return {"tokens": toks.to(device)}

# =========================
# Embedding utilities (NO PCA)
# =========================
@torch.no_grad()
def embed_windows(cfg: Dict[str, Any], records: List[Dict[str, Any]], layers: List[int]) -> Dict[int, Dict[str, Any]]:
    """
    Returns: {layer: {'X': np.ndarray [N, D], 'meta': List[dict]}}
    """
    loader = load_esm_from_config(cfg)
    model = loader["model"]
    device = loader["device"]

    results = {L: {"X": [], "meta": []} for L in layers}
    B = int(cfg.get("embedding", {}).get("batch_size", 8))
    bos_shift = 1
    hf_layers = set(layers)

    for i in tqdm(range(0, len(records), B), desc="Embedding windows"):
        batch = records[i:i+B]
        seqs = [r["sequence"] for r in batch]
        toks = tokenize_batch(loader, seqs)

        if loader["flavor"] == "hf":
            out = model(**toks, output_hidden_states=True)
            hidden_states = out.hidden_states  # tuple: [0]=embeddings, [1]=layer1, ...
        else:
            out = model(toks["tokens"], repr_layers=set(layers), return_contacts=False)
            reps = out["representations"]  # dict: L -> [B, T, D]

        for bi, rec in enumerate(batch):
            ws, we = rec["window_span_0based_halfopen"]
            s_tok, e_tok = ws + bos_shift, we + bos_shift

            for L in layers:
                if loader["flavor"] == "hf":
                    H = hidden_states[L][bi]   # [T, D]  (layer indexing: L => hidden_states[L])
                else:
                    H = reps[L][bi]            # [T, D]

                Tlen = H.shape[0]
                s = min(max(0, s_tok), Tlen)
                e = min(max(s, e_tok), Tlen)
                if e <= s:
                    continue

                if isinstance(H, torch.Tensor):
                    window = H[s:e].float()
                    v = window.mean(dim=0).detach().cpu().numpy().astype(np.float32)
                else:
                    v = H[s:e].astype(np.float32).mean(axis=0)

                if not np.isfinite(v).all():
                    continue

                results[L]["X"].append(v)
                results[L]["meta"].append({
                    "accession": rec["accession"],
                    "set": rec["set"],
                    "window_span": [ws, we],
                    "sequence_length": rec["sequence_length"],
                })

    for L in layers:
        if len(results[L]["X"]) == 0:
            raise RuntimeError(f"No valid embeddings for layer {L}")
        results[L]["X"] = np.vstack(results[L]["X"])
    return results

# =========================
# Scoring & metrics
# =========================
def load_layer_artifacts(model_name: str, layer: int, cav_dir: str):
    import io, json
    cav_path = os.path.join(cav_dir, f"L{layer}_concept_v1.npy")
    scaler_pkl = os.path.join(cav_dir, f"L{layer}_scaler_v1.pkl")
    scaler_joblib = os.path.join(cav_dir, f"L{layer}_scaler_v1.joblib")
    scaler_json = os.path.join(cav_dir, f"L{layer}_scaler_v1.json")

    # CAV (always .npy)
    cav = np.load(cav_path).astype(np.float32)

    # Scaler: try pickle → joblib → json
    scaler = None
    if os.path.exists(scaler_pkl):
        try:
            with open(scaler_pkl, "rb") as f:
                scaler = pickle.load(f)
        except Exception as e:
            # maybe it's joblib but named .pkl
            try:
                import joblib
                scaler = joblib.load(scaler_pkl)
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load scaler for L{layer}. Tried pickle and joblib.\n"
                    f"pkl error: {e}\njoblib error: {e2}\n"
                    f"Consider re-saving the scaler, or place a JSON fallback at {scaler_json}"
                )
    elif os.path.exists(scaler_joblib):
        import joblib
        scaler = joblib.load(scaler_joblib)
    elif os.path.exists(scaler_json):
        with open(scaler_json, "r") as f:
            s = json.load(f)
        # reconstruct a StandardScaler from JSON {mean_, scale_}
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.mean_ = np.array(s["mean_"], dtype=np.float64)
        scaler.scale_ = np.array(s["scale_"], dtype=np.float64)
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = scaler.mean_.shape[0]
    else:
        raise FileNotFoundError(
            f"No scaler found for L{layer}. Searched:\n  {scaler_pkl}\n  {scaler_joblib}\n  {scaler_json}"
        )

    # Random CAVs (optional)
    rand_paths = sorted(glob.glob(os.path.join(cav_dir, f"L{layer}_random_*_v1.npy")))
    random_cavs = [np.load(p).astype(np.float32) for p in rand_paths]
    return cav, scaler, random_cavs

def standardize(X: np.ndarray, scaler) -> np.ndarray:
    Xz = scaler.transform(X)
    Xz = np.nan_to_num(Xz, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    return Xz

def layer_scores(Xz: np.ndarray, cav: np.ndarray) -> np.ndarray:
    cav_u = cav / (np.linalg.norm(cav) + 1e-12)
    return Xz @ cav_u

def pick_threshold_for_fpr(scores_neg: np.ndarray, target_fpr: float) -> float:
    if not (0 < target_fpr < 1):
        raise ValueError("target_fpr must be in (0,1)")
    N = len(scores_neg)
    if N == 0:
        return float("inf")
    k = max(0, min(N-1, int(math.ceil(target_fpr * N)) - 1))
    thr = np.sort(scores_neg)[::-1][k]
    return float(thr)

def compute_basic_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    auroc = roc_auc_score(y_true, scores)
    auprc = average_precision_score(y_true, scores)
    fpr, tpr, thr = roc_curve(y_true, scores)
    idx = int(np.argmin(np.abs(tpr - 0.90)))
    fpr_at_tpr90 = float(fpr[idx])
    return {"auroc": float(auroc), "auprc": float(auprc), "fpr_at_tpr90": fpr_at_tpr90}

# =========================
# Protein-level scan
# =========================
@torch.no_grad()
def protein_scan_scores(cfg, records: List[Dict[str, Any]], layers: List[int], cavs: Dict[int, Dict[str, Any]],
                        stride: int, window_len: int) -> Dict[int, Dict[str, Any]]:
    loader = load_esm_from_config(cfg)
    model = loader["model"]
    device = loader["device"]
    bos_shift = 1

    # Prepare per-layer artifacts
    perL = {}
    for L in layers:
        cav = cavs[L]["cav"]
        scaler = cavs[L]["scaler"]
        cav_u = cav / (np.linalg.norm(cav) + 1e-12)
        perL[L] = {"cav_u": cav_u.astype(np.float32), "scaler": scaler}

    out = {L: {"protein_scores": [], "labels": [], "tops": []} for L in layers}

    # scan one-by-one (keeps memory modest)
    if loader["flavor"] == "hf":
        from transformers import AutoTokenizer  # already loaded inside loader; we reuse
        tok = loader["tokenizer"]
    else:
        alphabet = loader["alphabet"]
        batch_converter = alphabet.get_batch_converter()

    for rec in tqdm(records, desc="Protein scan"):
        seq = rec["sequence"]
        label = 1 if rec["set"] == "positive" else 0

        # tokenize single
        if loader["flavor"] == "hf":
            enc = tok([seq], return_tensors="pt", padding=False, truncation=True, add_special_tokens=True)
            toks = {k: v.to(device) for k, v in enc.items()}
            outm = model(**toks, output_hidden_states=True)
            hidden_states = outm.hidden_states
            Tlen = hidden_states[1].shape[1]  # infer token length
        else:
            labels = [("seq", seq)]
            _, _, toks = batch_converter(labels)
            toks = toks.to(device)
            outm = model(toks, repr_layers=set(layers), return_contacts=False)
            reps = outm["representations"]
            Tlen = list(reps.values())[0].shape[1]

        # sliding starts that respect token limit
        Lseq = len(seq)
        max_end_allowed = min(Lseq, ESM_MAX_TOKENS - 1)
        starts = []
        s = 0
        while s + window_len <= max_end_allowed:
            starts.append(s)
            s += stride
        if len(starts) == 0 and Lseq >= window_len and (window_len <= max_end_allowed):
            starts = [0]

        for L in layers:
            if loader["flavor"] == "hf":
                H = hidden_states[L][0].float().cpu().numpy()  # [T, D]
            else:
                H = reps[L][0].float().cpu().numpy()

            best = (-1e9, (0, 0))
            for s in starts:
                e = s + window_len
                s_tok, e_tok = s + bos_shift, e + bos_shift
                if e_tok > H.shape[0]:
                    continue
                v = H[s_tok:e_tok].mean(axis=0).astype(np.float32)
                v = standardize(v[None, :], perL[L]["scaler"])[0]
                score = float(v @ perL[L]["cav_u"])
                if score > best[0]:
                    best = (score, (s, e))
            top_score, (ts, te) = best
            out[L]["protein_scores"].append(top_score)
            out[L]["labels"].append(label)
            out[L]["tops"].append({
                "accession": rec["accession"],
                "top_window": [ts, te],
                "score": top_score
            })

    return out

# =========================
# IO helpers
# =========================
def write_jsonl(path: str, records: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load records from a JSONL file."""
    records = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records

def save_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def save_csv(path: str, header: List[str], rows: List[List[Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default=None, help="Override model.name from config")
    ap.add_argument("--layers", nargs="+", type=int, default=None, help="Override layers from config")
    ap.add_argument("--model_path", default=None, help="Override model.local_path (dir for HF, .pt for fair-esm)")
    ap.add_argument("--pos_n", type=int, default=500)
    ap.add_argument("--neg_n", type=int, default=2000)
    ap.add_argument("--window_len", type=int, default=WINDOW_LEN_DEFAULT)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--fpr_targets", nargs="+", type=float, default=[0.01, 0.05])
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.model:
        cfg["model"]["name"] = args.model
    if args.layers:
        cfg["model"]["layers"] = args.layers
    if args.model_path:
        cfg["model"]["local_path"] = args.model_path

    # … after loading cfg …
    model_name = cfg["model"]["name"]

    # Resolve model info early (also validates registry & layers)
    info = resolve_model_info(cfg)

    # Allow CLI override of layers; otherwise use registry/config-derived
    layers = args.layers if args.layers else info["layers"]

    outdir = args.outdir or os.path.join("outputs", "external_eval", model_name)
    os.makedirs(outdir, exist_ok=True)

    random.seed(42); np.random.seed(42)

    # 1) Collect test data (or load existing)
    test_pos_path = os.path.join(outdir, "test_pos.jsonl")
    test_neg_path = os.path.join(outdir, "test_neg.jsonl")
    
    if os.path.exists(test_pos_path) and os.path.exists(test_neg_path):
        log.info("Loading existing test data...")
        pos = load_jsonl(test_pos_path)
        neg = load_jsonl(test_neg_path)
        log.info(f"✓ Loaded existing test sets: {len(pos)} pos, {len(neg)} neg")
    else:
        # Build blacklist from training data to ensure disjoint accessions
        log.info("Building accession blacklist from training data...")
        blacklist = load_accession_blacklist("znf_pos_100.jsonl", "neg_100.jsonl")
        log.info(f"✓ Blacklisted {len(blacklist)} training accessions")
        
        log.info(f"Collecting clean external test set: pos={args.pos_n}, neg={args.neg_n}, window={args.window_len}")
        
        # Collect positives with exclusion
        pos = stream_positives(args.pos_n, args.window_len, exclude=blacklist)
        
        # Collect diverse negatives: easy + hard + intra-positive background
        easy_neg_n = int(args.neg_n * 0.5)  # 50% easy negatives
        hard_neg_n = int(args.neg_n * 0.3)  # 30% hard negatives  
        intra_neg_n = int(args.neg_n * 0.2)  # 20% intra-positive background
        
        log.info(f"Collecting negatives: {easy_neg_n} easy, {hard_neg_n} hard, {intra_neg_n} intra-positive")
        
        easy_neg = stream_negatives(easy_neg_n, args.window_len, exclude=blacklist)
        hard_neg = stream_hard_negatives(hard_neg_n, args.window_len, exclude=blacklist)
        intra_neg = sample_intra_positive_background(pos, intra_neg_n, args.window_len)
        
        # Combine all negatives
        neg = easy_neg + hard_neg + intra_neg
        
        # Add negative_type to easy negatives for consistency
        for rec in easy_neg:
            rec["negative_type"] = "easy_clean"
        
        log.info(f"✓ Collected: {len(pos)} pos, {len(easy_neg)} easy neg, {len(hard_neg)} hard neg, {len(intra_neg)} intra neg")

        write_jsonl(test_pos_path, pos)
        write_jsonl(test_neg_path, neg)
        log.info("✓ Wrote test sets.")

    # 2) Embed windows (pooled vector per record)
    recs = pos + neg
    Y = np.array([1]*len(pos) + [0]*len(neg), dtype=np.int32)

    emb = embed_windows(cfg, recs, layers)   # {L: {"X": np.ndarray, "meta": []}}
    log.info("✓ Embedded test windows.")

    # 3) Score & window-level metrics
    cav_dir = os.path.join("outputs", "cavs", model_name)
    metrics_window = {}
    thresholds_operational = {}
    random_baseline = {}

    rows_window = [["layer","accession","label","score"]]

    for L in layers:
        cav, scaler, rand_cavs = load_layer_artifacts(model_name, L, cav_dir)
        X = emb[L]["X"]
        meta = emb[L]["meta"]
        Xz = standardize(X, scaler)
        scores = layer_scores(Xz, cav)

        m = compute_basic_metrics(Y, scores)
        neg_scores = scores[Y == 0]
        thrs = {}
        for t in args.fpr_targets:
            th = pick_threshold_for_fpr(neg_scores, t)
            thrs[str(t)] = th
        metrics_window[str(L)] = m
        thresholds_operational[str(L)] = thrs

        if rand_cavs:
            rand_aurocs = []
            for rc in rand_cavs:
                rs = layer_scores(Xz, rc.astype(np.float32))
                rand_aurocs.append(roc_auc_score(Y, rs))
            random_baseline[str(L)] = {
                "mean_auroc": float(np.mean(rand_aurocs)),
                "std_auroc": float(np.std(rand_aurocs)),
                "n": len(rand_cavs)
            }

        for i, mrec in enumerate(meta):
            rows_window.append([f"L{L}", mrec["accession"], int(Y[i]), float(scores[i])])

        log.info(f"[Window] L{L}: AUROC={m['auroc']:.3f} AUPRC={m['auprc']:.3f} FPR@TPR90={m['fpr_at_tpr90']:.3f}")

    save_json(os.path.join(outdir, "metrics_window.json"), metrics_window)
    save_json(os.path.join(outdir, "thresholds_operational.json"), thresholds_operational)
    save_json(os.path.join(outdir, "random_baseline.json"), random_baseline)
    save_csv(os.path.join(outdir, "window_scores.csv"), rows_window[0], rows_window[1:])
    log.info("✓ Saved window-level metrics and thresholds.")

    # 4) Protein-level scanning & metrics
    cavs = {}
    for L in layers:
        cav, scaler, _ = load_layer_artifacts(model_name, L, cav_dir)
        cavs[L] = {"cav": cav, "scaler": scaler}

    scan_out = protein_scan_scores(cfg, recs, layers, cavs, stride=args.stride, window_len=args.window_len)

    metrics_protein = {}
    rows_protein = [["layer","accession","label","protein_score","top_start","top_end"]]
    for L in layers:
        pscores = np.array(scan_out[L]["protein_scores"], dtype=np.float32)
        plabels = np.array(scan_out[L]["labels"], dtype=np.int32)

        m = {
            "auroc": float(roc_auc_score(plabels, pscores)),
            "auprc": float(average_precision_score(plabels, pscores)),
        }

        rec_at_fpr = {}
        for t in args.fpr_targets:
            th = pick_threshold_for_fpr(pscores[plabels == 0], t)
            recall = float((pscores[plabels == 1] >= th).mean())
            rec_at_fpr[str(t)] = recall
        m["recall_at_fpr"] = rec_at_fpr

        metrics_protein[str(L)] = m

        for i, top in enumerate(scan_out[L]["tops"]):
            rows_protein.append([
                f"L{L}",
                top["accession"],
                int(plabels[i]),
                float(pscores[i]),
                top["top_window"][0],
                top["top_window"][1],
            ])

        log.info(f"[Protein] L{L}: AUROC={m['auroc']:.3f} AUPRC={m['auprc']:.3f}  " +
                 " ".join([f"Recall@FPR={k}={v:.3f}" for k, v in rec_at_fpr.items()]))

    save_json(os.path.join(outdir, "metrics_protein.json"), metrics_protein)
    save_csv(os.path.join(outdir, "protein_scores.csv"), rows_protein[0], rows_protein[1:])
    log.info("\nAll done. Outputs in: %s", outdir)

if __name__ == "__main__":
    main()
