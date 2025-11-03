#!/usr/bin/env python3
"""
Quick sampler for TCAV-on-ESM2:
- Positives: PF00096 (C2H2 zinc finger) from InterPro (with domain coordinates)
- Negatives: UniProt reviewed, NO zinc-finger, NO metal-binding

Outputs:
  - znf_pos_100.jsonl / znf_pos_100.fasta
  - neg_100.jsonl     / neg_100.fasta

This version GUARANTEES N valid sequences per class by checking that each
window fits within ESM-2's max token limit (1022 incl. BOS).
"""

import os
import re
import json
import time
import random
import requests
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm

# ----------------------------
# Config (tweak here)
# ----------------------------
POS_N = 100
NEG_N = 100
WINDOW_LEN = 41

SEED = 42
random.seed(SEED)

# ESM-2 token limit: ~1022 tokens including BOS (<cls>)
# A raw 0-based half-open window [w_s, w_e) is valid iff (w_e + 1) <= 1022
ESM_MAX_TOKENS = 1022
def fits_esm_limit(w_end: int) -> bool:
    return (w_end + 1) <= ESM_MAX_TOKENS

# Page sizes / soft limits (script streams until N valid are found)
INTERPRO_PAGE_SIZE = 200
UNIPROT_PAGE_SIZE = 500

# Endpoints
INTERPRO_PFAM_PROTEIN = "https://www.ebi.ac.uk/interpro/api/protein/uniprot/entry/pfam/PF00096/"
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{}.fasta"  # {} -> accession

# Regex helpers
FASTA_HEADER_RE = re.compile(r"^>(\S+)\s*(.*)$")
# Loose ZnF-ish check — FYI only (not a filter)
CYS_HIS_PATTERN = re.compile(r"C.{2,6}C.{8,20}H.{2,6}H", re.IGNORECASE)

# ----------------------------
# HTTP utility
# ----------------------------
def http_get(url: str, params: Dict[str, Any] = None, headers: Dict[str, str] = None,
             retries: int = 4, sleep: float = 0.7, timeout: int = 25):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r
            # Gentle backoff, handle rate limits
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

# ----------------------------
# FASTA helpers
# ----------------------------
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
    # Prefer exact accession match
    for acc, header, seq in recs:
        if accession in acc:
            return header, seq
    if recs:
        return recs[0][1], recs[0][2]
    raise RuntimeError(f"FASTA not found for {accession}")

# ----------------------------
# Window helpers
# ----------------------------
def center_window(i0: int, i1: int, L: int, win: int) -> Tuple[int, int]:
    """
    Domain span [i0, i1] is 1-based inclusive (InterPro/UniProt style).
    Return 0-based half-open [s, e) window of length win, clipped inside [0, L].
    """
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
    """
    Sample a random 0-based half-open window [s, e) of length 'win'
    that will fit the ESM token limit. Returns None if impossible.
    """
    # Available residues (no BOS) must satisfy: (e + 1) <= 1022 -> e <= 1021
    max_residue_end = min(L, ESM_MAX_TOKENS - 1)  # half-open end index limit
    if max_residue_end < win:  # can't place a full window
        return None
    # e = s + win, require e <= max_residue_end -> s <= max_residue_end - win
    s_max = max_residue_end - win
    if s_max < 0:
        return None
    s = random.randint(0, s_max)
    e = s + win
    return (s, e)

# ----------------------------
# InterPro (PF00096) positives - stream until N valid
# ----------------------------
def iter_interpro_pf00096():
    """
    Stream InterPro PF00096 UniProt matches, page by page.
    """
    url = INTERPRO_PFAM_PROTEIN
    params = {"page_size": INTERPRO_PAGE_SIZE}
    while url:
        r = http_get(url, params=params, headers={"Accept": "application/json"})
        data = r.json()
        for item in data.get("results", []):
            yield item
        url = data.get("next")
        params = None

def stream_positives(n: int) -> List[Dict[str, Any]]:
    """
    Stream InterPro, fetch FASTA per accession, keep the first 'n' positives
    whose centered window fits the ESM token limit.
    """
    pos = []
    seen = set()
    pbar = tqdm(total=n, desc="Collecting valid PF00096 positives", unit="seq")
    for item in iter_interpro_pf00096():
        if len(pos) >= n:
            break
        meta = item.get("metadata", {})
        acc = meta.get("accession")
        if not acc or acc in seen:
            continue

        # Gather PF00096 spans from the JSON structure
        spans = []
        # Preferred path: 'entries' → match PF00096 → 'entry_protein_locations'
        for entry in item.get("entries", []):
            if (entry.get("source_database", "").lower() == "pfam" and
                entry.get("accession") == "PF00096"):
                for loc in entry.get("entry_protein_locations", []):
                    for frag in loc.get("fragments", []):
                        s = frag.get("start"); e = frag.get("end")
                        if isinstance(s, int) and isinstance(e, int):
                            spans.append((s, e))
        # Fallback (older shape): 'entry_protein_locations' at top-level
        if not spans:
            for loc in item.get("entry_protein_locations", []) or []:
                entry = loc.get("entry") or {}
                if (entry.get("source_database", "").lower() == "pfam" and
                    entry.get("accession") == "PF00096"):
                    for frag in loc.get("fragments", []):
                        s = frag.get("start"); e = frag.get("end")
                        if isinstance(s, int) and isinstance(e, int):
                            spans.append((s, e))

        if not spans:
            continue

        # Fetch FASTA once for this accession
        try:
            header, seq = fasta_for_accession(acc)
        except Exception:
            continue
        L = len(seq)

        # Try up to 2 spans (randomized) and keep the first that fits
        random.shuffle(spans)
        kept = False
        for (s1, e1) in spans[:2]:
            w_s, w_e = center_window(s1, e1, L, WINDOW_LEN)
            # Must be in-bounds AND fit ESM token limit
            if not (0 <= w_s < w_e <= L):
                continue
            if not fits_esm_limit(w_e):
                continue
            record = {
                "set": "positive",
                "accession": acc,
                "header": header,
                "sequence_length": L,
                "sequence": seq,
                "domain_source": "PF00096",
                "domain_span_1based_inclusive": [s1, e1],
                "window_span_0based_halfopen": [w_s, w_e],
                "window_length": w_e - w_s,
                "quality": {
                    "window_in_bounds": True,
                    "span_length": (e1 - s1 + 1),
                    "loose_cys_his_pattern_in_window": bool(CYS_HIS_PATTERN.search(seq[w_s:w_e])),
                    "fits_esm_limit": True
                },
            }
            pos.append(record)
            seen.add(acc)
            kept = True
            pbar.update(1)
            break

        if not kept:
            seen.add(acc)  # don't retry this accession endlessly

    pbar.close()
    if len(pos) < n:
        raise RuntimeError(f"Only collected {len(pos)} valid positives before stream ended.")
    return pos[:n]

# ----------------------------
# UniProt negatives (reviewed, no ZnF/metal-binding) - stream until N valid
# ----------------------------
def iter_uniprot_negatives():
    """
    Cursor-paginate a UniProt search for negatives (infinite stream until exhausted).
    """
    query = 'reviewed:true AND length:[50 TO 5000] AND NOT keyword:Zinc-finger AND NOT keyword:Metal-binding'
    params = {
        "query": query,
        "format": "tsv",
        "fields": "accession,id,organism_name,length,protein_name",
        "size": UNIPROT_PAGE_SIZE
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
        # Pagination via Link header
        link = r.headers.get("Link")
        next_url = None
        if link:
            parts = [p.strip() for p in link.split(",")]
            for p in parts:
                if 'rel="next"' in p:
                    m = re.search(r'<([^>]+)>', p)
                    if m:
                        next_url = m.group(1)
                        break
        if not next_url:
            break
        url = next_url
        params = None

def stream_negatives(n: int) -> List[Dict[str, Any]]:
    neg = []
    seen = set()
    pbar = tqdm(total=n, desc="Collecting valid negatives", unit="seq")

    for item in iter_uniprot_negatives():
        if len(neg) >= n:
            break
        acc = item["accession"]
        if acc in seen:
            continue
        seen.add(acc)

        try:
            header, seq = fasta_for_accession(acc)
        except Exception:
            continue
        L = len(seq)
        if L < WINDOW_LEN or L > 5000:
            continue

        # Resample a negative window until it fits (bounded tries)
        ok = False
        for _ in range(5):
            w = sample_neg_window(L, WINDOW_LEN)
            if w is None:
                break
            w_s, w_e = w
            if fits_esm_limit(w_e):
                ok = True
                break
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
            "window_length": WINDOW_LEN,
            "quality": {
                "filters_requested": {
                    "reviewed_true": True,
                    "no_zinc_finger_keyword": True,
                    "no_metal_binding_keyword": True,
                    "length_50_5000": 50 <= L <= 5000
                },
                "fits_esm_limit": True
            },
            "uniprot_metadata": {
                "id": item.get("id", ""),
                "organism": item.get("organism", ""),
                "protein_name": item.get("protein_name", ""),
            }
        }
        neg.append(rec)
        pbar.update(1)

    pbar.close()
    if len(neg) < n:
        raise RuntimeError(f"Only collected {len(neg)} valid negatives before stream ended.")
    return neg[:n]

# ----------------------------
# Writers
# ----------------------------
def write_jsonl(path: str, records: List[Dict[str, Any]]):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_fasta(path: str, records: List[Dict[str, Any]]):
    with open(path, "w") as f:
        for r in records:
            acc = r["accession"]
            header = r.get("header", acc)
            seq = r["sequence"]
            f.write(f">{acc} {header}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")

# ----------------------------
# Main
# ----------------------------
def main():
    print(f"Targeting EXACTLY {POS_N} valid positives and {NEG_N} valid negatives (ESM-limit safe).")

    print("Streaming positives (PF00096)…")
    pos = stream_positives(POS_N)
    print(f"✓ Collected positives: {len(pos)}")

    print("Streaming negatives (reviewed/no-ZnF/no-metal)…")
    neg = stream_negatives(NEG_N)
    print(f"✓ Collected negatives: {len(neg)}")

    write_jsonl("znf_pos_100.jsonl", pos)
    write_fasta("znf_pos_100.fasta", pos)

    write_jsonl("neg_100.jsonl", neg)
    write_fasta("neg_100.fasta", neg)

    print("\nDone. Files written:")
    print("  znf_pos_100.jsonl  | znf_pos_100.fasta")
    print("  neg_100.jsonl      | neg_100.fasta")

if __name__ == "__main__":
    main()
