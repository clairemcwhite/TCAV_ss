#!/usr/bin/env python3
"""
Fetch a curated set of well-known, diverse Pfam motifs.

Strategy: Use hand-picked Pfam IDs representing major domain families
across different functional categories. This ensures:
- Known biological relevance
- Good protein representation
- Functional diversity
- Faster execution (no need to scan all 20k+ Pfam entries)

Categories covered:
- DNA/RNA binding domains
- Enzymatic domains (kinases, proteases, transferases, etc.)
- Structural/repeat motifs
- Protein-protein interaction domains
- Signaling domains
- Membrane proteins
"""

import json
import time
import requests
from typing import Dict, Any, List
from tqdm import tqdm

# ----------------------------
# Curated Pfam IDs by category
# ----------------------------


CURATED_MOTIFS = {
    # DNA/RNA Binding Domains
    "DNA/RNA_Binding": [
        "PF00096",  # Zinc finger, C2H2
        "PF00010",  # Helix-loop-helix DNA-binding domain
        "PF00046",  # Homeobox domain
        "PF00170",  # bZIP transcription factor
        "PF00249",  # Myb-like DNA-binding domain
        "PF00313",  # 'Cold-shock' DNA-binding domain
        "PF00072",  # Response regulator receiver domain
        "PF00439",  # Bromodomain (acetyl-lysine binding)
        "PF00628",  # PHD-finger
        "PF00856",  # SET domain
        "PF13912",  # C2H2-type zinc finger
        "PF13894",  # C2H2-type zinc finger
        "PF01381",  # Helix-turn-helix DNA-binding domain
        "PF07716",  # Basic Region Leucine Zipper
    ],
    
    # Enzymatic - Kinases & Phosphatases
    "Kinases_Phosphatases": [
        "PF00069",  # Protein kinase domain
        "PF07714",  # Protein tyrosine kinase
        "PF00433",  # Protein phosphatase 2C
        "PF00102",  # Protein-tyrosine phosphatase
        "PF00149",  # Calcineurin-like phosphoesterase
    ],
    
    # Enzymatic - Proteases
    "Proteases": [
        "PF00112",  # Papain family cysteine protease
        "PF00089",  # Trypsin (serine protease)
        "PF00413",  # Matrixin (matrix metalloprotease)
        "PF01650",  # Peptidase C13 (caspase)
        "PF00026",  # Eukaryotic aspartyl protease
    ],
    
    # Enzymatic - Transferases
    "Transferases": [
        "PF00891",  # O-methyltransferase
        "PF00067",  # Cytochrome P450
        "PF00698",  # Acyl transferase domain
        "PF01209",  # ubiE/COQ5 methyltransferase
        "PF00155",  # Aminotransferase class I and II
    ],
    
    # Enzymatic - Hydrolases & Others
    "Hydrolases_Others": [
        "PF00004",  # ATPase family
        "PF00005",  # ABC transporter
        "PF00083",  # Sugar (and other) transporter
        "PF00144",  # Beta-lactamase
        "PF00153",  # Mitochondrial carrier protein
    ],
    
    # Protein-Protein Interaction
    "Protein_Interaction": [
        "PF00018",  # SH3 domain
        "PF00017",  # SH2 domain
        "PF00595",  # PDZ domain
        "PF00536",  # SAM domain (sterile alpha motif)
        "PF00169",  # PH domain
        "PF00614",  # Phosphotyrosine interaction domain (PTB/PID)
        "PF07645",  # WWE domain
        "PF00019",  # Transforming growth factor beta like domain
    ],
    
    # Structural & Repeat Motifs
    "Structural_Repeats": [
        "PF00001",  # 7 transmembrane receptor (rhodopsin family)
        "PF00041",  # Fibronectin type III domain
        "PF00047",  # Immunoglobulin domain
        "PF00400",  # WD domain, G-beta repeat
        "PF00515",  # TPR repeat
        "PF00560",  # Leucine Rich Repeat
        "PF12796",  # Ankyrin repeats (3 copies)
        "PF00023",  # Ankyrin repeat
        "PF13855",  # Leucine rich repeat
        "PF00013",  # KH domain (RNA binding)
    ],
    
    # Signaling Domains
    "Signaling": [
        "PF00130",  # C1 domain (phorbol ester/diacylglycerol binding)
        "PF00168",  # C2 domain
        "PF00071",  # Ras family
        "PF00025",  # ADP-ribosylation factor family
        "PF00616",  # GTPase-activator protein for Ras-like GTPase
        "PF00621",  # RhoGEF domain
        "PF00788",  # RVT_1 Reverse transcriptase (RNA-dependent DNA polymerase)
    ],
    
    # Membrane & Transport
    "Membrane_Transport": [
        "PF00002",  # 7 transmembrane receptor (Secretin family)
        "PF00003",  # 7TM GPCR
        "PF00008",  # EGF-like domain
        "PF00028",  # Cadherin domain
        "PF00822",  # Potassium channel domain
        "PF00520",  # Ion transport protein
        "PF00036",  # EF-hand domain
    ],
    
    # Chromatin & RNA Processing  
    "Chromatin_RNA": [
        "PF00012",  # HSP70 protein
        "PF00076",  # RNA recognition motif
        "PF00098",  # Zinc knuckle
        "PF00271",  # Helicase conserved C-terminal domain
        "PF01423",  # LSM domain
        "PF00575",  # S1 RNA binding domain
    ],
    
    # Immunity & Defense
    "Immunity": [
        "PF00008",  # EGF-like domain
        "PF00031",  # Cystatin domain
        "PF07679",  # Immunoglobulin I-set domain
        "PF07654",  # Immunoglobulin C1-set domain
        "PF00092",  # von Willebrand factor type A domain
    ],
    
    # Metabolic Enzymes
    "Metabolic": [
        "PF00106",  # short chain dehydrogenase
        "PF00107",  # Zinc-binding dehydrogenase
        "PF00171",  # Aldehyde dehydrogenase family
        "PF00109",  # Beta-ketoacyl synthase, N-terminal domain
        "PF00141",  # Peroxidase
    ]
}

# ----------------------------
# HTTP utility
# ----------------------------
def http_get(url: str, params: Dict[str, Any] = None, retries: int = 4, sleep: float = 0.7):
    """Robust HTTP GET with retries."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers={"Accept": "application/json"}, timeout=30)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 502, 503, 504):
                time.sleep(sleep * (attempt + 1))
            else:
                time.sleep(sleep * (attempt + 1))
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(sleep * (attempt + 1))
                continue
            raise
    r.raise_for_status()
    return r

# ----------------------------
# Fetch motif details
# ----------------------------
def get_motif_details(pfam_id: str) -> Dict[str, Any]:
    """
    Fetch detailed information for a specific Pfam ID.
    """
    # Get entry metadata
    entry_url = f"https://www.ebi.ac.uk/interpro/api/entry/pfam/{pfam_id}/"
    try:
        r = http_get(entry_url)
        entry_data = r.json()
        metadata = entry_data.get("metadata", {})
    except Exception as e:
        print(f"  ⚠️  Error fetching {pfam_id}: {e}")
        return None
    
    # Get protein count
    protein_url = f"https://www.ebi.ac.uk/interpro/api/protein/uniprot/entry/pfam/{pfam_id}/"
    try:
        r = http_get(protein_url, params={"page_size": 1})
        protein_data = r.json()
        protein_count = protein_data.get("count", 0)
    except Exception:
        protein_count = 0
    
    # Extract name
    name_field = metadata.get("name", "")
    if isinstance(name_field, dict):
        name = name_field.get("name", "")
        short_name = name_field.get("short", "")
    else:
        name = name_field
        short_name = ""
    
    # Extract description
    desc_list = metadata.get("description", [])
    description = ""
    if desc_list and len(desc_list) > 0:
        text = desc_list[0].get("text", "")
        # Strip HTML tags
        import re
        description = re.sub('<[^<]+?>', '', text)[:200]
    
    return {
        "accession": pfam_id,
        "name": name,
        "short_name": short_name,
        "type": metadata.get("type", ""),
        "protein_count": protein_count,
        "description": description,
    }

# ----------------------------
# Main
# ----------------------------
def main():
    print("=" * 70)
    print("TCAV Curated Motif Selection")
    print("=" * 70)
    
    # Flatten all motif IDs
    all_motif_ids = []
    category_map = {}
    for category, ids in CURATED_MOTIFS.items():
        for pfam_id in ids:
            if pfam_id not in all_motif_ids:  # Avoid duplicates
                all_motif_ids.append(pfam_id)
                category_map[pfam_id] = category
    
    print(f"Total curated motifs: {len(all_motif_ids)}")
    print(f"Categories: {len(CURATED_MOTIFS)}")
    print()
    
    # Fetch details for each
    motifs = []
    failed = []
    
    for pfam_id in tqdm(all_motif_ids, desc="Fetching motif details"):
        details = get_motif_details(pfam_id)
        
        if details and details["protein_count"] > 0:
            details["category"] = category_map.get(pfam_id, "Unknown")
            motifs.append(details)
        else:
            failed.append(pfam_id)
        
        time.sleep(0.15)  # Be nice to API
    
    print(f"\n✓ Successfully fetched {len(motifs)} motifs")
    if failed:
        print(f"⚠️  Failed to fetch {len(failed)} motifs: {', '.join(failed[:10])}")
    
    # Print summary by category
    from collections import defaultdict
    by_category = defaultdict(list)
    for m in motifs:
        by_category[m["category"]].append(m)
    
    print(f"\nMotifs by category:")
    for cat, mots in sorted(by_category.items()):
        print(f"  {cat}: {len(mots)} motifs")
    
    if motifs:
        print(f"\nProtein count range: {min(m['protein_count'] for m in motifs):,} - {max(m['protein_count'] for m in motifs):,}")
    
    # Save to JSON
    output_file = "motifs_list.json"
    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "total_selected": len(motifs),
                "selection_strategy": "curated_diverse_families",
                "categories": list(CURATED_MOTIFS.keys())
            },
            "motifs": motifs
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved motif list to: {output_file}")
    
    # Show samples from each category
    print(f"\nSample motifs:")
    for cat in sorted(by_category.keys())[:3]:
        print(f"\n  {cat}:")
        for m in by_category[cat][:3]:
            print(f"    - {m['accession']}: {m['name'][:50]} ({m['protein_count']:,} proteins)")
    
    print("\n" + "=" * 70)
    print("Next step: Run batch_collect_data.py to collect training data")
    print("=" * 70)

if __name__ == "__main__":
    main()


