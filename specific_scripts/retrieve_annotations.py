import requests
import argparse
import time
from pathlib import Path

# Maps CLI --feature choice -> (UniProt field name, feature type string in JSON response)
FEATURE_CONFIG = {
    "transmem":     ("ft_transmem", "TRANSMEM"),
    "signal":       ("ft_signal",   "SIGNAL"),
    "topo_dom":     ("ft_topo_dom", "TOPO_DOM"),
    "active_site":  ("ft_act_site", "ACT_SITE"),
    "binding_site": ("ft_binding",  "BINDING"),
    "disulfide":    ("ft_disulfid", "DISULFID"),
    "ptm":          ("ft_mod_res",  "MOD_RES"),
    "glycosylation":("ft_carbohyd", "CARBOHYD"),
}


def fetch_uniprot_annotations(input_file, output_file, feature, batch_size=100, delay=0.1):
    """
    Fetch UniProt structural/functional annotations for a list of accessions.

    Coordinates in output are 0-based half-open [start, end) — i.e. converted
    from UniProt's 1-based inclusive format — so the output is directly usable
    as a spans file for prepare_embeddings.py.

    input_file:  file with one UniProt accession per line
    output_file: path to output TSV (header: accession, feature_type, start, end)
    feature:     annotation type to retrieve (see FEATURE_CONFIG keys)
    batch_size:  number of accessions per API request
    delay:       seconds to wait between requests
    """
    ft_field, feat_type = FEATURE_CONFIG[feature]

    with open(input_file) as f:
        accessions = [line.strip() for line in f if line.strip()]

    url = "https://rest.uniprot.org/uniprotkb/search"
    rows = []

    for i in range(0, len(accessions), batch_size):
        batch = accessions[i:i + batch_size]
        query = " OR ".join(f"accession:{acc}" for acc in batch)
        params = {
            "query": query,
            "fields": f"accession,{ft_field}",
            "format": "json",
            "size": batch_size,
        }
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                for entry in data.get("results", []):
                    acc = entry["primaryAccession"]
                    for feat in entry.get("features", []):
                        if feat["type"] == feat_type:
                            # UniProt: 1-based inclusive -> 0-based half-open
                            start = feat["location"]["start"]["value"] - 1
                            end   = feat["location"]["end"]["value"]
                            rows.append((acc, feat_type, start, end))
            else:
                print(f"[Warning] Batch {i // batch_size + 1} failed: "
                      f"{response.status_code} — {response.text[:200]}")
        except requests.exceptions.RequestException as e:
            print(f"[Error] Batch {i // batch_size + 1} exception: {e}")

        time.sleep(delay)

    with open(output_file, "w") as f:
        f.write("accession\tfeature_type\tstart\tend\n")
        for acc, ftype, start, end in rows:
            f.write(f"{acc}\t{ftype}\t{start}\t{end}\n")

    n_with_features = len({r[0] for r in rows})
    print(f"Done. Found {len(rows)} {feat_type} annotations across "
          f"{n_with_features}/{len(accessions)} accessions.")
    print(f"Coordinates are 0-based half-open. Written to: {output_file}")


def _add_0idx_suffix(path_str):
    """Insert _0idx before the file extension: foo.tsv -> foo_0idx.tsv"""
    p = Path(path_str)
    return str(p.with_stem(p.stem + "_0idx")) if p.suffix else path_str + "_0idx"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch UniProt annotations for a list of accessions. "
                    "Output coordinates are 0-based half-open."
    )
    parser.add_argument("input",
                        help="Input file with UniProt accessions (one per line).")
    parser.add_argument("output",
                        help="Output TSV file. '_0idx' is appended before the "
                             "extension to indicate 0-based coordinates.")
    parser.add_argument("--feature", required=True,
                        choices=list(FEATURE_CONFIG.keys()),
                        help="Annotation type to retrieve.")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Accessions per API request (default: 100).")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Delay in seconds between requests (default: 0.1).")
    args = parser.parse_args()

    output_path = _add_0idx_suffix(args.output)
    fetch_uniprot_annotations(args.input, output_path, args.feature,
                              args.batch_size, args.delay)
