import requests
import argparse
import time

def fetch_uniprot_fasta(input_file, output_file, batch_size=400, delay=0.1):
    """
    Fetch UniProt FASTA sequences using the UniProt REST API.
    input_file: file with one UniProt ID per line
    output_file: path to output FASTA file
    batch_size: number of IDs per POST request
    delay: optional delay between requests to avoid rate limits
    """
    with open(input_file) as f:
        accessions = [line.strip() for line in f if line.strip()]

    fasta_sequences = []
    url = "https://rest.uniprot.org/uniprotkb/stream"

    for i in range(0, len(accessions), batch_size):
        batch = accessions[i:i + batch_size]
        query = " OR ".join(f"accession:{acc}" for acc in batch)
        params = {
            "query": query,
            "format": "fasta"
        }
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                text = response.text.strip()
                if text:
                    fasta_sequences.append(text)
            else:
                print(f"[Warning] Batch {i // batch_size + 1} failed: {response.status_code} — {response.text[:200]}")
        except requests.exceptions.RequestException as e:
            print(f"[Error] Batch {i // batch_size + 1} exception: {e}")

        time.sleep(delay)

    with open(output_file, "w") as f:
        f.write("\n\n".join(fasta_sequences) + "\n")

    print(f"Done. Wrote sequences for {len(accessions)} accessions to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch UniProt FASTA sequences via the REST API.")
    parser.add_argument("input", help="Input file with UniProt IDs (one per line).")
    parser.add_argument("output", help="Output FASTA file.")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size per request (default 400)")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay (seconds) between requests (default 0.1)")
    args = parser.parse_args()
    fetch_uniprot_fasta(args.input, args.output, args.batch_size, args.delay)
