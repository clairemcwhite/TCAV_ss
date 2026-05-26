#!/usr/bin/env python3
"""
Convert JSONL test data files to .span format.

The JSONL files contain ground truth annotations with start/end positions.
This script extracts those and creates tab-separated span files.

JSONL format:
    {"accession": "P12345", "ground_truth_annotations": [{"start": 10, "end": 50, ...}], ...}

Output .span format:
    accession    start    end

Usage:
    # Single file
    python scripts/jsonl_to_span.py test_data/PF00001/test_100.jsonl

    # Process entire directory
    python scripts/jsonl_to_span.py test_data/ --recursive

    # Custom output path
    python scripts/jsonl_to_span.py input.jsonl --output output.span
"""

import json
import argparse
from pathlib import Path
import sys


def convert_jsonl_to_span(jsonl_path, output_path=None):
    """
    Convert a JSONL file to span format.
    
    Args:
        jsonl_path: Path to input JSONL file
        output_path: Path to output span file (default: input_path with .span extension)
    
    Returns:
        Number of spans written
    """
    jsonl_path = Path(jsonl_path)
    
    if output_path is None:
        output_path = jsonl_path.with_suffix('.span')
    else:
        output_path = Path(output_path)
    
    span_count = 0
    
    with open(jsonl_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}", file=sys.stderr)
                continue
            
            # Extract accession
            accession = data.get('accession')
            if not accession:
                print(f"Warning: No accession at line {line_num}, skipping", file=sys.stderr)
                continue
            
            # Extract ground truth annotations
            annotations = data.get('ground_truth_annotations', [])
            
            if not annotations:
                # No annotations - write accession only (whole sequence)
                f_out.write(f"{accession}\n")
                span_count += 1
            else:
                # Write each annotation as a span
                for ann in annotations:
                    start = ann.get('start')
                    end = ann.get('end')
                    
                    if start is None or end is None:
                        print(f"Warning: Annotation missing start/end for {accession}, skipping", 
                              file=sys.stderr)
                        continue
                    
                    # Write tab-separated span (accession, start, end)
                    f_out.write(f"{accession}\t{start}\t{end}\n")
                    span_count += 1
    
    return span_count


def convert_jsonl_to_fasta(jsonl_path, output_path=None):
    """
    Convert a JSONL file to FASTA format.
    
    Args:
        jsonl_path: Path to input JSONL file
        output_path: Path to output FASTA file (default: input_path with .fasta extension)
    
    Returns:
        Number of sequences written
    """
    jsonl_path = Path(jsonl_path)
    
    if output_path is None:
        output_path = jsonl_path.with_suffix('.fasta')
    else:
        output_path = Path(output_path)
    
    seq_count = 0
    
    with open(jsonl_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}", file=sys.stderr)
                continue
            
            accession = data.get('accession')
            sequence = data.get('sequence')
            
            if not accession or not sequence:
                print(f"Warning: Missing accession or sequence at line {line_num}, skipping", 
                      file=sys.stderr)
                continue
            
            # Optional: add protein name and gene name to header
            protein_name = data.get('protein_name', '')
            gene_name = data.get('gene_name', '')
            
            header = f">{accession}"
            if protein_name:
                header += f" {protein_name}"
            if gene_name:
                header += f" GN={gene_name}"
            
            f_out.write(f"{header}\n{sequence}\n")
            seq_count += 1
    
    return seq_count


def process_directory(directory, recursive=False, format='span'):
    """
    Process all JSONL files in a directory.
    
    Args:
        directory: Path to directory
        recursive: Whether to search recursively
        format: Output format ('span' or 'fasta')
    """
    directory = Path(directory)
    
    if recursive:
        jsonl_files = directory.rglob('*.jsonl')
    else:
        jsonl_files = directory.glob('*.jsonl')
    
    jsonl_files = list(jsonl_files)
    
    if not jsonl_files:
        print(f"No .jsonl files found in {directory}", file=sys.stderr)
        return
    
    print(f"Found {len(jsonl_files)} JSONL files")
    
    for jsonl_file in jsonl_files:
        print(f"Converting {jsonl_file}...")
        
        if format == 'span':
            count = convert_jsonl_to_span(jsonl_file)
            print(f"  → Wrote {count} spans to {jsonl_file.with_suffix('.span')}")
        elif format == 'fasta':
            count = convert_jsonl_to_fasta(jsonl_file)
            print(f"  → Wrote {count} sequences to {jsonl_file.with_suffix('.fasta')}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert JSONL files to span or FASTA format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('input', help='Input JSONL file or directory')
    parser.add_argument('-o', '--output', help='Output file path (for single file conversion)')
    parser.add_argument('-r', '--recursive', action='store_true',
                        help='Process directory recursively')
    parser.add_argument('-f', '--format', choices=['span', 'fasta', 'both'], default='span',
                        help='Output format (default: span)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: {input_path} does not exist", file=sys.stderr)
        sys.exit(1)
    
    if input_path.is_dir():
        if args.output:
            print("Warning: --output is ignored for directory input", file=sys.stderr)
        
        if args.format == 'both':
            process_directory(input_path, args.recursive, 'span')
            process_directory(input_path, args.recursive, 'fasta')
        else:
            process_directory(input_path, args.recursive, args.format)
    
    elif input_path.is_file():
        if args.format in ('span', 'both'):
            count = convert_jsonl_to_span(input_path, args.output)
            output = args.output or input_path.with_suffix('.span')
            print(f"Wrote {count} spans to {output}")
        
        if args.format in ('fasta', 'both'):
            count = convert_jsonl_to_fasta(input_path)
            output = input_path.with_suffix('.fasta')
            print(f"Wrote {count} sequences to {output}")
    
    else:
        print(f"Error: {input_path} is neither a file nor a directory", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
