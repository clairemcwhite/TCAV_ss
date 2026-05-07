#!/usr/bin/env python3
"""
Convert a global_pca_v1.pkl {scaler, pca} bundle into the {pcamatrix, bias}
format expected by hf_embed_new.py --aa_pcamatrix_pkl.

The two representations are mathematically identical:
    x @ pcamatrix + bias  ==  pca.transform(scaler.transform(x))

Usage
-----
python specific_scripts/convert_pca_bundle.py \\
    --pca-pkl reference_population/global_pca_v1.pkl \\
    --out     reference_population/global_pca_v1_matrix.pkl
"""

import argparse
import pickle
import numpy as np
import joblib

def main():
    parser = argparse.ArgumentParser(
        description="Convert scaler+PCA bundle to pcamatrix+bias format."
    )
    parser.add_argument('--pca-pkl', required=True,
                        help='Input bundle: reference_population/global_pca_v1.pkl')
    parser.add_argument('--out', required=True,
                        help='Output pkl with pcamatrix and bias keys.')
    args = parser.parse_args()

    bundle = joblib.load(args.pca_pkl)
    scaler = bundle['scaler']
    pca    = bundle['pca']

    # Collapse scaler + PCA into a single affine transform:
    #   x_scaled = (x - μ_s) / σ_s
    #   x_pca    = (x_scaled - μ_p) @ V^T
    # Combined: x_pca = x @ pcamatrix + bias
    pcamatrix = (pca.components_ / scaler.scale_).T          # (d_raw, d_pca)
    bias      = (-scaler.mean_ / scaler.scale_ - pca.mean_) @ pca.components_.T  # (d_pca,)

    with open(args.out, 'wb') as f:
        pickle.dump({'pcamatrix': pcamatrix, 'bias': bias}, f, protocol=4)

    print(f"Saved to {args.out}  (pcamatrix: {pcamatrix.shape}, bias: {bias.shape})")

if __name__ == '__main__':
    main()
