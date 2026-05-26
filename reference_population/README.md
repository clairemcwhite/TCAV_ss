# Reference Population

This directory contains the reference negative population used for training all CAVs.

## Files

- **`neg_embeddings.npy`** - Embeddings matrix for 10,000 random background proteins (shape: [10000, embedding_dim])
- **`neg_metadata.json`** - Metadata for each negative sample (accession, sequence length, window span)
- **`_neg_embed_info.json`** - Information about the source data and pooling strategy used

## Purpose

The reference negative population provides background embeddings that are used to train linear classifiers (CAVs) to distinguish motif-containing sequences from random proteins. These negatives are shared across all CAV training runs for consistency.

## Building Your Own Reference Population

If you need to create your own reference population:

1. Collect a large set of random protein sequences (e.g., 10,000) in FASTA format
2. Embed them using the same model and layer used for your positive examples:

```bash
python hf_embed_new.py \
    -f random_proteins.fasta \
    -o reference_population/neg_embeddings.pkl \
    -ss mean \
    -s \
    -l -11 \
    -m /path/to/ESMplusplus_large \
    -b 1 \
    --max_length 2048
```

3. Use the output embeddings as your reference negative set
4. Update `config.yaml` with the path to your new reference files

## Notes

- The layer used (-11, second-to-last) must match the layer used for all positive examples
- Typically 5,000-10,000 negatives provides sufficient background
- The same reference population can be used across multiple CAV training experiments
