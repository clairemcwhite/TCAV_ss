# Motif Detection Pipeline with ESM2-TCAV

This directory contains all scripts for the complete motif detection pipeline using ESM2 embeddings and TCAV (Testing with Concept Activation Vectors).

## Overview

The pipeline consists of three main phases:
1. **Data Collection & CAV Training**: Collect training/test data and train CAV vectors for each motif
2. **Test Detection**: Run motif detection on test proteins using trained CAVs
3. **Post-Processing & Evaluation**: Clean predictions, expand intervals, and evaluate performance

---

## Directory Structure

```
final_content/
├── esm2_tcav/              # Core TCAV library and utilities
│   ├── src/                # Source code (embed, train_cav, detect, evaluate)
│   ├── scripts/            # Wrapper scripts for pipeline steps
│   └── slurm/              # SLURM job templates for HPC
│
├── Phase 1: Data Collection & Training
│   ├── fetch_curated_motifs.py       # Download motif data from Pfam/UniProt
│   ├── batch_collect_data.py         # Collect positive/negative samples for motifs
│   ├── check_window_sizes.py         # Analyze optimal window sizes per motif
│   ├── update_windows.py             # Update window lengths in metadata
│   ├── batch_train_cavs.py           # Train CAV vectors for all motifs
│
├── Phase 2: Test Detection
│   ├── collect_test_data.py          # Collect test proteins (held-out set)
│   ├── create_small_test_set.py      # Create smaller test subset
│   ├── run_test_detection_onthefly.py # Run detection with on-the-fly embedding
│   ├── batch_detect_all_motifs.py    # Batch detection (pre-computed embeddings)
│
├── Phase 3: Post-Processing & Evaluation
│   ├── preprocess_detections.py      # Remove duplicates/overlaps (NMS)
│   ├── expand_intervals.py           # Greedy expansion to maximize scores
│   └── evaluate_metrics.py           # Compute metrics (P/R/F1, IoU, etc.)
│
└── README.md                          # This file
```

---

## Phase 1: Data Collection & CAV Training

### 1.1 Fetch Motif Data
```bash
python fetch_curated_motifs.py
```
Downloads curated motif families from Pfam and protein sequences from UniProt.

### 1.2 Collect Training Data
```bash
python batch_collect_data.py \
    --data-dir ./tcav_data \
    --motif-list <path_to_motif_list.txt> \
    --samples-per-motif 100
```
Collects positive (with motif) and negative (without motif) protein samples for each motif.

### 1.3 Check and Update Window Sizes
```bash
# Analyze optimal window sizes
python check_window_sizes.py --data-dir ./tcav_data

# Update metadata with optimal windows
python update_windows.py \
    --data-dir ./tcav_data \
    --window-sizes <path_to_window_sizes.json>
```

### 1.4 Train CAV Vectors
```bash
python batch_train_cavs.py \
    --data-dir ./tcav_data \
    --output-dir ./tcav_outputs_650m \
    --model-name esm2_t33_650M_UR50D \
    --layers 22 \
    --device cuda
```
Trains TCAV vectors for all motifs at specified layer(s).

**Output**: `tcav_outputs_650m/cavs/{motif_id}/L{layer}_concept_v1.npy` and scalers

---

## Phase 2: Test Detection

### 2.1 Collect Test Data
```bash
python collect_test_data.py
```
Collects held-out test proteins (not in training set) with ground truth annotations.

**Output**: `test_data/{motif_id}/test_100.jsonl`

### 2.2 (Optional) Create Small Test Set
```bash
python create_small_test_set.py \
    --input-dir ./test_data \
    --output-dir ./test_data_small \
    --n-samples 10 \
    --max-length 1000
```

### 2.3 Run Detection (On-the-Fly Embedding - Recommended)
```bash
python run_test_detection_onthefly.py \
    --test-data-dir ./test_data \
    --tcav-dir ./tcav_outputs_650m \
    --data-dir ./tcav_data \
    --model-name esm2_t33_650M_UR50D \
    --layers 22 \
    --rank-by score \
    --rank-by-layer 22 \
    --topk 20 \
    --device cuda \
    --output-file ./test_detection_results.json
```

**Alternative: Pre-computed Embeddings**
```bash
# Step 1: Generate embeddings (if needed)
python esm2_tcav/scripts/run_embed.py \
    --test-data-dir ./test_data \
    --output-dir ./test_data/embeddings \
    --model-name esm2_t33_650M_UR50D \
    --layers 22 \
    --device cuda

# Step 2: Run detection on cached embeddings
python batch_detect_all_motifs.py \
    --embeddings-dir ./test_data/embeddings \
    --tcav-dir ./tcav_outputs_650m \
    --layers 22 \
    --output-file ./test_detection_results.json
```

**Output**: `test_detection_results.json` with all predictions

---

## Phase 3: Post-Processing & Evaluation

### 3.1 Remove Duplicates/Overlaps (NMS)
```bash
python preprocess_detections.py \
    --input test_detection_results.json \
    --output test_detection_results_cleaned.json \
    --iou-threshold 0.3
```

**Parameters**:
- `--iou-threshold`: IoU overlap threshold (0.3 = aggressive, 0.5 = conservative)
- `--filter-negative`: Optional flag to remove negative scores (default: keep all)

**Output**: `test_detection_results_cleaned.json`

### 3.2 Expand Intervals (Greedy Optimization)
```bash
python expand_intervals.py \
    --detection-results test_detection_results_cleaned.json \
    --test-data-dir ./test_data \
    --tcav-dir ./tcav_outputs_650m \
    --model-name esm2_t33_650M_UR50D \
    --layer 22 \
    --expansion-step 5 \
    --device cuda \
    --output expanded_predictions.json
```

**Parameters**:
- `--expansion-step`: Residues to expand per iteration (5 = balanced, 1 = fine-grain, 10 = coarse)
- `--max-length`: Maximum interval size (default: 500)
- `--score-threshold`: Minimum improvement to continue (default: 0.01)
- `--max-iterations`: Stop after N iterations (default: 50)

**Output**: `expanded_predictions.json` with optimized intervals

### 3.3 Evaluate Performance
```bash
python evaluate_metrics.py \
    --detection-results expanded_predictions.json \
    --output-dir ./evaluation_results \
    --k-max 20 \
    --overlap-thresholds 80.0 85.0 90.0 95.0 100.0
```

**Output**: Metrics tables (Precision, Recall, F1, IoU) per motif and overall

---

## HPC Commands (SLURM Examples)

### Full Pipeline on HPC

**1. Train CAVs**
```bash
sbatch esm2_tcav/slurm/full_pipeline.slurm
# Or individual steps:
sbatch esm2_tcav/slurm/embed.slurm
sbatch esm2_tcav/slurm/train_cav.slurm
```

**2. Run Detection**
```bash
python run_test_detection_onthefly.py \
    --test-data-dir ./test_data_small \
    --tcav-dir ./tcav_outputs_650m \
    --data-dir ./tcav_data \
    --model-name esm2_t33_650M_UR50D \
    --layers 22 \
    --rank-by score \
    --device cuda \
    --output-file test_detection_results.json
```

**3. Post-Process & Evaluate**
```bash
# Preprocess
python preprocess_detections.py \
    --input test_detection_results.json \
    --output test_detection_results_cleaned.json \
    --iou-threshold 0.3

# Expand
python expand_intervals.py \
    --detection-results test_detection_results_cleaned.json \
    --test-data-dir ./test_data_small \
    --tcav-dir ./tcav_outputs_650m \
    --model-name esm2_t33_650M_UR50D \
    --layer 22 \
    --device cuda \
    --output expanded_predictions.json

# Evaluate
python evaluate_metrics.py \
    --detection-results expanded_predictions.json \
    --output-dir ./evaluation_results
```

---

## Key Files & Formats

### Training Data (`tcav_data/{motif_id}/`)
- `pos_100.jsonl`: Positive samples (proteins with motif)
- `neg_100.jsonl`: Negative samples (proteins without motif)
- `metadata.json`: Motif metadata (window_length, description, etc.)

### Test Data (`test_data/{motif_id}/`)
- `test_100.jsonl`: Test proteins with ground truth annotations

### Detection Results
```json
{
  "n_proteins": 356,
  "n_motifs": 72,
  "topk": 20,
  "results": [
    {
      "accession": "P12345",
      "predictions": [
        {
          "motif_id": "PF00001",
          "span": [10, 100],
          "ranking_score": 5.23,
          "per_layer_scores": {"L22": 5.23}
        }
      ],
      "ground_truth": [
        {
          "motif_id": "PF00001",
          "start": 15,
          "end": 95
        }
      ]
    }
  ]
}
```

---

## Dependencies

See `esm2_tcav/requirements.txt` for full list. Key dependencies:
- `torch`
- `transformers` (ESM2 models)
- `scikit-learn` (TCAV training)
- `numpy`, `pandas`
- `tqdm`, `joblib`

Install:
```bash
pip install -r esm2_tcav/requirements.txt
```

---

## Notes

- **Layer Selection**: Layer 22 works well for ESM2-650M. Experiment with different layers if needed.
- **Window Sizes**: Each motif has an optimal window size based on its typical length.
- **IoU Threshold**: 0.3 is aggressive (removes more overlaps), 0.5 is conservative.
- **Expansion**: Greedy expansion refines boundaries to maximize TCAV scores.
- **Device**: Use `cuda` for GPU acceleration (highly recommended).

---

## Citation

If you use this pipeline, please cite:
- ESM2: https://github.com/facebookresearch/esm
- TCAV: https://arxiv.org/abs/1711.11279

---

## Troubleshooting

**Out of Memory**: Reduce batch size, use smaller model, or switch to CPU
**No CAV artifacts found**: Check that `batch_train_cavs.py` completed successfully
**Sequence not found**: Ensure `collect_test_data.py` was run before detection
**Slow expansion**: Use larger `--expansion-step` or reduce `--max-iterations`

