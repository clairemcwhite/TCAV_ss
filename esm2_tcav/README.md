# TCAV + ESM2 for Zinc Finger Motif Detection

End-to-end pipeline for detecting zinc finger (C2H2) motifs in proteins using **Concept Activation Vectors (CAVs)** on **ESM-2 embeddings**.

## 🎯 Features

**Core 8 Production Features:**
1. ✅ **Config-driven layer taps** - Switch models without code edits
2. ✅ **Indexing guardrails** - BOS token validation with auto-correction
3. ✅ **Threshold registry** - Systematic threshold selection & storage
4. ✅ **Random CAV comparison** - Statistical validation vs null hypothesis
5. ✅ **Smoke tests** - Fast health checks before HPC jobs
6. ✅ **Localization analysis** - Smart heatmap selection (top/bottom/random)
7. ✅ **Versioned artifacts** - Reproducible CAVs with manifests
8. ✅ **Enhanced model registry** - Dimension validation & version tracking

## 📁 Project Structure

```
esm2_tcav/
├── config.yaml                 # Main configuration
├── models/model_registry.yaml  # Model specifications
├── data/                       # Input data (JSONL/FASTA)
├── src/                        # Core modules
│   ├── embed.py               # Embedding extraction
│   ├── train_cav.py           # CAV training
│   ├── evaluate.py            # Evaluation
│   ├── detect.py              # Detection
│   └── utils/                 # Utilities
├── scripts/                   # Executable scripts
│   ├── run_pipeline.py        # Full pipeline
│   ├── run_embed.py           # Standalone embedding
│   ├── run_train_cav.py       # Standalone training
│   ├── run_evaluate.py        # Standalone evaluation
│   ├── run_detect.py          # Standalone detection
│   └── smoke_test.py          # Health check
├── slurm/                     # HPC job scripts
│   ├── embed.slurm
│   ├── train_cav.slurm
│   ├── evaluate.slurm
│   ├── full_pipeline.slurm
│   └── detect.slurm
└── outputs/                   # Results
    ├── embeddings/
    ├── cavs/
    ├── evaluations/
    └── detections/
```

## 🚀 Quick Start

### 1. Installation

```bash
# Activate conda environment (on HPC)
source /groups/clairemcwhite/envs/core_pkgs/bin/activate

# Or create new environment
conda create -n tcav python=3.9
conda activate tcav
pip install -r requirements.txt
```

### 2. Configuration

Edit `config.yaml` to set:
- Model name (`esm2_t6_8M_UR50D` to start)
- Data paths (JSONL files with positive/negative samples)
- Layer extraction settings (read from `model_registry.yaml`)

Edit `models/model_registry.yaml` to set:
- Local model paths on HPC (`/groups/clairemcwhite/models/...`)

### 3. Smoke Test (Recommended!)

**Run before submitting expensive jobs:**

```bash
python scripts/smoke_test.py --config config.yaml --n-samples 5
```

This runs the full pipeline on 5 samples in ~30 seconds to validate:
- ✓ Model loading
- ✓ Embedding extraction
- ✓ CAV training
- ✓ Evaluation
- ✓ Artifact creation

### 4. Run Full Pipeline

**Option A: Interactive (for testing)**
```bash
python scripts/run_pipeline.py --config config.yaml
```

**Option B: SLURM (for production)**
```bash
sbatch slurm/full_pipeline.slurm
```

**Option C: Step-by-step**
```bash
# Step 1: Extract embeddings (requires GPU)
sbatch slurm/embed.slurm

# Step 2: Train CAVs (CPU only)
sbatch slurm/train_cav.slurm

# Step 3: Evaluate (CPU only)
sbatch slurm/evaluate.slurm
```

### 5. Detect in Unannotated Proteins

```bash
# Interactive
python scripts/run_detect.py --config config.yaml --input proteins.fasta

# SLURM
sbatch slurm/detect.slurm proteins.fasta
```

## 📊 Outputs

### Embeddings
```
outputs/embeddings/esm2_t6_8M_UR50D/
├── L2_pos.npy          # Positive embeddings
├── L2_neg.npy          # Negative embeddings
├── L2_meta.json        # Metadata
└── ... (same for L4, L6)
```

### CAVs
```
outputs/cavs/esm2_t6_8M_UR50D/
├── L2_concept_v1.npy           # Main CAV vector
├── L2_random_00_v1.npy         # Random CAV #1
├── ...
├── L2_random_19_v1.npy         # Random CAV #20
├── L2_scaler_v1.pkl            # Fitted scaler
├── L2_pca_v1.pkl               # Fitted PCA
├── L2_report_v1.json           # Training metrics
└── L2_manifest_v1.json         # Artifact manifest
```

### Evaluation
```
outputs/evaluations/esm2_t6_8M_UR50D/
├── projection_eval.json        # All metrics
├── thresholds.json             # Threshold registry
└── plots/
    ├── roc_pr_curves.png
    ├── auroc_by_layer.png
    ├── L2_random_comparison.png
    └── ...
```

### Detection
```
outputs/detections/unannotated_proteins/
├── predictions.json            # Per-protein results
└── detection_summary.json      # Summary stats
```

## 🔧 Key Configuration Options

### Model Selection
```yaml
model:
  name: "esm2_t6_8M_UR50D"  # Start small, scale up
  # Options: esm2_t6_8M, esm2_t12_35M, esm2_t30_150M, esm2_t33_650M
```

### Layer Extraction (Auto from Registry)
```yaml
# Defined in model_registry.yaml
esm2_t6_8M_UR50D:
  layers_to_extract: [2, 4, 6]
esm2_t33_650M_UR50D:
  layers_to_extract: [11, 22, 33]
```

### Random CAVs
```yaml
cav:
  n_random_cavs: 20
  random_mode: "label_shuffle"  # or "feature_permute", "gaussian_noise"
```

### Threshold Selection
```yaml
evaluation:
  thresholding:
    method: "f1_max"  # or "precision_at_recall_90", "fpr_0.05"
```

## 📈 Expected Performance

**With 100 pos / 100 neg samples:**
- **AUROC:** >0.90 (vs ~0.50 for random CAVs)
- **Localization IOU:** >0.75
- **Recall@1:** >0.80 (top window overlaps annotated motif)

## 🔄 Scaling Strategy

1. **Pilot:** esm2_t6_8M (320 dim, layers 2/4/6) - **START HERE**
2. **Intermediate:** esm2_t12_35M (480 dim, layers 4/8/12)
3. **Production:** esm2_t33_650M (1280 dim, layers 11/22/33) - best accuracy

Just change `model.name` in `config.yaml` - everything else auto-updates!

## 🐛 Troubleshooting

### Indexing errors
✓ Automatic validation enabled by default
✓ Check `outputs/logs/indexing_corrections.jsonl` for fixes

### Dimension mismatches
✓ Model registry validates `hidden_size` on load
✓ Error will show expected vs actual dimensions

### Low AUROC
- Check random CAV comparison plot
- Verify data labels are correct
- Try different layers or larger model

### SLURM job failures
- Run smoke test first: `python scripts/smoke_test.py`
- Check logs: `outputs/logs/`
- Verify conda environment: `source /groups/clairemcwhite/envs/core_pkgs/bin/activate`

## 📚 Data Format

### Input JSONL
```json
{
  "accession": "A0A015IIP1",
  "sequence": "MSSNNAPCNKFECKI...",
  "window_span_0based_halfopen": [0, 41],
  "set": "positive",
  "domain_source": "PF00096"
}
```

### Detection Output
```json
{
  "accession": "Q12345",
  "ensemble": {
    "has_ZnF": true,
    "consensus_window": [125, 166],
    "consensus_sequence": "CPECGKAF..."
  },
  "detections_by_layer": {
    "L4": {
      "top_window": {
        "span_0based": [125, 166],
        "projection_score": 0.82,
        "confidence": "high"
      }
    }
  }
}
```

## 🔬 Understanding Results

### CAV Training Report
```json
{
  "concept_metrics": {
    "cv_auroc_mean": 0.94,
    "train_auroc": 0.96,
    "pca_variance_explained": 0.97
  },
  "random_cav_stats": {
    "auroc_mean": 0.51,
    "auroc_std": 0.03
  }
}
```

**Good signs:**
- Concept AUROC >> 0.5
- Random AUROC ≈ 0.5 (null hypothesis)
- High PCA variance (>95% with few components)

### Threshold Registry
```json
{
  "L4": {
    "threshold": 0.352,
    "method": "f1_max",
    "auroc": 0.94,
    "f1_score": 0.89
  }
}
```

Used by `detect.py` for predictions.

## 📝 Citation

If you use this pipeline, please cite:
- ESM-2: [Lin et al. 2022](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1)
- TCAV: [Kim et al. 2018](https://arxiv.org/abs/1711.11279)

## 📧 Contact

For questions about this pipeline, contact [your contact info].

## 🔗 Related Resources

- ESM repository: https://github.com/facebookresearch/esm
- TCAV paper: https://arxiv.org/abs/1711.11279
- Pfam PF00096: https://www.ebi.ac.uk/interpro/entry/pfam/PF00096/

---

**Happy motif hunting! 🧬🔬**


