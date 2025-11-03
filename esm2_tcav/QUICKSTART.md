# TCAV ESM2 Quick Reference

## 🚀 Getting Started (3 Steps)

### 1. Activate Environment
```bash
source /groups/clairemcwhite/envs/core_pkgs/bin/activate
cd /path/to/esm2_tcav
```

### 2. Run Smoke Test
```bash
python scripts/smoke_test.py --config config.yaml --n-samples 5
```
✅ **Must pass before running expensive jobs!**

### 3. Submit Pipeline
```bash
sbatch slurm/full_pipeline.slurm
```

---

## 📋 Common Commands

### Run Full Pipeline (Interactive)
```bash
python scripts/run_pipeline.py --config config.yaml
```

### Run Individual Steps
```bash
# Step 1: Embeddings (GPU required)
python scripts/run_embed.py --config config.yaml

# Step 2: Train CAVs (CPU only)
python scripts/run_train_cav.py --config config.yaml

# Step 3: Evaluate (CPU only)
python scripts/run_evaluate.py --config config.yaml
```

### SLURM Jobs
```bash
# Individual steps
sbatch slurm/embed.slurm          # ~2-4 hours, GPU
sbatch slurm/train_cav.slurm      # ~30 min, CPU
sbatch slurm/evaluate.slurm       # ~15 min, CPU

# Full pipeline
sbatch slurm/full_pipeline.slurm  # ~4-6 hours, GPU

# Detection
sbatch slurm/detect.slurm proteins.fasta
```

### Check Job Status
```bash
squeue -u $USER
tail -f outputs/logs/pipeline_*.out
```

---

## 🔄 Switching Models

**Edit `config.yaml`:**
```yaml
model:
  name: "esm2_t33_650M_UR50D"  # Change this line
```

**Available models:**
- `esm2_t6_8M_UR50D` (layers: 2, 4, 6) ← **Start here**
- `esm2_t12_35M_UR50D` (layers: 4, 8, 12)
- `esm2_t30_150M_UR50D` (layers: 10, 20, 30)
- `esm2_t33_650M_UR50D` (layers: 11, 22, 33) ← **Best accuracy**

Layers auto-update from `models/model_registry.yaml` - no code changes needed!

---

## 📊 Key Outputs

### Embeddings
```
outputs/embeddings/esm2_t6_8M_UR50D/
├── L2_pos.npy    # Positive embeddings
├── L2_neg.npy    # Negative embeddings
└── L2_meta.json  # Metadata
```

### CAVs
```
outputs/cavs/esm2_t6_8M_UR50D/
├── L2_concept_v1.npy       # Main CAV
├── L2_random_*_v1.npy      # 20 random CAVs
├── L2_scaler_v1.pkl        # Preprocessing
├── L2_report_v1.json       # Metrics
└── L2_manifest_v1.json     # File list
```

### Evaluation
```
outputs/evaluations/esm2_t6_8M_UR50D/
├── projection_eval.json       # All metrics
├── thresholds.json           # For detection
└── plots/
    ├── roc_pr_curves.png
    └── L2_random_comparison.png
```

### Detection
```
outputs/detections/protein_set/
├── predictions.json          # Per-protein results
└── detection_summary.json    # Stats
```

---

## 🔍 Interpreting Results

### CAV Report (`L2_report_v1.json`)
```json
{
  "concept_metrics": {
    "cv_auroc_mean": 0.94,     // ✓ Good if >0.85
    "train_auroc": 0.96
  },
  "random_cav_stats": {
    "auroc_mean": 0.51,        // ✓ Should be ~0.50
    "auroc_std": 0.03
  }
}
```

### Evaluation Results
```json
{
  "L2": {
    "auroc": 0.94,              // ✓ >0.90 is excellent
    "threshold": 0.352,
    "random_cav_comparison": {
      "is_significant": true,   // ✓ Must be true
      "p_value": 0.001
    }
  }
}
```

### Detection Output
```json
{
  "ensemble": {
    "has_ZnF": true,
    "consensus_window": [125, 166],
    "consensus_sequence": "CPECGKAF..."
  }
}
```

---

## 🐛 Troubleshooting

### Job Failed?
1. **Run smoke test first:**
   ```bash
   python scripts/smoke_test.py --config config.yaml
   ```

2. **Check logs:**
   ```bash
   ls -lht outputs/logs/  # Find latest log
   tail -50 outputs/logs/pipeline_*.err
   ```

3. **Verify environment:**
   ```bash
   python -c "import torch; print(torch.__version__)"
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Low AUROC?
- Check `L*_random_comparison.png` plots
- Try different layer: `--layer 4` or `--layer 6`
- Scale up model: edit `config.yaml` → `esm2_t33_650M`

### Indexing Errors?
- Check: `outputs/logs/indexing_corrections.jsonl`
- Auto-fixed by default (with BOS validation)

### Dimension Mismatch?
- Model registry validates on load
- Error will show: "Expected X, got Y"
- Check `models/model_registry.yaml`

---

## ⚙️ Configuration Tips

### Speed up for testing
```yaml
embedding:
  batch_size: 16          # Increase for faster GPU

cav:
  n_random_cavs: 10       # Reduce for faster training
  pca_dim: 64             # Smaller for speed
```

### Better accuracy
```yaml
cav:
  cv_folds: 10            # More robust CV
  pca_dim: 256            # More components
  n_random_cavs: 50       # Better null hypothesis
```

### Data efficiency
```yaml
evaluation:
  localization:
    top_k_windows: 1      # Save only top window
    save_heatmaps:
      top_iou: 3          # Fewer heatmaps
      bottom_iou: 3
```

---

## 📈 Expected Timeline

| Step | Time | Resource | Can Skip? |
|------|------|----------|-----------|
| Smoke test | 30s | Local/CPU | ❌ Never |
| Embeddings | 2-4h | GPU | After first run |
| CAV training | 30m | CPU | After first run |
| Evaluation | 15m | CPU | For reruns |
| Detection | 1-2h | GPU | As needed |

**Total first run:** ~4-6 hours (mostly embedding extraction)

---

## 🎯 Workflow Examples

### First Time Setup
```bash
# 1. Smoke test
python scripts/smoke_test.py --config config.yaml

# 2. Full pipeline
sbatch slurm/full_pipeline.slurm

# 3. Wait for completion, check results
tail -f outputs/logs/pipeline_*.out
```

### Trying Different Layers
```bash
# Already have embeddings, just retrain CAVs for layer 6
python scripts/run_train_cav.py --config config.yaml --layer 6
python scripts/run_evaluate.py --config config.yaml
```

### Scaling to Larger Model
```bash
# 1. Edit config.yaml: name: "esm2_t33_650M_UR50D"
# 2. Rerun (will use new layers automatically)
sbatch slurm/full_pipeline.slurm
```

### Production Detection
```bash
# 1. Make sure thresholds exist
ls outputs/evaluations/thresholds.json

# 2. Run detection
sbatch slurm/detect.slurm my_proteins.fasta

# 3. Check results
cat outputs/detections/my_proteins/detection_summary.json
```

---

## 📝 Quick Checks

✅ **Before submitting jobs:**
- [ ] Smoke test passed
- [ ] Config paths correct
- [ ] Model exists at HPC path
- [ ] Data files in place

✅ **After embeddings:**
- [ ] Files exist: `outputs/embeddings/MODEL/L*_pos.npy`
- [ ] No errors in: `outputs/logs/indexing_corrections.jsonl`
- [ ] Shapes look right: `(100, 320)` for t6_8M

✅ **After CAV training:**
- [ ] Concept AUROC > 0.85
- [ ] Random AUROC ≈ 0.50
- [ ] `is_significant: true` in reports

✅ **After evaluation:**
- [ ] `thresholds.json` exists
- [ ] Plots generated in `plots/`
- [ ] AUROC > 0.90 for at least one layer

---

## 🔗 File Locations on HPC

```bash
# Models
/groups/clairemcwhite/models/esm2_t6_8M/
/groups/clairemcwhite/models/esm2_t33_650M/

# Environment
/groups/clairemcwhite/envs/core_pkgs/

# Project
~/esm2_tcav/  # or wherever you cloned
```

---

**Need help? Check the full README.md or smoke_test.py output!**


