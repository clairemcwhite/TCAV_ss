## GSE111976 10x -> lean Geneformer-ready .h5ad
## Native R route using anndataR
## Expected files:
##   GSE111976_ct_endo_10x.rds
##   GSE111976_summary_10x_day_donor_ctype.csv
##   GSE111976_summary_10x_donor_phase.csv

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

pkgs <- c("Matrix", "biomaRt", "SingleCellExperiment", "S4Vectors", "anndataR", "rhdf5")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) {
    BiocManager::install(p, ask = FALSE, update = FALSE)
  }
}

library(Matrix)
library(biomaRt)
library(SingleCellExperiment)
library(S4Vectors)
library(anndataR)

## 1) Read inputs
obj <- readRDS("GSE111976_ct_endo_10x.rds")

meta_cell <- read.csv(
  "GSE111976_summary_10x_day_donor_ctype.csv",
  stringsAsFactors = FALSE,
  check.names = FALSE
)

meta_phase <- read.csv(
  "GSE111976_summary_10x_donor_phase.csv",
  stringsAsFactors = FALSE,
  check.names = FALSE
)

stopifnot(inherits(obj, "dgCMatrix"))

## 2) Clean and align metadata
meta_cell$cell_type <- gsub("epit/helia", "epithelia", meta_cell$cell_type)

cell_id_col <- if ("cell_name" %in% colnames(meta_cell)) "cell_name" else "X"
stopifnot(all(colnames(obj) %in% meta_cell[[cell_id_col]]))

meta_cell2 <- meta_cell[match(colnames(obj), meta_cell[[cell_id_col]]), , drop = FALSE]
stopifnot(all(meta_cell2[[cell_id_col]] == colnames(obj)))

donor_col_phase <- if ("donor" %in% colnames(meta_phase)) "donor" else grep("donor", colnames(meta_phase), ignore.case = TRUE, value = TRUE)[1]
phase_col <- if ("phase" %in% colnames(meta_phase)) "phase" else grep("phase", colnames(meta_phase), ignore.case = TRUE, value = TRUE)[1]

meta_phase2 <- meta_phase[, c(donor_col_phase, phase_col), drop = FALSE]
colnames(meta_phase2) <- c("donor", "phase")

meta2 <- merge(meta_cell2, meta_phase2, by = "donor", all.x = TRUE, sort = FALSE)
meta2 <- meta2[match(colnames(obj), meta2[[cell_id_col]]), , drop = FALSE]
stopifnot(all(meta2[[cell_id_col]] == colnames(obj)))

meta2$n_counts <- Matrix::colSums(obj)

## Keep only the obs columns you actually need
obs <- meta2[, c(cell_id_col, "day", "donor", "cell_type", "phase", "n_counts"), drop = FALSE]
rownames(obs) <- obs[[cell_id_col]]

## 3) Map gene symbols -> Ensembl IDs
cache_dir <- file.path(tempdir(), "biomart_cache")
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
Sys.setenv(BIOMART_CACHE = cache_dir)

genes <- rownames(obj)

mart <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")
mapping <- getBM(
  attributes = c("hgnc_symbol", "ensembl_gene_id"),
  filters = "hgnc_symbol",
  values = unique(genes),
  mart = mart,
  useCache = FALSE
)

mapping <- mapping[mapping$hgnc_symbol != "" & mapping$ensembl_gene_id != "", , drop = FALSE]
mapping <- mapping[!duplicated(mapping$hgnc_symbol), , drop = FALSE]

keep <- rownames(obj) %in% mapping$hgnc_symbol
obj2 <- obj[keep, , drop = FALSE]

orig_symbols <- rownames(obj2)
ens_ids <- mapping$ensembl_gene_id[match(orig_symbols, mapping$hgnc_symbol)]

valid <- !is.na(ens_ids) & ens_ids != ""
obj2 <- obj2[valid, , drop = FALSE]
orig_symbols <- orig_symbols[valid]
ens_ids <- ens_ids[valid]

## Collapse duplicated Ensembl IDs
if (anyDuplicated(ens_ids) > 0) {
  idx_list <- split(seq_along(ens_ids), ens_ids)

  collapsed <- lapply(idx_list, function(idx) {
    if (length(idx) == 1) {
      obj2[idx, , drop = FALSE]
    } else {
      Matrix(
        matrix(Matrix::colSums(obj2[idx, , drop = FALSE]), nrow = 1),
        sparse = TRUE
      )
    }
  })

  obj2 <- do.call(rbind, collapsed)
  rownames(obj2) <- names(idx_list)

  symbol_by_ens <- vapply(
    idx_list,
    function(idx) paste(unique(orig_symbols[idx]), collapse = ";"),
    character(1)
  )
} else {
  rownames(obj2) <- ens_ids
  symbol_by_ens <- stats::setNames(orig_symbols, rownames(obj2))
}

var <- data.frame(
  ensembl_id = rownames(obj2),
  gene_symbol = unname(symbol_by_ens[rownames(obj2)]),
  row.names = rownames(obj2),
  stringsAsFactors = FALSE
)

## 4) Build a minimal SCE
sce <- SingleCellExperiment(
  assays = list(counts = obj2),
  colData = S4Vectors::DataFrame(obs),
  rowData = S4Vectors::DataFrame(var)
)

out_file = "GSE111976_10x_geneformer_ready.h5ad"
if (file.exists(out_file)) {
  message("Removing existing file: ", out_file)
  file.remove(out_file)
}

## 5) Write directly from SCE
## This avoids keeping a separate full AnnData object around in your script.
write_h5ad(
  sce,
  out_file,
  x_mapping = "counts",
  layers_mapping = FALSE,
  obs_mapping = TRUE,
  var_mapping = TRUE,
  obsm_mapping = FALSE,
  varm_mapping = FALSE,
  obsp_mapping = FALSE,
  varp_mapping = FALSE,
  uns_mapping = FALSE
)

cat("Wrote GSE111976_10x_geneformer_ready.h5ad\n")
cat("Cells:", ncol(sce), "\n")
cat("Genes:", nrow(sce), "\n")
