## Convert count matrices -> lean Geneformer-ready .h5ad
## Native R route using anndataR
##
## Supports three modes:
##   --mode 10x   (default) : reads RDS + 10x metadata CSVs
##   --mode ct              : reads genes x cells CSV + C1 metadata CSVs
##   --mode ncbi            : reads NCBI bulk RNA-seq TPM/counts TSV +
##                            NCBI gene annotation TSV (Entrez -> Ensembl)
##
## Usage:
##   Rscript convert_mat_to_h5af.R --mode 10x \
##       --mat   data/GSE111976_ct_endo_10x.rds \
##       --meta  data/GSE111976_summary_10x_day_donor_ctype.csv \
##       --phase data/GSE111976_summary_10x_donor_phase.csv \
##       --out   data/GSE111976_10x_geneformer_ready.h5ad
##
##   Rscript convert_mat_to_h5af.R --mode ct \
##       --mat   data/GSE111976_ct.csv \
##       --meta  data/GSE111976_summary_C1_day_donor_ctype.csv \
##       --phase data/GSE111976_summary_C1_donor_phase.csv \
##       --out   data/GSE111976_C1_geneformer_ready.h5ad
##
##   Rscript convert_mat_to_h5af.R --mode ncbi \
##       --mat   data/GSE226870_norm_counts_TPM_GRCh38.p13_NCBI.tsv.gz \
##       --annot data/Human.GRCh38.p13.annot.tsv.gz \
##       --out   data/GSE226870_TPM_geneformer_ready.h5ad

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

pkgs <- c("Matrix", "biomaRt", "SingleCellExperiment", "S4Vectors", "anndataR", "rhdf5", "optparse")
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
library(optparse)

## ---- Parse arguments -------------------------------------------------------
option_list <- list(
  make_option("--mode",  type="character", default="10x",
              help="Input mode: '10x' (RDS), 'ct' (genes x cells CSV), 'ncbi' (NCBI TSV), or 'microarray' (gene x sample CSV, gene symbols as rownames) [default: 10x]"),
  make_option("--mat",   type="character", help="Count matrix file (RDS, CSV, or TSV/TSV.gz)"),
  make_option("--meta",  type="character", help="Per-cell metadata CSV (10x and ct modes only)"),
  make_option("--phase", type="character", help="Per-donor phase CSV (10x and ct modes only)"),
  make_option("--annot", type="character", help="NCBI gene annotation TSV or TSV.gz (ncbi mode, optional; uses biomaRt if omitted)"),
  make_option("--out",   type="character", help="Output .h5ad path")
)
opt <- parse_args(OptionParser(option_list=option_list))

## Defaults for backward compatibility (no args = original 10x behaviour)
if (is.null(opt$mat))   opt$mat   <- "GSE111976_ct_endo_10x.rds"
if (is.null(opt$meta))  opt$meta  <- "GSE111976_summary_10x_day_donor_ctype.csv"
if (is.null(opt$phase)) opt$phase <- "GSE111976_summary_10x_donor_phase.csv"
if (is.null(opt$out))   opt$out   <- "GSE111976_10x_geneformer_ready.h5ad"

cat("Mode:", opt$mode, "\n")
cat("Matrix:", opt$mat, "\n")

## ===========================================================================
## MODE: ncbi
## NCBI bulk RNA-seq TSV (samples as columns, Entrez GeneIDs as rows)
## Gene mapping: Entrez -> Ensembl via NCBI annotation file (no biomaRt)
## obs: sample IDs only (no day/donor/phase metadata)
## ===========================================================================
if (opt$mode == "ncbi") {

  ## 1) Read the count/TPM matrix ------------------------------------------------
  cat("Reading NCBI matrix (this may take a moment)...\n")
  if (requireNamespace("data.table", quietly = TRUE)) {
    library(data.table)
    dt         <- data.table::fread(opt$mat, header = TRUE, data.table = FALSE)
  } else {
    dt <- read.table(opt$mat, header = TRUE, sep = "\t",
                     check.names = FALSE, stringsAsFactors = FALSE)
  }

  ## First column is GeneID (Entrez integer), remainder are samples
  entrez_ids  <- as.character(dt[, 1])
  sample_ids  <- colnames(dt)[-1]
  mat         <- as.matrix(dt[, -1, drop = FALSE])
  rownames(mat) <- entrez_ids
  colnames(mat) <- sample_ids

  cat("  Loaded:", nrow(mat), "genes x", ncol(mat), "samples\n")

  ## 2) Entrez -> Ensembl mapping (annotation file if given, else biomaRt) ------
  if (!is.null(opt$annot)) {
    cat("Reading NCBI annotation file:", opt$annot, "\n")
    if (requireNamespace("data.table", quietly = TRUE)) {
      annot <- data.table::fread(opt$annot, header = TRUE, data.table = FALSE,
                                 sep = "\t", quote = "")
    } else {
      annot <- read.table(opt$annot, header = TRUE, sep = "\t",
                          check.names = FALSE, stringsAsFactors = FALSE,
                          quote = "", comment.char = "")
    }

    ## Column names may start with '#' in some releases
    colnames(annot)[1] <- sub("^#*", "", colnames(annot)[1])

    ## Extract Ensembl ID from the dbXrefs column ("...|Ensembl:ENSGXXX|...")
    has_ensembl      <- grepl("Ensembl:ENSG", annot$dbXrefs)
    annot$ensembl_id <- NA_character_
    annot$ensembl_id[has_ensembl] <- sub(
      ".*Ensembl:(ENSG[0-9]+).*", "\\1", annot$dbXrefs[has_ensembl]
    )

    annot2 <- annot[!is.na(annot$ensembl_id) & annot$ensembl_id != "", , drop = FALSE]
    annot2 <- annot2[!duplicated(annot2$GeneID), , drop = FALSE]
    cat("  Annotation:", nrow(annot2), "Entrez IDs mapped to Ensembl\n")

    entrez_to_ens    <- stats::setNames(annot2$ensembl_id, as.character(annot2$GeneID))
    entrez_to_symbol <- stats::setNames(annot2$Symbol,     as.character(annot2$GeneID))

  } else {
    cat("No --annot provided; querying biomaRt for Entrez -> Ensembl mapping...\n")
    cache_dir <- file.path(tempdir(), "biomart_cache")
    dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
    Sys.setenv(BIOMART_CACHE = cache_dir)

    mart    <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")
    mapping <- getBM(
      attributes = c("entrezgene_id", "ensembl_gene_id", "hgnc_symbol"),
      filters    = "entrezgene_id",
      values     = unique(entrez_ids),
      mart       = mart,
      useCache   = FALSE
    )

    mapping <- mapping[!is.na(mapping$entrezgene_id) &
                       mapping$ensembl_gene_id != "", , drop = FALSE]
    mapping <- mapping[!duplicated(mapping$entrezgene_id), , drop = FALSE]
    cat("  biomaRt:", nrow(mapping), "Entrez IDs mapped to Ensembl\n")

    entrez_to_ens    <- stats::setNames(mapping$ensembl_gene_id,
                                        as.character(mapping$entrezgene_id))
    entrez_to_symbol <- stats::setNames(mapping$hgnc_symbol,
                                        as.character(mapping$entrezgene_id))
  }

  ## 3) Map matrix rows to Ensembl ----------------------------------------------
  ens_ids     <- entrez_to_ens[entrez_ids]
  gene_syms   <- entrez_to_symbol[entrez_ids]

  keep <- !is.na(ens_ids)
  cat("  Genes with Ensembl mapping:", sum(keep), "of", length(entrez_ids), "\n")

  mat2      <- mat[keep, , drop = FALSE]
  ens_ids2  <- ens_ids[keep]
  syms2     <- gene_syms[keep]

  ## Collapse duplicated Ensembl IDs by summing
  if (anyDuplicated(ens_ids2) > 0) {
    idx_list <- split(seq_along(ens_ids2), ens_ids2)

    collapsed <- lapply(idx_list, function(idx) {
      if (length(idx) == 1) {
        mat2[idx, , drop = FALSE]
      } else {
        matrix(colSums(mat2[idx, , drop = FALSE]), nrow = 1)
      }
    })

    mat3        <- do.call(rbind, collapsed)
    rownames(mat3) <- names(idx_list)

    sym_by_ens  <- vapply(
      idx_list,
      function(idx) paste(unique(syms2[idx]), collapse = ";"),
      character(1)
    )
  } else {
    mat3        <- mat2
    rownames(mat3) <- ens_ids2
    sym_by_ens  <- stats::setNames(syms2, ens_ids2)
  }

  cat("  Final matrix:", nrow(mat3), "genes x", ncol(mat3), "samples\n")

  obj2 <- Matrix(mat3, sparse = TRUE)

  ## 4) Build obs (samples only — no day/donor/phase for bulk data) -------------
  obs <- data.frame(
    sample_id = colnames(obj2),
    n_counts  = Matrix::colSums(obj2),
    row.names = colnames(obj2),
    stringsAsFactors = FALSE
  )

  ## 5) Build var ---------------------------------------------------------------
  var <- data.frame(
    ensembl_id  = rownames(obj2),
    gene_symbol = unname(sym_by_ens[rownames(obj2)]),
    row.names   = rownames(obj2),
    stringsAsFactors = FALSE
  )

  ## 6) Build SCE and write h5ad ------------------------------------------------
  sce <- SingleCellExperiment(
    assays  = list(counts = obj2),
    colData = S4Vectors::DataFrame(obs),
    rowData = S4Vectors::DataFrame(var)
  )

  out_file <- opt$out
  if (file.exists(out_file)) {
    message("Removing existing file: ", out_file)
    file.remove(out_file)
  }

  write_h5ad(
    sce,
    out_file,
    x_mapping    = "counts",
    layers_mapping = FALSE,
    obs_mapping  = TRUE,
    var_mapping  = TRUE,
    obsm_mapping = FALSE,
    varm_mapping = FALSE,
    obsp_mapping = FALSE,
    varp_mapping = FALSE,
    uns_mapping  = FALSE
  )

  cat("Wrote", out_file, "\n")
  cat("Samples:", ncol(sce), "\n")
  cat("Genes:  ", nrow(sce), "\n")
  quit(status = 0)
}

## ===========================================================================
## MODE: microarray
## Gene x sample CSV (gene symbols as rownames, samples as columns).
## No per-sample metadata required. Gene symbols -> Ensembl via biomaRt.
## Usage:
##   Rscript convert_mat_to_h5af.R --mode microarray \
##       --mat  data/GSE36318_for_geneformer.csv \
##       --out  data/GSE36318_geneformer_ready.h5ad
## ===========================================================================
if (opt$mode == "microarray") {

  ## 1) Read gene x sample matrix -----------------------------------------------
  cat("Reading microarray matrix...\n")
  if (requireNamespace("data.table", quietly = TRUE)) {
    library(data.table)
    dt          <- data.table::fread(opt$mat, header = TRUE, data.table = FALSE)
    gene_symbols <- dt[, 1]
    dt           <- dt[, -1, drop = FALSE]
    rownames(dt) <- gene_symbols
    obj          <- as(as.matrix(dt), "dgCMatrix")
  } else {
    df  <- read.csv(opt$mat, row.names = 1, check.names = FALSE)
    obj <- as(as.matrix(df), "dgCMatrix")
  }
  cat("  Loaded:", nrow(obj), "genes x", ncol(obj), "samples\n")

  ## 2) Map gene symbols -> Ensembl via biomaRt ---------------------------------
  cache_dir <- file.path(tempdir(), "biomart_cache")
  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
  Sys.setenv(BIOMART_CACHE = cache_dir)

  genes   <- rownames(obj)
  mart    <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")
  mapping <- getBM(
    attributes = c("hgnc_symbol", "ensembl_gene_id"),
    filters    = "hgnc_symbol",
    values     = unique(genes),
    mart       = mart,
    useCache   = FALSE
  )
  mapping <- mapping[mapping$hgnc_symbol != "" & mapping$ensembl_gene_id != "", , drop = FALSE]
  mapping <- mapping[!duplicated(mapping$hgnc_symbol), , drop = FALSE]
  cat("  biomaRt:", nrow(mapping), "gene symbols mapped to Ensembl\n")

  keep         <- rownames(obj) %in% mapping$hgnc_symbol
  obj2         <- obj[keep, , drop = FALSE]
  orig_symbols <- rownames(obj2)
  ens_ids      <- mapping$ensembl_gene_id[match(orig_symbols, mapping$hgnc_symbol)]

  valid <- !is.na(ens_ids) & ens_ids != ""
  obj2         <- obj2[valid, , drop = FALSE]
  orig_symbols <- orig_symbols[valid]
  ens_ids      <- ens_ids[valid]

  ## Collapse duplicated Ensembl IDs by summing
  if (anyDuplicated(ens_ids) > 0) {
    idx_list <- split(seq_along(ens_ids), ens_ids)
    collapsed <- lapply(idx_list, function(idx) {
      if (length(idx) == 1) {
        obj2[idx, , drop = FALSE]
      } else {
        Matrix(matrix(Matrix::colSums(obj2[idx, , drop = FALSE]), nrow = 1), sparse = TRUE)
      }
    })
    obj2 <- do.call(rbind, collapsed)
    rownames(obj2) <- names(idx_list)
    symbol_by_ens <- vapply(idx_list,
      function(idx) paste(unique(orig_symbols[idx]), collapse = ";"), character(1))
  } else {
    rownames(obj2)  <- ens_ids
    symbol_by_ens   <- stats::setNames(orig_symbols, rownames(obj2))
  }
  cat("  Final matrix:", nrow(obj2), "genes x", ncol(obj2), "samples\n")

  ## 3) Build obs (sample IDs only — no phase/donor metadata) -------------------
  obs <- data.frame(
    sample_id = colnames(obj2),
    n_counts  = Matrix::colSums(obj2),
    row.names = colnames(obj2),
    stringsAsFactors = FALSE
  )

  ## 4) Build var ---------------------------------------------------------------
  var <- data.frame(
    ensembl_id  = rownames(obj2),
    gene_symbol = unname(symbol_by_ens[rownames(obj2)]),
    row.names   = rownames(obj2),
    stringsAsFactors = FALSE
  )

  ## 5) Write h5ad --------------------------------------------------------------
  sce <- SingleCellExperiment(
    assays  = list(counts = obj2),
    colData = S4Vectors::DataFrame(obs),
    rowData = S4Vectors::DataFrame(var)
  )

  out_file <- opt$out
  if (file.exists(out_file)) { message("Removing existing: ", out_file); file.remove(out_file) }

  write_h5ad(sce, out_file,
    x_mapping = "counts", layers_mapping = FALSE,
    obs_mapping = TRUE,   var_mapping = TRUE,
    obsm_mapping = FALSE, varm_mapping = FALSE,
    obsp_mapping = FALSE, varp_mapping = FALSE,
    uns_mapping = FALSE)

  cat("Wrote", out_file, "\n")
  cat("Samples:", ncol(sce), "\n")
  cat("Genes:  ", nrow(sce), "\n")
  quit(status = 0)
}

## ===========================================================================
## MODES: 10x and ct  (original behaviour, unchanged)
## ===========================================================================

## 1) Read inputs -------------------------------------------------------------
if (opt$mode == "10x") {
  obj <- readRDS(opt$mat)
  stopifnot(inherits(obj, "dgCMatrix"))

} else if (opt$mode == "ct") {
  ## genes x cells CSV: first column = gene names, remaining = cell counts
  cat("Reading CT CSV (this may take a moment)...\n")
  if (requireNamespace("data.table", quietly = TRUE)) {
    library(data.table)
    dt         <- data.table::fread(opt$mat, header = TRUE, data.table = FALSE)
    gene_names <- dt[, 1]
    dt         <- dt[, -1, drop = FALSE]
    rownames(dt) <- gene_names
    obj <- as(as.matrix(dt), "dgCMatrix")
  } else {
    df  <- read.csv(opt$mat, row.names = 1, check.names = FALSE)
    obj <- as(as.matrix(df), "dgCMatrix")
  }
  cat("  Loaded:", nrow(obj), "genes x", ncol(obj), "cells\n")

} else {
  stop("Unknown --mode '", opt$mode, "'. Use '10x', 'ct', or 'ncbi'.")
}

meta_cell <- read.csv(opt$meta,  stringsAsFactors = FALSE, check.names = FALSE)
meta_phase <- read.csv(opt$phase, stringsAsFactors = FALSE, check.names = FALSE)

## 2) Clean and align metadata ------------------------------------------------
meta_cell$cell_type <- gsub("epit/helia", "epithelia", meta_cell$cell_type)

cell_id_col <- if ("cell_name" %in% colnames(meta_cell)) "cell_name" else colnames(meta_cell)[1]

## Intersect: keep only cells present in both matrix and metadata
shared_cells <- intersect(colnames(obj), meta_cell[[cell_id_col]])
if (length(shared_cells) == 0) stop("No cell IDs match between matrix and metadata.")
if (length(shared_cells) < ncol(obj)) {
  cat("  Keeping", length(shared_cells), "of", ncol(obj),
      "cells that appear in metadata\n")
}
obj <- obj[, shared_cells, drop = FALSE]

meta_cell2 <- meta_cell[match(shared_cells, meta_cell[[cell_id_col]]), , drop = FALSE]
stopifnot(all(meta_cell2[[cell_id_col]] == shared_cells))

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

out_file <- opt$out
if (file.exists(out_file)) {
  message("Removing existing file: ", out_file)
  file.remove(out_file)
}

## 5) Write directly from SCE
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

cat("Wrote", out_file, "\n")
cat("Cells:", ncol(sce), "\n")
cat("Genes:", nrow(sce), "\n")
