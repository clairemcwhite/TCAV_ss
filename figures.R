#!/usr/bin/env Rscript
# figures.R — Paper figures from pipeline-exported CSVs
#
# Prerequisites:
#   install.packages(c("tidyverse", "cowplot", "ggridges", "ggrepel"))
#
# Usage:
#   Rscript figures.R
#   # or interactively: source("figures.R")
#
# Data source:
#   Run pipeline scripts with --figure-data-dir figure_data --label <mf|bp|cc>
#   to populate figure_data/ before running this script.

suppressPackageStartupMessages({
  library(tidyverse)
  library(cowplot)
  library(ggridges)
  library(ggrepel)
})

DATA <- "figure_data"
OUT  <- "figures"
dir.create(OUT, showWarnings = FALSE)

# Okabe-Ito palette (colorblind-friendly)
oi <- c(
  orange     = "#E69F00",
  sky_blue   = "#56B4E9",
  green      = "#009E73",
  yellow     = "#F0E442",
  blue       = "#0072B2",
  vermillion = "#D55E00",
  pink       = "#CC79A7",
  black      = "#000000"
)

CAV_COLOR  <- unname(oi["blue"])
TOOL_COLOR <- unname(oi["vermillion"])

# ---------------------------------------------------------------------------
# Helper: base theme
# ---------------------------------------------------------------------------
base_theme <- function(font_size = 11, ...) {
  theme_cowplot(font_size = font_size, ...) +
    theme(plot.title = element_text(size = font_size, face = "plain"))
}

# ---------------------------------------------------------------------------
# Helper: load a per-ontology file, return NULL if missing
# ---------------------------------------------------------------------------
load_ont <- function(stem, ont) {
  f <- file.path(DATA, paste0(stem, "_", ont, ".csv"))
  if (!file.exists(f)) {
    message("Missing: ", f)
    return(NULL)
  }
  read_csv(f, show_col_types = FALSE) |> mutate(ontology = toupper(ont))
}

# ===========================================================================
# Figure 2 — GO and EC evaluation
# ===========================================================================

# ---------------------------------------------------------------------------
# 2A: Overlapping histograms — per-GO-term AUC, CAV vs DeepGoSE (MF only)
# ---------------------------------------------------------------------------
comp_mf <- load_ont("temporal_tool_comparison", "mf")

if (!is.null(comp_mf)) {
  p_2a <- comp_mf |>
    select(go_term, CAV = auc_val_vs_test_neg, DeepGoSE = tool_auc) |>
    pivot_longer(c(CAV, DeepGoSE), names_to = "method", values_to = "auc") |>
    drop_na(auc) |>
    ggplot(aes(x = auc, fill = method, color = method)) +
    geom_histogram(
      position = "identity", alpha = 0.5,
      bins = 30, boundary = 0.5
    ) +
    scale_fill_manual(values  = c(CAV = CAV_COLOR, DeepGoSE = TOOL_COLOR)) +
    scale_color_manual(values = c(CAV = CAV_COLOR, DeepGoSE = TOOL_COLOR)) +
    base_theme() +
    labs(
      x     = "AUC  (val vs. test-neg)",
      y     = "GO terms",
      fill  = NULL,
      color = NULL
    )
} else {
  message("Skipping 2A: figure_data/temporal_tool_comparison_mf.csv not found")
  p_2a <- NULL
}

# ---------------------------------------------------------------------------
# 2B: Sparklines (ggridges) — AUC and AUPR distributions across MF, BP, CC
#
# ggridges is Claus Wilke's package for stacked density / ridgeline plots.
# Each row is one ontology × method combination; the filled curve shows the
# distribution of per-GO-term AUC (or AUPR) values.
# ---------------------------------------------------------------------------

# Load and combine all three ontologies
ont_comp <- bind_rows(
  load_ont("temporal_tool_comparison", "mf"),
  load_ont("temporal_tool_comparison", "bp"),
  load_ont("temporal_tool_comparison", "cc")
)

if (nrow(ont_comp) > 0) {

  # Tidy: one row per (go_term, ontology, metric, method, value)
  ridges_auc <- ont_comp |>
    select(ontology, go_term,
           CAV     = auc_val_vs_test_neg,
           DeepGoSE = tool_auc) |>
    pivot_longer(c(CAV, DeepGoSE), names_to = "method", values_to = "auc") |>
    drop_na(auc) |>
    mutate(
      row_label = factor(
        paste(ontology, method),
        levels = c("CC DeepGoSE", "CC CAV",
                   "BP DeepGoSE", "BP CAV",
                   "MF DeepGoSE", "MF CAV")
      )
    )

  ridges_aupr <- ont_comp |>
    select(ontology, go_term,
           CAV      = aupr_val_vs_test_neg,
           DeepGoSE = tool_aupr) |>
    pivot_longer(c(CAV, DeepGoSE), names_to = "method", values_to = "aupr") |>
    drop_na(aupr) |>
    mutate(
      row_label = factor(
        paste(ontology, method),
        levels = c("CC DeepGoSE", "CC CAV",
                   "BP DeepGoSE", "BP CAV",
                   "MF DeepGoSE", "MF CAV")
      )
    )

  ridges_opts <- list(
    scale         = 0.9,
    rel_min_height = 0.01,
    linewidth     = 0.4,
    alpha         = 0.55
  )

  p_2b_auc <- ridges_auc |>
    ggplot(aes(x = auc, y = row_label, fill = method, color = method)) +
    do.call(geom_density_ridges, ridges_opts) +
    scale_fill_manual(values  = c(CAV = CAV_COLOR, DeepGoSE = TOOL_COLOR)) +
    scale_color_manual(values = c(CAV = CAV_COLOR, DeepGoSE = TOOL_COLOR)) +
    scale_x_continuous(limits = c(0.4, 1.0), expand = c(0, 0)) +
    base_theme() +
    labs(x = "AUC", y = NULL, fill = NULL, color = NULL) +
    theme(legend.position = "none")

  p_2b_aupr <- ridges_aupr |>
    ggplot(aes(x = aupr, y = row_label, fill = method, color = method)) +
    do.call(geom_density_ridges, ridges_opts) +
    scale_fill_manual(values  = c(CAV = CAV_COLOR, DeepGoSE = TOOL_COLOR)) +
    scale_color_manual(values = c(CAV = CAV_COLOR, DeepGoSE = TOOL_COLOR)) +
    scale_x_continuous(limits = c(0, 1.0), expand = c(0, 0)) +
    base_theme() +
    labs(x = "AUPR", y = NULL, fill = NULL, color = NULL) +
    theme(
      axis.text.y  = element_blank(),
      axis.ticks.y = element_blank()
    )

  # Shared legend from p_2b_auc
  legend_2b <- get_legend(
    p_2b_auc + theme(legend.position = "right")
  )

  p_2b <- plot_grid(
    p_2b_auc, p_2b_aupr, legend_2b,
    nrow  = 1,
    rel_widths = c(1, 1, 0.25)
  )

} else {
  message("Skipping 2B: no temporal_tool_comparison_*.csv files found")
  p_2b <- NULL
}

# ---------------------------------------------------------------------------
# Assemble Figure 2
# ---------------------------------------------------------------------------
fig2_panels <- Filter(Negate(is.null), list(p_2a, p_2b))

if (length(fig2_panels) > 0) {
  fig2 <- plot_grid(
    plotlist  = fig2_panels,
    ncol      = 1,
    labels    = "AUTO",
    rel_heights = rep(1, length(fig2_panels))
  )
  ggsave(file.path(OUT, "fig2.pdf"), fig2, width = 10, height = 4 * length(fig2_panels))
  message("Saved fig2.pdf")
}

# ===========================================================================
# Other figures (scaffold — add panels as data becomes available)
# ===========================================================================

# --- EC tool comparison ---
ec_summary_path <- file.path(DATA, "ec_tool_comparison_summary.csv")
ec_llr_path     <- file.path(DATA, "ec_recall_vs_llr.csv")

if (file.exists(ec_summary_path)) {
  ec_summary <- read_csv(ec_summary_path, show_col_types = FALSE)

  p_ec_recall <- ec_summary |>
    mutate(
      tool_label = fct_reorder(tool, recall_exact),
      is_cav     = tool == "CAV"
    ) |>
    ggplot(aes(x = recall_exact, y = tool_label, fill = is_cav)) +
    geom_col(width = 0.65) +
    scale_fill_manual(values = c(`FALSE` = unname(oi["sky_blue"]),
                                 `TRUE`  = TOOL_COLOR)) +
    base_theme() +
    labs(x = "Recall (exact match)", y = NULL) +
    theme(legend.position = "none") +
    xlim(0, 1)

  p_ec_coverage <- ec_summary |>
    mutate(is_cav = tool == "CAV") |>
    ggplot(aes(x = coverage, y = recall_exact,
               color = is_cav, size = is_cav, label = tool)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                color = "gray60", linewidth = 0.7) +
    geom_point(alpha = 0.9) +
    geom_text_repel(size = 3, show.legend = FALSE, max.overlaps = 20) +
    scale_color_manual(values = c(`FALSE` = unname(oi["sky_blue"]),
                                  `TRUE`  = TOOL_COLOR),
                       labels = c("Other tools", "CAV")) +
    scale_size_manual(values = c(`FALSE` = 2, `TRUE` = 3.5), guide = "none") +
    coord_equal(xlim = c(0, 1.05), ylim = c(0, 1.05)) +
    base_theme() +
    labs(x = "Coverage", y = "Recall (exact match)", color = NULL)

  if (file.exists(ec_llr_path)) {
    ec_llr <- read_csv(ec_llr_path, show_col_types = FALSE)

    p_ec_llr <- ec_llr |>
      ggplot(aes(x = llr_threshold, y = recall)) +
      geom_line(color = CAV_COLOR, linewidth = 1) +
      geom_vline(xintercept = 0, linetype = "dashed",
                 color = "gray60", linewidth = 0.7) +
      base_theme() +
      labs(x = "LLR threshold", y = "Recall (fraction of val pairs)")

    fig_ec <- plot_grid(
      p_ec_recall, p_ec_coverage, p_ec_llr,
      nrow   = 1,
      labels = "AUTO",
      align  = "hv",
      axis   = "tblr"
    )
  } else {
    fig_ec <- plot_grid(
      p_ec_recall, p_ec_coverage,
      nrow   = 1,
      labels = "AUTO",
      align  = "hv",
      axis   = "tblr"
    )
  }

  ggsave(file.path(OUT, "fig_ec_eval.pdf"), fig_ec, width = 12, height = 4.5)
  message("Saved fig_ec_eval.pdf")
}

message("\nAll figures written to ", OUT, "/")
