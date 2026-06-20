#!/usr/bin/env Rscript
# figures.R — Paper figures from pipeline-exported CSVs
#
# Prerequisites:
#   install.packages(c("tidyverse", "cowplot", "ggrepel"))
#
# Usage:
#   Rscript figures.R
#   # or interactively: source("figures.R")
#
# Data source:
#   Run the pipeline scripts with --figure-data-dir figure_data to populate
#   figure_data/ before running this script.

suppressPackageStartupMessages({
  library(tidyverse)
  library(cowplot)
  library(ggrepel)
})

DATA <- "figure_data"
OUT  <- "figures"
dir.create(OUT, showWarnings = FALSE)

# Okabe-Ito palette (colorblind-friendly)
oi <- c(
  orange       = "#E69F00",
  sky_blue     = "#56B4E9",
  green        = "#009E73",
  yellow       = "#F0E442",
  blue         = "#0072B2",
  vermillion   = "#D55E00",
  pink         = "#CC79A7",
  black        = "#000000"
)

CAV_COLOR  <- unname(oi["blue"])
TOOL_COLOR <- unname(oi["vermillion"])

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
temporal_terms <- read_csv(file.path(DATA, "temporal_per_term_summary.csv"),
                           show_col_types = FALSE)

temporal_comp <- read_csv(file.path(DATA, "temporal_tool_comparison.csv"),
                          show_col_types = FALSE)

temporal_pairs <- read_csv(file.path(DATA, "temporal_protein_pairs.csv"),
                           show_col_types = FALSE)

temporal_pr <- read_csv(file.path(DATA, "temporal_pr_curves.csv"),
                        show_col_types = FALSE)

temporal_density <- read_csv(file.path(DATA, "temporal_pos_neg_density.csv"),
                             show_col_types = FALSE)

ec_terms <- read_csv(file.path(DATA, "ec_per_term_summary.csv"),
                     show_col_types = FALSE)

ec_summary <- read_csv(file.path(DATA, "ec_tool_comparison_summary.csv"),
                       show_col_types = FALSE)

ec_llr <- read_csv(file.path(DATA, "ec_recall_vs_llr.csv"),
                   show_col_types = FALSE)

# ---------------------------------------------------------------------------
# Helper: base theme
# ---------------------------------------------------------------------------
base_theme <- function(...) {
  theme_cowplot(font_size = 11, ...) +
    theme(plot.title = element_text(size = 11, face = "plain"))
}

# ---------------------------------------------------------------------------
# Panel: GO temporal — AUC violin (CAV vs external tool)
# ---------------------------------------------------------------------------
p_auc_violin <- temporal_comp |>
  select(go_term, CAV = auc_val_vs_test_neg, Tool = tool_auc) |>
  pivot_longer(c(CAV, Tool), names_to = "method", values_to = "auc") |>
  drop_na(auc) |>
  ggplot(aes(x = method, y = auc, fill = method)) +
  geom_violin(alpha = 0.65, color = NA, width = 0.8) +
  geom_boxplot(width = 0.12, outlier.shape = NA, fill = "white", color = "gray30",
               linewidth = 0.5) +
  scale_fill_manual(values = c(CAV = CAV_COLOR, Tool = TOOL_COLOR)) +
  base_theme() +
  labs(x = NULL, y = "AUC  (val vs. test-neg)") +
  theme(legend.position = "none")

# ---------------------------------------------------------------------------
# Panel: GO temporal — CAV AUC vs tool AUC, coloured by GO depth
# ---------------------------------------------------------------------------
p_auc_scatter <- temporal_comp |>
  drop_na(auc_val_vs_test_neg, tool_auc) |>
  ggplot(aes(x = tool_auc, y = auc_val_vs_test_neg)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed",
              color = "gray60", linewidth = 0.7) +
  geom_point(aes(color = depth), alpha = 0.75, size = 1.5) +
  scale_color_gradient(low = oi["sky_blue"], high = oi["vermillion"],
                       na.value = "gray70") +
  coord_equal(xlim = c(0.4, 1), ylim = c(0.4, 1)) +
  base_theme() +
  labs(x = "Tool AUC", y = "CAV AUC", color = "GO\ndepth")

# ---------------------------------------------------------------------------
# Panel: Macro-averaged PR curve (CAV vs tool)
# ---------------------------------------------------------------------------
pr_long <- bind_rows(
  temporal_pr |> transmute(recall,
                            precision = cav_precision,
                            sd        = cav_precision_sd,
                            method    = "CAV"),
  temporal_pr |> transmute(recall,
                            precision = tool_precision,
                            sd        = tool_precision_sd,
                            method    = "Tool")
)

p_pr_curve <- pr_long |>
  ggplot(aes(x = recall, y = precision, color = method, fill = method)) +
  geom_ribbon(aes(ymin = pmax(precision - sd, 0),
                  ymax = pmin(precision + sd, 1)),
              alpha = 0.15, color = NA) +
  geom_line(linewidth = 1) +
  scale_color_manual(values = c(CAV = CAV_COLOR, Tool = TOOL_COLOR)) +
  scale_fill_manual(values  = c(CAV = CAV_COLOR, Tool = TOOL_COLOR)) +
  base_theme() +
  labs(x = "Recall", y = "Precision", color = NULL, fill = NULL) +
  xlim(0, 1) + ylim(0, 1)

# ---------------------------------------------------------------------------
# Panel: GO temporal — pooled CAV score density, positives vs. negatives
# ---------------------------------------------------------------------------
p_density_cav <- temporal_density |>
  mutate(label = str_to_title(label)) |>
  ggplot(aes(x = cav_score, color = label, fill = label)) +
  geom_density(alpha = 0.2, linewidth = 0.9) +
  scale_color_manual(values = c(Positive = CAV_COLOR, Negative = TOOL_COLOR)) +
  scale_fill_manual(values  = c(Positive = CAV_COLOR, Negative = TOOL_COLOR)) +
  base_theme() +
  labs(x = "CAV score", y = "Density", color = NULL, fill = NULL)

# ---------------------------------------------------------------------------
# Panel: EC — tool recall comparison (horizontal bar)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Panel: EC — coverage vs recall scatter
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Panel: EC — CAV recall vs LLR threshold
# ---------------------------------------------------------------------------
p_ec_llr <- ec_llr |>
  ggplot(aes(x = llr_threshold, y = recall)) +
  geom_line(color = CAV_COLOR, linewidth = 1) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray60", linewidth = 0.7) +
  base_theme() +
  labs(x = "LLR threshold", y = "Recall (fraction of val pairs)")

# ===========================================================================
# Assemble figures
# ===========================================================================

# --- Figure: GO temporal evaluation (3 panels) ---
fig_temporal <- plot_grid(
  p_auc_violin, p_auc_scatter, p_pr_curve,
  nrow   = 1,
  labels = "AUTO",
  align  = "hv",
  axis   = "tblr"
)

ggsave(file.path(OUT, "fig_temporal_eval.pdf"),
       fig_temporal, width = 12, height = 4)
message("Saved fig_temporal_eval.pdf")

# --- Figure: EC evaluation (3 panels) ---
fig_ec <- plot_grid(
  p_ec_recall, p_ec_coverage, p_ec_llr,
  nrow   = 1,
  labels = "AUTO",
  align  = "hv",
  axis   = "tblr"
)

ggsave(file.path(OUT, "fig_ec_eval.pdf"),
       fig_ec, width = 14, height = 5)
message("Saved fig_ec_eval.pdf")

# --- Combined main figure (2 rows) ---
fig_main <- plot_grid(
  p_auc_violin,  p_auc_scatter,  p_pr_curve,
  p_ec_recall,   p_ec_coverage,  p_ec_llr,
  nrow   = 2,
  labels = "AUTO",
  align  = "hv",
  axis   = "tblr"
)

ggsave(file.path(OUT, "fig_main.pdf"),
       fig_main, width = 14, height = 8)
message("Saved fig_main.pdf")

message("\nAll figures written to ", OUT, "/")
