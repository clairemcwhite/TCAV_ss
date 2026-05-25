#!/usr/bin/env python

import argparse

import math

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


ROOT_TERMS = {
    "mf": "GO:0003674",  # molecular_function
    "bp": "GO:0008150",  # biological_process
    "cc": "GO:0005575",  # cellular_component
}


def load_scores(args):
    if args.pred_wide:
        scores = pd.read_csv(args.pred_wide, sep="\t")

        if "Entry" not in scores.columns:
            raise ValueError("Wide prediction file must contain an 'Entry' column.")

        scores = scores.set_index("Entry")
        scores = scores.drop(columns=["Entry_Name"], errors="ignore")

        return scores.astype(float)

    pred = pd.read_csv(args.pred_long, sep="\t")

    required = {"Entry", "go_term", "score"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"Long prediction file is missing columns: {missing}")

    scores = pred.pivot(
        index="Entry",
        columns="go_term",
        values="score",
    )

    return scores.astype(float)


def load_truth(truth_file, allowed_terms):
    truth = pd.read_csv(truth_file, sep="\t")

    required = {"Entry", "go_term"}
    missing = required - set(truth.columns)
    if missing:
        raise ValueError(f"Truth file is missing columns: {missing}")

    truth = truth[truth["go_term"].isin(allowed_terms)].copy()

    truth_sets = (
        truth.groupby("Entry")["go_term"]
        .apply(set)
        .to_dict()
    )

    return truth_sets


def load_eval_terms(eval_terms_file):
    """
    Load a TSV/CSV/plain file containing GO terms.

    Accepts a column named:
      - go_term
      - gos
      - GO
      - Entry

    Otherwise uses the first column.
    """
    if eval_terms_file.endswith(".csv"):
        df = pd.read_csv(eval_terms_file)
    else:
        df = pd.read_csv(eval_terms_file, sep="\t")

    for col in ["go_term", "gos", "GO", "Entry"]:
        if col in df.columns:
            return set(df[col].dropna().astype(str))

    return set(df.iloc[:, 0].dropna().astype(str))


def restrict_to_eval_terms(scores, args):
    if args.eval_terms is None:
        return scores

    eval_terms = load_eval_terms(args.eval_terms)
    score_terms = set(scores.columns)

    missing_terms = eval_terms - score_terms
    keep_terms = sorted(eval_terms & score_terms)

    print(f"Requested eval terms: {len(eval_terms)}")
    print(f"Eval terms available in predictions: {len(keep_terms)}")
    print(f"Eval terms missing from predictions: {len(missing_terms)}")

    if args.require_all_eval_terms and missing_terms:
        examples = sorted(missing_terms)[:10]
        raise ValueError(
            f"{len(missing_terms)} eval terms are missing from prediction columns. "
            f"Examples: {examples}"
        )

    if len(keep_terms) == 0:
        raise ValueError("No overlap between --eval-terms and prediction score columns.")

    return scores[keep_terms]


def protein_centric_metrics(scores, truth_sets, threshold):
    """
    DeepGO-style protein-centric precision/recall/F1.

    Recall is averaged over proteins with at least one true annotation.
    Precision is averaged only over proteins with at least one prediction.
    F1 is computed from the averaged precision and recall.
    """
    precisions = []
    recalls = []

    for entry, row in scores.iterrows():
        true_terms = truth_sets.get(entry, set())

        if len(true_terms) == 0:
            continue

        pred_terms = set(row.index[row.values >= threshold])

        tp = len(true_terms & pred_terms)
        fp = len(pred_terms - true_terms)
        fn = len(true_terms - pred_terms)

        recall = tp / (tp + fn) if (tp + fn) else 0.0
        recalls.append(recall)

        if len(pred_terms) > 0:
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            precisions.append(precision)

    mean_precision = float(np.mean(precisions)) if precisions else 0.0
    mean_recall = float(np.mean(recalls)) if recalls else 0.0

    if mean_precision + mean_recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * mean_precision * mean_recall / (mean_precision + mean_recall)

    return mean_precision, mean_recall, f1


def parse_obo(obo_file):
    """Return {term_id: {"parents": [...], "namespace": str}} for non-obsolete terms."""
    terms = {}
    current_id = current_namespace = None
    current_parents = []
    current_is_obsolete = in_term = False

    with open(obo_file) as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                if in_term and current_id and not current_is_obsolete:
                    terms[current_id] = {"parents": current_parents, "namespace": current_namespace}
                in_term = True
                current_id = current_namespace = None
                current_parents = []
                current_is_obsolete = False
            elif line == "[Typedef]":
                if in_term and current_id and not current_is_obsolete:
                    terms[current_id] = {"parents": current_parents, "namespace": current_namespace}
                in_term = False
            elif in_term:
                if line.startswith("id: "):
                    current_id = line[4:].strip()
                elif line.startswith("namespace: "):
                    current_namespace = line[11:].strip()
                elif line.startswith("is_a: "):
                    current_parents.append(line[6:].split()[0])
                elif line.startswith("relationship: part_of "):
                    current_parents.append(line[22:].split()[0])
                elif line.startswith("is_obsolete: true"):
                    current_is_obsolete = True

    if in_term and current_id and not current_is_obsolete:
        terms[current_id] = {"parents": current_parents, "namespace": current_namespace}

    return terms


def propagate_annotations(annots, obo_terms):
    """Walk each term up to the root, returning the full propagated set."""
    result = set(annots)
    queue = list(annots)
    while queue:
        term = queue.pop()
        for parent in obo_terms.get(term, {}).get("parents", []):
            if parent not in result:
                result.add(parent)
                queue.append(parent)
    return result


def compute_ic_from_obo(truth_sets, obo_terms):
    """IC computed from ontology-propagated truth annotations."""
    n = len(truth_sets)
    counts = {}
    for annots in truth_sets.values():
        for t in propagate_annotations(annots, obo_terms):
            counts[t] = counts.get(t, 0) + 1

    ic = {t: -math.log2(c / n) for t, c in counts.items() if c > 0}
    max_ic = max(ic.values(), default=1.0)
    norm_ic = {t: v / max_ic for t, v in ic.items()}
    return ic, norm_ic



def protein_centric_metrics_ic(scores, truth_sets, threshold, ic, norm_ic):
    """IC-weighted protein-centric metrics: WFmax component and Smin component."""
    total = p_total = 0
    r = wr = p = wp = 0.0
    ru_sum = mi_sum = tp_ic_sum = 0.0

    for entry, row in scores.iterrows():
        true_terms = truth_sets.get(entry, set())
        if not true_terms:
            continue

        pred_terms = set(row.index[row.values >= threshold])
        tp = true_terms & pred_terms
        fp = pred_terms - true_terms
        fn = true_terms - pred_terms

        tpic = sum(norm_ic.get(t, 0.0) for t in tp)
        fpic = sum(norm_ic.get(t, 0.0) for t in fp)
        fnic = sum(norm_ic.get(t, 0.0) for t in fn)

        tp_ic_sum += sum(ic.get(t, 0.0) for t in tp)
        mi_sum    += sum(ic.get(t, 0.0) for t in fp)
        ru_sum    += sum(ic.get(t, 0.0) for t in fn)

        tpn, fpn, fnn = len(tp), len(fp), len(fn)
        total += 1
        r  += tpn / (tpn + fnn) if (tpn + fnn) else 0.0
        wr += tpic / (tpic + fnic) if (tpic + fnic) > 0 else 0.0

        if pred_terms:
            p_total += 1
            p  += tpn / (tpn + fpn) if (tpn + fpn) else 0.0
            wp += tpic / (tpic + fpic) if (tpic + fpic) > 0 else 0.0

    if total == 0:
        return 0.0, float("inf"), 0.0

    r  /= total
    wr /= total
    if p_total > 0:
        p  /= p_total
        wp /= p_total

    wf = 2 * wp * wr / (wp + wr) if (wp + wr) > 0 else 0.0
    s  = math.sqrt((ru_sum / total) ** 2 + (mi_sum / total) ** 2)
    avg_ic_val = (tp_ic_sum + mi_sum) / total

    return wf, s, avg_ic_val


def make_label_matrix(scores, truth_sets):
    labels = np.zeros(scores.shape, dtype=np.uint8)
    term_to_idx = {go: i for i, go in enumerate(scores.columns)}

    for row_i, entry in enumerate(scores.index):
        for go in truth_sets.get(entry, set()):
            if go in term_to_idx:
                labels[row_i, term_to_idx[go]] = 1

    return labels


def macro_term_roc_auc(scores, labels):
    aucs = []

    for j in range(scores.shape[1]):
        y_true = labels[:, j]
        y_score = scores.iloc[:, j].values

        # Skip terms with all-negative or all-positive truth.
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue

        aucs.append(roc_auc_score(y_true, y_score))

    return float(np.mean(aucs)) if aucs else np.nan


def macro_term_aupr(scores, labels):
    auprs = []

    for j in range(scores.shape[1]):
        y_true = labels[:, j]
        y_score = scores.iloc[:, j].values

        # Skip terms with no positives.
        if y_true.sum() == 0:
            continue

        auprs.append(average_precision_score(y_true, y_score))

    return float(np.mean(auprs)) if auprs else np.nan


def trapezoid_area(y, x):
    """
    NumPy 2.x uses np.trapezoid.
    Older NumPy used np.trapz.
    """
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def make_thresholds(scores, args):
    if args.threshold_min is None:
        threshold_min = float(np.nanmin(scores.values))
    else:
        threshold_min = args.threshold_min

    if args.threshold_max is None:
        threshold_max = float(np.nanmax(scores.values))
    else:
        threshold_max = args.threshold_max

    if threshold_min > threshold_max:
        raise ValueError(
            f"threshold-min ({threshold_min}) is greater than "
            f"threshold-max ({threshold_max})."
        )

    return np.arange(
        threshold_min,
        threshold_max + args.step / 2,
        args.step,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute simplified DeepGO-style Fmax/AUPR/AUC from prediction TSVs. "
            "Uses Entry as the protein key."
        )
    )

    pred_group = parser.add_mutually_exclusive_group(required=True)
    pred_group.add_argument(
        "--pred-wide",
        help="Wide prediction TSV with Entry, Entry_Name, GO score columns.",
    )
    pred_group.add_argument(
        "--pred-long",
        help="Long prediction TSV with Entry, Entry_Name, go_term, score.",
    )

    parser.add_argument(
        "--truth-long",
        required=True,
        help="Positive-only truth TSV with Entry, Entry_Name, go_term.",
    )
    parser.add_argument(
        "--eval-terms",
        default=None,
        help=(
            "Optional TSV/CSV file of GO terms to evaluate. "
            "Accepts column go_term, gos, GO, Entry, or uses first column."
        ),
    )
    parser.add_argument(
        "--require-all-eval-terms",
        action="store_true",
        help="Fail if any --eval-terms are missing from prediction columns.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output TSV for threshold curve.",
    )
    parser.add_argument(
        "--ont",
        choices=["mf", "bp", "cc"],
        default="mf",
        help="GO ontology. Used only for choosing root term to drop.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.01,
        help="Threshold step. Default: 0.01.",
    )
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=None,
        help="Minimum threshold after score transformation. Default: min observed score.",
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=None,
        help="Maximum threshold after score transformation. Default: max observed score.",
    )
    parser.add_argument(
        "--score-direction",
        choices=["higher", "lower"],
        default="higher",
        help=(
            "'higher' means larger scores are better, e.g. probabilities. "
            "'lower' means smaller scores are better, e.g. p-values."
        ),
    )
    parser.add_argument(
        "--auc",
        action="store_true",
        help="Also compute macro term ROC-AUC.",
    )
    parser.add_argument(
        "--aupr-macro",
        action="store_true",
        help="Also compute macro term average precision.",
    )
    parser.add_argument(
        "--keep-root",
        action="store_true",
        help="Do not drop the ontology root term.",
    )
    parser.add_argument(
        "--go-obo",
        required=True,
        help="Path to go.obo. IC for --smin/--wfmax is computed from ontology-propagated truth annotations.",
    )
    parser.add_argument(
        "--smin",
        action="store_true",
        help="Compute Smin (semantic distance: sqrt(RU^2 + MI^2)).",
    )
    parser.add_argument(
        "--wfmax",
        action="store_true",
        help="Compute WFmax (IC-weighted Fmax) and avgic at Fmax threshold.",
    )

    args = parser.parse_args()

    scores = load_scores(args)

    # Drop ontology root term before optional eval-term restriction.
    if not args.keep_root:
        root = ROOT_TERMS[args.ont]
        if root in scores.columns:
            scores = scores.drop(columns=[root])
            print(f"Dropped root term: {root}")

    # Restrict to requested GO terms, if provided.
    scores = restrict_to_eval_terms(scores, args)

    # Convert so the rest of the script always treats larger = better.
    # For p-values, p <= 0.05 becomes -p >= -0.05.
    if args.score_direction == "lower":
        scores = -scores

    truth_sets = load_truth(args.truth_long, set(scores.columns))

    # Keep only proteins with at least one truth annotation after filtering.
    keep = [entry for entry in scores.index if entry in truth_sets]
    scores = scores.loc[keep]

    print(f"Proteins evaluated: {scores.shape[0]}")
    print(f"GO terms evaluated: {scores.shape[1]}")

    if scores.shape[0] == 0:
        raise ValueError("No overlapping proteins between predictions and truth after filtering.")

    thresholds = make_thresholds(scores, args)

    use_ic_metrics = args.smin or args.wfmax
    if use_ic_metrics:
        obo_terms = parse_obo(args.go_obo)
        ic, norm_ic = compute_ic_from_obo(truth_sets, obo_terms)

    rows = []

    for threshold in thresholds:
        precision, recall, f1 = protein_centric_metrics(scores, truth_sets, threshold)

        row = {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        if use_ic_metrics:
            wf, s, avg_ic_val = protein_centric_metrics_ic(
                scores, truth_sets, threshold, ic, norm_ic
            )
            row["wf"] = wf
            row["s"] = s
            row["avg_ic"] = avg_ic_val

        if args.score_direction == "lower":
            row["original_score_cutoff"] = -threshold

        rows.append(row)

    curve = pd.DataFrame(rows)

    best = curve.sort_values(
        ["f1", "threshold"],
        ascending=[False, True],
    ).iloc[0]

    print()
    print(f"Fmax: {best['f1']:.4f}")

    if args.score_direction == "lower":
        print(f"threshold_transformed: {best['threshold']:.6g}")
        print(f"original_score_cutoff: <= {best['original_score_cutoff']:.6g}")
    else:
        print(f"threshold: {best['threshold']:.6g}")

    print(f"precision: {best['precision']:.4f}")
    print(f"recall: {best['recall']:.4f}")

    if use_ic_metrics and args.wfmax:
        print(f"avgic: {best['avg_ic']:.4f}")

    curve_sorted = curve.sort_values("recall")
    aupr_threshold_curve = trapezoid_area(
        curve_sorted["precision"].values,
        curve_sorted["recall"].values,
    )
    print(f"AUPR_threshold_curve: {aupr_threshold_curve:.4f}")

    if use_ic_metrics:
        if args.wfmax:
            wf_best = curve.sort_values(["wf", "threshold"], ascending=[False, True]).iloc[0]
            print(f"WFmax: {wf_best['wf']:.4f}")
            if args.score_direction == "lower":
                print(f"wfmax_threshold_transformed: {wf_best['threshold']:.6g}")
            else:
                print(f"wfmax_threshold: {wf_best['threshold']:.6g}")
        if args.smin:
            print(f"Smin: {curve['s'].min():.4f}")

    if args.auc or args.aupr_macro:
        labels = make_label_matrix(scores, truth_sets)

        if args.auc:
            avg_auc = macro_term_roc_auc(scores, labels)
            print(f"macro_term_ROC_AUC: {avg_auc:.4f}")

        if args.aupr_macro:
            avg_aupr = macro_term_aupr(scores, labels)
            print(f"macro_term_AUPR: {avg_aupr:.4f}")

    if args.out:
        curve.to_csv(args.out, sep="\t", index=False)


if __name__ == "__main__":
    main()
