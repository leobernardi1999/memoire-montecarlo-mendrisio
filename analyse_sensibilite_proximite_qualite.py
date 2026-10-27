#!/usr/bin/env python3
"""Analyse agrégée des commentaires 45–48 et 50.

L'analyse conserve le réseau, les trajets, le seuil de 15 minutes, les quatre
indicateurs de qualité, la loi Dirichlet(1, 1, 1, 1), les 1 000 tirages et la
graine 42. Seule la répartition entre proximité et qualité varie.

Les sorties ne contiennent ni coordonnées, ni géométries, ni résultats
individuels. Elles servent à documenter la sensibilité du modèle dans la thèse.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ROUTES = ROOT / "data/routes_recalculated_20m.csv"
DECLARED = ROOT / "data/declared_pairs.csv"
OUT = ROOT / "outputs/sensibilite_modele"

N_SIM = 1000
RANDOM_SEED = 42
THRESHOLD = 15.0
SCENARIOS = (
    ("qualite_100", 0.0),
    ("proximite_30_qualite_70", 0.3),
    ("proximite_50_qualite_50", 0.5),
    ("proximite_70_qualite_30", 0.7),
    ("proximite_100", 1.0),
)


def select_by_probability(summary: pd.DataFrame, number: int) -> pd.DataFrame:
    return (
        summary.sort_values(
            ["casa_id", "probability", "poi_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .groupby("casa_id", sort=False)
        .head(number)
    )


def simulate(
    candidates: pd.DataFrame,
    weights: np.ndarray,
    alpha_proximity: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return mean/std scores and TOP1/TOP3 probabilities."""
    score_parts: list[pd.DataFrame] = []
    top1_parts: list[pd.DataFrame] = []
    top3_parts: list[pd.DataFrame] = []
    alpha_quality = 1.0 - alpha_proximity

    for simulation, draw in enumerate(weights, start=1):
        temp = candidates.copy()
        quality = (
            draw[0] * temp["traf_mean"]
            + draw[1] * temp["marcia_mean"]
            + draw[2] * temp["green_mean"]
            + draw[3] * temp["slope_mean"]
        )
        temp["score"] = (
            alpha_proximity * temp["proximity"] + alpha_quality * quality
        )
        temp["simulation"] = simulation
        score_parts.append(
            temp[["casa_id", "poi_id", "cat_funz", "score"]]
        )
        ordered = temp.sort_values(
            ["casa_id", "score", "poi_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        top1_parts.append(ordered.groupby("casa_id", sort=False).head(1))
        top3_parts.append(ordered.groupby("casa_id", sort=False).head(3))

    scores = pd.concat(score_parts, ignore_index=True)
    score_summary = (
        scores.groupby(["casa_id", "poi_id", "cat_funz"], as_index=False)
        .agg(score_mean=("score", "mean"), score_std=("score", "std"))
    )
    top1 = (
        pd.concat(top1_parts, ignore_index=True)
        .groupby(["casa_id", "poi_id"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    top1["probability"] = top1["n"] / N_SIM
    top3 = (
        pd.concat(top3_parts, ignore_index=True)
        .groupby(["casa_id", "poi_id"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    top3["probability"] = top3["n"] / N_SIM
    return score_summary, top1, top3


def validate(
    candidates: pd.DataFrame,
    declared: pd.DataFrame,
    scores: pd.DataFrame,
    top1: pd.DataFrame,
    top3: pd.DataFrame,
) -> tuple[dict[str, float | int], dict[int, int], dict[int, set[int]]]:
    accessible = declared.merge(
        candidates[["casa_id", "poi_id"]],
        on=["casa_id", "poi_id"],
        how="inner",
        validate="one_to_one",
    )
    comparable = sorted(accessible["casa_id"].unique())
    declared_by_child = {
        int(child): set(group["poi_id"].astype(int))
        for child, group in accessible.groupby("casa_id")
    }
    accessible_set = {
        (int(row.casa_id), int(row.poi_id)) for row in accessible.itertuples()
    }

    comparisons: list[dict[str, float | int]] = []
    ranks: list[dict[str, float | int]] = []
    for child in comparable:
        child_scores = (
            scores.loc[scores["casa_id"] == child]
            .sort_values(
                ["score_mean", "poi_id"],
                ascending=[False, True],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )
        declared_ids = declared_by_child[int(child)]
        declared_rows = child_scores.loc[
            child_scores["poi_id"].isin(declared_ids)
        ]
        for row in declared_rows.itertuples():
            alternatives = child_scores.loc[
                (~child_scores["poi_id"].isin(declared_ids))
                & (child_scores["cat_funz"] == row.cat_funz)
            ]
            if alternatives.empty:
                continue
            alternative_score = float(alternatives["score_mean"].mean())
            comparisons.append(
                {
                    "casa_id": int(child),
                    "declared_score": float(row.score_mean),
                    "alternative_score": alternative_score,
                    "delta": float(row.score_mean - alternative_score),
                }
            )
            comparison_set = pd.concat(
                [
                    pd.DataFrame(
                        [
                            {
                                "poi_id": int(row.poi_id),
                                "score_mean": float(row.score_mean),
                            }
                        ]
                    ),
                    alternatives[["poi_id", "score_mean"]],
                ],
                ignore_index=True,
            ).sort_values(
                ["score_mean", "poi_id"],
                ascending=[False, True],
                kind="mergesort",
            ).reset_index(drop=True)
            position = (
                int(
                    comparison_set.index[
                        comparison_set["poi_id"] == int(row.poi_id)
                    ][0]
                )
                + 1
            )
            ranks.append(
                {
                    "casa_id": int(child),
                    "rank": 1.0
                    - (position - 1) / (len(comparison_set) - 1),
                }
            )

    comparison_detail = pd.DataFrame(comparisons)
    comparison = comparison_detail.groupby("casa_id", as_index=False).agg(
        declared_score=("declared_score", "mean"),
        alternative_score=("alternative_score", "mean"),
        delta=("delta", "mean"),
    )
    rank = pd.DataFrame(ranks).groupby("casa_id", as_index=False).agg(
        rank=("rank", "mean")
    )

    chosen_top1 = select_by_probability(top1, 1)
    chosen_top3 = select_by_probability(top3, 3)
    chosen_top1 = chosen_top1.loc[chosen_top1["casa_id"].isin(comparable)]
    chosen_top3 = chosen_top3.loc[chosen_top3["casa_id"].isin(comparable)]
    top1_hits = sum(
        (int(row.casa_id), int(row.poi_id)) in accessible_set
        for row in chosen_top1.itertuples()
    )
    top3_hits_by_child = {
        int(child): int(
            group["poi_id"].isin(declared_by_child[int(child)]).sum()
        )
        for child, group in chosen_top3.groupby("casa_id")
    }
    top3_hits = int(sum(top3_hits_by_child.values()))
    top3_any = int(sum(value > 0 for value in top3_hits_by_child.values()))
    proposals = int(len(chosen_top3))

    random_top1: list[float] = []
    random_top3_any: list[float] = []
    expected_top3_hits: list[float] = []
    for child in comparable:
        n = int((candidates["casa_id"] == child).sum())
        d = len(declared_by_child[int(child)])
        k = min(3, n)
        random_top1.append(d / n)
        expected_top3_hits.append(k * d / n)
        random_top3_any.append(
            1.0
            if n == d
            else 1.0 - math.comb(n - d, k) / math.comb(n, k)
        )

    top1_observed = 100.0 * top1_hits / len(comparable)
    top1_random = 100.0 * float(np.mean(random_top1))
    top3_any_observed = 100.0 * top3_any / len(comparable)
    top3_any_random = 100.0 * float(np.mean(random_top3_any))
    top3_concordance = 100.0 * top3_hits / proposals
    top3_random_concordance = 100.0 * sum(expected_top3_hits) / proposals

    chosen_top1_all = select_by_probability(top1, 1)
    top1_choice = {
        int(row.casa_id): int(row.poi_id) for row in chosen_top1_all.itertuples()
    }
    top3_choice = {
        int(child): set(group["poi_id"].astype(int))
        for child, group in select_by_probability(top3, 3).groupby("casa_id")
    }

    maximum_top1_probability = top1.groupby("casa_id")["probability"].max()
    result = {
        "candidate_routes_n": int(len(candidates)),
        "comparable_children_n": int(len(comparable)),
        "children_in_delta_n": int(len(comparison)),
        "declared_score_mean": float(comparison["declared_score"].mean()),
        "alternative_score_mean": float(
            comparison["alternative_score"].mean()
        ),
        "delta_mean": float(comparison["delta"].mean()),
        "positive_delta_n": int((comparison["delta"] > 0).sum()),
        "rank_normalized_mean": float(rank["rank"].mean()),
        "top1_hits_n": int(top1_hits),
        "top1_observed_pct": top1_observed,
        "top1_random_pct": top1_random,
        "top1_lift": top1_observed / top1_random,
        "top3_any_hits_n": top3_any,
        "top3_any_observed_pct": top3_any_observed,
        "top3_any_random_pct": top3_any_random,
        "top3_any_lift": top3_any_observed / top3_any_random,
        "top3_hits_n": top3_hits,
        "top3_proposals_n": proposals,
        "top3_concordance_pct": top3_concordance,
        "top3_random_concordance_pct": top3_random_concordance,
        "top3_concordance_lift": (
            top3_concordance / top3_random_concordance
        ),
        "top3_recall_pct": 100.0 * top3_hits / len(accessible),
        "score_std_mean": float(scores["score_std"].mean()),
        "score_std_median": float(scores["score_std"].median()),
        "score_std_p95": float(scores["score_std"].quantile(0.95)),
        "score_std_max": float(scores["score_std"].max()),
        "top1_probability_max_mean": float(
            maximum_top1_probability.mean()
        ),
        "top1_probability_max_median": float(
            maximum_top1_probability.median()
        ),
        "top1_probability_max_min": float(
            maximum_top1_probability.min()
        ),
        "children_top1_probability_ge_50_n": int(
            (maximum_top1_probability >= 0.50).sum()
        ),
        "children_top1_probability_ge_75_n": int(
            (maximum_top1_probability >= 0.75).sum()
        ),
        "children_top1_probability_ge_90_n": int(
            (maximum_top1_probability >= 0.90).sum()
        ),
    }
    return result, top1_choice, top3_choice


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    routes = pd.read_csv(ROUTES)
    declared = pd.read_csv(DECLARED)
    candidates = routes.loc[
        (~routes["poi_name"].str.contains("scuola", case=False, na=False))
        & (routes["tempo_min"] <= THRESHOLD)
    ].copy()
    candidates["proximity"] = 1.0 - candidates["tempo_min"] / THRESHOLD

    rng = np.random.RandomState(RANDOM_SEED)
    weights = rng.dirichlet([1, 1, 1, 1], size=N_SIM)

    rows: list[dict[str, float | int | str]] = []
    choices: dict[str, tuple[dict[int, int], dict[int, set[int]]]] = {}
    central_scores: pd.DataFrame | None = None
    for name, alpha_proximity in SCENARIOS:
        scores, top1, top3 = simulate(candidates, weights, alpha_proximity)
        if name == "proximite_30_qualite_70":
            central_scores = scores.copy()
        result, top1_choice, top3_choice = validate(
            candidates, declared, scores, top1, top3
        )
        result.update(
            {
                "scenario": name,
                "alpha_proximity": alpha_proximity,
                "alpha_quality": 1.0 - alpha_proximity,
            }
        )
        rows.append(result)
        choices[name] = (top1_choice, top3_choice)

    table = pd.DataFrame(rows)
    central_top1, central_top3 = choices["proximite_30_qualite_70"]
    agreement_rows: list[dict[str, float | int | str]] = []
    for name, _ in SCENARIOS:
        top1_choice, top3_choice = choices[name]
        common = sorted(set(central_top1) & set(top1_choice))
        top1_agreement = np.mean(
            [central_top1[child] == top1_choice[child] for child in common]
        )
        top3_jaccard = np.mean(
            [
                len(central_top3[child] & top3_choice[child])
                / len(central_top3[child] | top3_choice[child])
                for child in common
            ]
        )
        agreement_rows.append(
            {
                "scenario": name,
                "children_n": len(common),
                "top1_agreement_with_central_pct": 100.0 * top1_agreement,
                "top3_mean_jaccard_with_central": float(top3_jaccard),
            }
        )
    agreement = pd.DataFrame(agreement_rows)

    if central_scores is None:
        raise ValueError("Scores centraux absents")
    equal_weight = candidates[["casa_id", "poi_id"]].copy()
    equal_weight["score_equal_weights"] = (
        0.3 * candidates["proximity"]
        + 0.7
        * 0.25
        * candidates[
            ["traf_mean", "marcia_mean", "green_mean", "slope_mean"]
        ].sum(axis=1)
    )
    equal_weight_check = central_scores.merge(
        equal_weight,
        on=["casa_id", "poi_id"],
        how="inner",
        validate="one_to_one",
    )
    equal_weight_check["absolute_difference"] = (
        equal_weight_check["score_mean"]
        - equal_weight_check["score_equal_weights"]
    ).abs()

    diagnostic = {
        "inputs": {
            "routes": str(ROUTES),
            "declared_pairs": str(DECLARED),
            "threshold_minutes": THRESHOLD,
            "candidate_routes": int(len(candidates)),
            "houses": int(candidates["casa_id"].nunique()),
            "pois": int(candidates["poi_id"].nunique()),
        },
        "monte_carlo": {
            "simulations": N_SIM,
            "seed": RANDOM_SEED,
            "law": "Dirichlet(1,1,1,1)",
            "theoretical_expected_weight": [0.25, 0.25, 0.25, 0.25],
            "realized_mean_weights": weights.mean(axis=0).tolist(),
            "realized_weight_sums_min": float(weights.sum(axis=1).min()),
            "realized_weight_sums_max": float(weights.sum(axis=1).max()),
            "equal_weight_score_verification": {
                "pairs_n": int(len(equal_weight_check)),
                "mean_absolute_difference": float(
                    equal_weight_check["absolute_difference"].mean()
                ),
                "maximum_absolute_difference": float(
                    equal_weight_check["absolute_difference"].max()
                ),
                "note": (
                    "The finite-simulation mean only approximates the exact "
                    "equal-weight expectation because the realized means of "
                    "the 1,000 sampled weights are close to, but not exactly, "
                    "0.25."
                ),
            },
        },
        "interpretation": {
            "mean_quality_equivalence": (
                "Because the quality score is linear in the four weights, "
                "its expectation under the symmetric Dirichlet law equals "
                "the equal-weight quality score. Monte Carlo remains needed "
                "for TOP1/TOP3 rank acceptability frequencies."
            ),
            "framework": (
                "The procedure is SMAA-inspired but is not a complete SMAA "
                "implementation: it samples the criterion-weight simplex and "
                "reports rank frequencies, without estimating central weight "
                "vectors or confidence factors."
            ),
        },
    }

    table.to_csv(OUT / "sensibilite_proximite_qualite.csv", index=False)
    agreement.to_csv(OUT / "stabilite_classements_vs_scenario_central.csv", index=False)
    (OUT / "diagnostic_dirichlet_et_cadre.json").write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    central = table.loc[
        table["scenario"] == "proximite_30_qualite_70"
    ].iloc[0]
    if int(central["candidate_routes_n"]) != 720:
        raise ValueError("Nombre de routes candidates central inattendu")
    if int(central["comparable_children_n"]) != 28:
        raise ValueError("Nombre d'enfants comparables inattendu")
    if not math.isclose(
        float(central["delta_mean"]),
        0.03147660311388399,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Le scénario central ne reproduit pas le delta validé")

    print(table.to_string(index=False))
    print("\nSTABILITE PAR RAPPORT AU SCENARIO 30/70")
    print(agreement.to_string(index=False))
    print("\nPOIDS DIRICHLET MOYENS REALISES")
    print(weights.mean(axis=0))


if __name__ == "__main__":
    main()
