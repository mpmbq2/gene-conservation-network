import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Hypothesis Evaluation: Gene Network Properties and Conservation

        This notebook evaluates two core hypotheses about the relationship between
        gene coexpression network properties and cross-species ortholog conservation:

        1. **Hub genes are more conserved:** High-degree genes in coexpression networks
           have higher ortholog confidence scores across species.
        2. **Hub genes have more ambiguous orthologs:** High-degree genes have more
           ortholog candidates (many-to-many mappings), reflecting evolutionary
           redundancy for critical genes.

        We use 6 WORMHOLE species (worm, fly, zebrafish, human, mouse, yeast),
        COXPRESdb coexpression data (unified variant), and WORMHOLE ortholog scores.
        All analysis functions accept the association threshold as a parameter so results
        can be compared across thresholds (z >= 3, 4, 5, 6) to assess robustness.
        """
    )
    return (mo,)


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    from gene_conservation_network.analysis.correlation import (
        compute_correlation_matrix,
        compute_pairwise_correlations,
    )
    from gene_conservation_network.analysis.hypotheses import (
        describe_all_hypotheses,
        describe_hub_ambiguity,
        describe_hub_conservation,
    )
    from gene_conservation_network.analysis.visualization import (
        plot_correlation_heatmap,
        plot_degree_distribution,
        plot_feature_scatter,
        plot_species_comparison,
    )
    from gene_conservation_network.data.species import (
        ALL_SPECIES,
        all_species_pairs,
        species_by_wormhole_code,
    )

    MERGED_DIR = Path("data/04_merged")
    NETWORK_DIR = Path("data/03_features/network")

    THRESHOLDS = [3, 4, 5, 6]
    NETWORK_COLS = ["degree", "weighted_degree", "betweenness", "eigenvector", "pagerank"]
    ORTHOLOG_COLS = [
        "ortholog_count",
        "max_wormhole_score",
        "mean_wormhole_score",
        "max_votes",
        "vote_entropy",
    ]
    return (
        MERGED_DIR,
        NETWORK_COLS,
        NETWORK_DIR,
        ORTHOLOG_COLS,
        THRESHOLDS,
        all_species_pairs,
        compute_pairwise_correlations,
        describe_all_hypotheses,
        describe_hub_ambiguity,
        describe_hub_conservation,
        np,
        pl,
        plt,
        plot_correlation_heatmap,
        plot_degree_distribution,
        plot_feature_scatter,
        plot_species_comparison,
        species_by_wormhole_code,
    )


@app.cell
def _(MERGED_DIR, all_species_pairs, mo, pl):
    mo.md("## 1. Data Overview")

    # Load all merged files into a summary table
    _summary_rows = []
    for _pair in all_species_pairs():
        for _t in [3, 4, 5, 6]:
            _path = MERGED_DIR / f"{_pair.wormhole_prefix}_t{_t}.parquet"
            if _path.exists():
                _df = pl.read_parquet(_path)
                _summary_rows.append(
                    {
                        "pair": f"{_pair.query.common_name} -> {_pair.target.common_name}",
                        "pair_code": _pair.wormhole_prefix,
                        "threshold": _t,
                        "n_genes": len(_df),
                        "mean_degree": round(_df["degree"].mean(), 1),
                        "median_degree": _df["degree"].median(),
                        "mean_ortholog_count": round(_df["ortholog_count"].mean(), 1),
                        "mean_max_score": round(_df["max_wormhole_score"].mean(), 3),
                    }
                )

    data_summary = pl.DataFrame(_summary_rows)
    data_summary
    return (data_summary,)


@app.cell
def _(data_summary, mo, pl):
    # Show gene counts at threshold=5 per species pair
    t5_summary = (
        data_summary.filter(pl.col("threshold") == 5)
        .sort("pair")
        .select("pair", "n_genes", "mean_degree", "mean_ortholog_count", "mean_max_score")
    )

    mo.md(
        f"""
        ### Merged dataset sizes at threshold = 5

        {mo.as_html(t5_summary)}

        Each row represents a species pair where network features (from the query
        species' coexpression graph) are joined with ortholog features (from the
        query→target ortholog data) via gene ID resolution.
        """
    )
    return


@app.cell
def _(NETWORK_DIR, mo, np, pl, plt):
    mo.md("## 2. Network Topology")

    # Degree distributions for all 6 species at threshold=5
    _fig, _axes = plt.subplots(2, 3, figsize=(14, 8))
    _axes = _axes.flatten()

    _species_labels = {
        "cel": "C. elegans (worm)",
        "dme": "D. melanogaster (fly)",
        "dre": "D. rerio (zebrafish)",
        "hsa": "H. sapiens (human)",
        "mmu": "M. musculus (mouse)",
        "sce": "S. cerevisiae (yeast)",
    }

    for _i, (_code, _label) in enumerate(_species_labels.items()):
        _path = NETWORK_DIR / f"{_code}_u_t5.parquet"
        if _path.exists():
            _df = pl.read_parquet(_path)
            _degrees = _df["degree"].to_numpy()
            _unique, _counts = np.unique(_degrees, return_counts=True)
            _axes[_i].scatter(_unique, _counts, s=5, alpha=0.5, color="steelblue")
            _axes[_i].set_xscale("log")
            _axes[_i].set_yscale("log")
            _axes[_i].set_title(_label, fontsize=10)
            _axes[_i].set_xlabel("Degree")
            _axes[_i].set_ylabel("Frequency")
            _axes[_i].text(
                0.95,
                0.95,
                f"n={len(_df):,}",
                transform=_axes[_i].transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )

    _fig.suptitle("Degree Distributions (threshold = 5, log-log)", fontsize=13, y=1.01)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Hypothesis 1: Hub Genes Are More Conserved

    We test whether genes with higher network centrality (degree, betweenness,
    eigenvector, etc.) tend to have higher ortholog confidence scores
    (max WORMHOLE score, max votes).

    A positive Spearman correlation supports the hypothesis.
    """)
    return


@app.cell
def _(
    MERGED_DIR,
    describe_hub_conservation,
    pl,
    plt,
    species_by_wormhole_code,
):
    # Run H1 for all species pairs at threshold=5 using degree vs max_wormhole_score
    _h1_results = []
    for _pair_code in sorted(
        [f.stem.rsplit("_", 1)[0] for f in MERGED_DIR.glob("*_t5.parquet")]
    ):
        _path = MERGED_DIR / f"{_pair_code}_t5.parquet"
        _merged = pl.read_parquet(_path)
        _query = species_by_wormhole_code(_pair_code[:2])
        _target = species_by_wormhole_code(_pair_code[2:])
        _result = describe_hub_conservation(_merged, _query, _target, threshold=5.0)
        _h1_results.append(_result)

    # Forest plot
    _fig, _ax = plt.subplots(figsize=(10, 10))
    _labels = []
    _values = []
    for _r in sorted(_h1_results, key=lambda x: x.statistic_value, reverse=True):
        _labels.append(f"{_r.species} -> {_r.target_species} (n={_r.n_genes:,})")
        _values.append(_r.statistic_value)

    _y_pos = range(len(_labels))
    _colors = ["#2171b5" if v > 0 else "#cb181d" for v in _values]
    _ax.barh(_y_pos, _values, color=_colors, alpha=0.7, height=0.7)
    _ax.axvline(x=0, color="gray", linewidth=0.8, linestyle="--")
    _ax.set_yticks(_y_pos)
    _ax.set_yticklabels(_labels, fontsize=8)
    _ax.set_xlabel("Spearman r (degree vs max WORMHOLE score)")
    _ax.set_title("Hypothesis 1: Hub genes are more conserved (threshold = 5)")
    _ax.invert_yaxis()
    _fig.tight_layout()
    _fig
    
    h1_results = _h1_results
    return (h1_results,)


@app.cell
def _(h1_results, mo, pl):
    # Summary table for H1
    _h1_df = pl.DataFrame(
        [
            {
                "species": r.species,
                "target": r.target_species,
                "spearman_r": round(r.statistic_value, 4),
                "n_genes": r.n_genes,
            }
            for r in sorted(h1_results, key=lambda x: x.statistic_value, reverse=True)
        ]
    )

    _positive = _h1_df.filter(pl.col("spearman_r") > 0)
    _negative = _h1_df.filter(pl.col("spearman_r") < 0)
    _strong = _h1_df.filter(pl.col("spearman_r").abs() > 0.2)

    mo.md(
        f"""
        ### Hypothesis 1 Summary (threshold = 5)

        - **{len(_positive)}/{len(_h1_df)}** pairs show positive correlation
          (hub genes tend to be more conserved)
        - **{len(_negative)}/{len(_h1_df)}** pairs show negative correlation
        - **{len(_strong)}/{len(_h1_df)}** pairs have |r| > 0.2

        {mo.as_html(_h1_df)}
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Hypothesis 2: Hub Genes Have More Ambiguous Orthologs

    We test whether genes with higher network centrality tend to have more ortholog
    candidates (higher ortholog count, higher vote entropy).

    A positive Spearman correlation supports the hypothesis that important genes
    have more complex ortholog relationships (possibly due to evolutionary
    redundancy through duplication).
    """)
    return


@app.cell
def _(MERGED_DIR, describe_hub_ambiguity, pl, plt, species_by_wormhole_code):
    # Run H2 for all species pairs at threshold=5 using degree vs ortholog_count
    _h2_results = []
    for _pair_code in sorted(
        [f.stem.rsplit("_", 1)[0] for f in MERGED_DIR.glob("*_t5.parquet")]
    ):
        _path = MERGED_DIR / f"{_pair_code}_t5.parquet"
        _merged = pl.read_parquet(_path)
        _query = species_by_wormhole_code(_pair_code[:2])
        _target = species_by_wormhole_code(_pair_code[2:])
        _result = describe_hub_ambiguity(_merged, _query, _target, threshold=5.0)
        _h2_results.append(_result)

    # Forest plot
    _fig, _ax = plt.subplots(figsize=(10, 10))
    _labels = []
    _values = []
    for _r in sorted(_h2_results, key=lambda x: x.statistic_value, reverse=True):
        _labels.append(f"{_r.species} -> {_r.target_species} (n={_r.n_genes:,})")
        _values.append(_r.statistic_value)

    _y_pos = range(len(_labels))
    _colors = ["#2171b5" if v > 0 else "#cb181d" for v in _values]
    _ax.barh(_y_pos, _values, color=_colors, alpha=0.7, height=0.7)
    _ax.axvline(x=0, color="gray", linewidth=0.8, linestyle="--")
    _ax.set_yticks(_y_pos)
    _ax.set_yticklabels(_labels, fontsize=8)
    _ax.set_xlabel("Spearman r (degree vs ortholog count)")
    _ax.set_title("Hypothesis 2: Hub genes have more ambiguous orthologs (threshold = 5)")
    _ax.invert_yaxis()
    _fig.tight_layout()
    _fig
    
    h2_results = _h2_results
    return (h2_results,)


@app.cell
def _(h2_results, mo, pl):
    _h2_df = pl.DataFrame(
        [
            {
                "species": r.species,
                "target": r.target_species,
                "spearman_r": round(r.statistic_value, 4),
                "n_genes": r.n_genes,
            }
            for r in sorted(h2_results, key=lambda x: x.statistic_value, reverse=True)
        ]
    )

    _positive = _h2_df.filter(pl.col("spearman_r") > 0)
    _negative = _h2_df.filter(pl.col("spearman_r") < 0)
    _strong = _h2_df.filter(pl.col("spearman_r").abs() > 0.2)

    mo.md(
        f"""
        ### Hypothesis 2 Summary (threshold = 5)

        - **{len(_positive)}/{len(_h2_df)}** pairs show positive correlation
          (hub genes tend to have more ortholog candidates)
        - **{len(_negative)}/{len(_h2_df)}** pairs show negative correlation
        - **{len(_strong)}/{len(_h2_df)}** pairs have |r| > 0.2

        {mo.as_html(_h2_df)}
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Detailed Scatter Plots (Fly -> Human)

    A closer look at fly->human as a representative species pair.
    These scatter plots show the raw relationship between each network
    feature and each ortholog feature.
    """)
    return


@app.cell
def _(MERGED_DIR, pl, plt, plot_feature_scatter):
    # Load fly->human at threshold=5
    dmhs = pl.read_parquet(MERGED_DIR / "dmhs_t5.parquet")

    # Scatter plots: degree vs key ortholog features
    _fig, _axes = plt.subplots(2, 3, figsize=(15, 9))
    _scatter_pairs = [
        ("degree", "max_wormhole_score"),
        ("degree", "ortholog_count"),
        ("degree", "vote_entropy"),
        ("betweenness", "max_wormhole_score"),
        ("eigenvector", "max_wormhole_score"),
        ("pagerank", "ortholog_count"),
    ]

    for _ax, (_x, _y) in zip(_axes.flatten(), _scatter_pairs):
        plot_feature_scatter(dmhs, _x, _y, species_label=f"fly -> human", ax=_ax)

    _fig.suptitle("Feature Relationships: fly -> human (threshold = 5)", fontsize=13, y=1.01)
    _fig.tight_layout()
    _fig
    return (dmhs,)


@app.cell
def _(
    NETWORK_COLS,
    ORTHOLOG_COLS,
    compute_pairwise_correlations,
    dmhs,
    mo,
    plt,
    plot_correlation_heatmap,
):
    # Pairwise correlation heatmap for fly->human
    _corr = compute_pairwise_correlations(
        dmhs,
        network_cols=NETWORK_COLS,
        ortholog_cols=ORTHOLOG_COLS,
    )

    _fig, _ax = plt.subplots(figsize=(9, 6))
    plot_correlation_heatmap(_corr, title="Network vs Ortholog Feature Correlations: fly -> human (t=5)", ax=_ax)
    _fig.tight_layout()

    mo.md(
        f"""
        ### Correlation Heatmap: fly -> human (threshold = 5)

        Rows are network features, columns are ortholog features.
        Blue = positive correlation, Red = negative correlation.
        """
    )

    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Threshold Sensitivity Analysis

    A key concern: do the results depend on the choice of association threshold?
    If a finding only holds at one threshold, it is likely an artifact.

    Below we track how the degree-conservation and degree-ambiguity correlations
    change across thresholds (z >= 3, 4, 5, 6) for all species pairs.
    """)
    return


@app.cell
def _(
    MERGED_DIR,
    THRESHOLDS,
    all_species_pairs,
    describe_hub_ambiguity,
    describe_hub_conservation,
    pl,
):
    # Threshold sensitivity: compute H1 and H2 at all thresholds for all pairs
    _sensitivity_rows = []
    for _pair in all_species_pairs():
        for _t in THRESHOLDS:
            _path = MERGED_DIR / f"{_pair.wormhole_prefix}_t{_t}.parquet"
            if not _path.exists():
                continue
            _merged = pl.read_parquet(_path)
            _h1 = describe_hub_conservation(_merged, _pair.query, _pair.target, float(_t))
            _h2 = describe_hub_ambiguity(_merged, _pair.query, _pair.target, float(_t))
            _sensitivity_rows.append(
                {
                    "pair": f"{_pair.query.common_name}->{_pair.target.common_name}",
                    "pair_code": _pair.wormhole_prefix,
                    "threshold": _t,
                    "h1_r": _h1.statistic_value,
                    "h1_n": _h1.n_genes,
                    "h2_r": _h2.statistic_value,
                    "h2_n": _h2.n_genes,
                }
            )

    sensitivity = pl.DataFrame(_sensitivity_rows)
    return (sensitivity,)


@app.cell
def _(THRESHOLDS, pl, plt, sensitivity):
    # Plot H1 threshold sensitivity: one line per species pair
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 6))

    _pairs = sensitivity["pair"].unique().sort().to_list()
    _cmap = plt.cm.get_cmap("tab20", len(_pairs))

    for _i, _pair_name in enumerate(_pairs):
        _pair_data = sensitivity.filter(pl.col("pair") == _pair_name).sort("threshold")
        _ts = _pair_data["threshold"].to_numpy()
        _h1_vals = _pair_data["h1_r"].to_numpy()
        _h2_vals = _pair_data["h2_r"].to_numpy()

        _ax1.plot(_ts, _h1_vals, alpha=0.4, linewidth=1, color=_cmap(_i))
        _ax2.plot(_ts, _h2_vals, alpha=0.4, linewidth=1, color=_cmap(_i))

    # Add mean across all pairs
    for _t in THRESHOLDS:
        _t_data = sensitivity.filter(pl.col("threshold") == _t)
        _ax1.scatter(_t, _t_data["h1_r"].mean(), color="black", s=60, zorder=5, marker="D")
        _ax2.scatter(_t, _t_data["h2_r"].mean(), color="black", s=60, zorder=5, marker="D")

    _ax1.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
    _ax1.set_xlabel("Association Threshold (z-score)")
    _ax1.set_ylabel("Spearman r")
    _ax1.set_title("H1: degree vs max WORMHOLE score")
    _ax1.set_xticks(THRESHOLDS)

    _ax2.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
    _ax2.set_xlabel("Association Threshold (z-score)")
    _ax2.set_ylabel("Spearman r")
    _ax2.set_title("H2: degree vs ortholog count")
    _ax2.set_xticks(THRESHOLDS)

    _fig.suptitle(
        "Threshold Sensitivity (each line = one species pair, diamonds = mean)",
        fontsize=12,
        y=1.02,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(THRESHOLDS, mo, pl, sensitivity):
    # Summary statistics across thresholds
    _threshold_summary_rows = []
    for _t in THRESHOLDS:
        _t_data = sensitivity.filter(pl.col("threshold") == _t)
        _threshold_summary_rows.append(
            {
                "threshold": _t,
                "h1_mean_r": round(_t_data["h1_r"].mean(), 4),
                "h1_median_r": round(_t_data["h1_r"].median(), 4),
                "h1_pct_positive": round(
                    100 * (_t_data["h1_r"] > 0).sum() / len(_t_data), 1
                ),
                "h2_mean_r": round(_t_data["h2_r"].mean(), 4),
                "h2_median_r": round(_t_data["h2_r"].median(), 4),
                "h2_pct_positive": round(
                    100 * (_t_data["h2_r"] > 0).sum() / len(_t_data), 1
                ),
                "mean_n_genes": int(_t_data["h1_n"].mean()),
            }
        )

    threshold_summary = pl.DataFrame(_threshold_summary_rows)

    mo.md(
        f"""
        ### Threshold Sensitivity Summary

        {mo.as_html(threshold_summary)}

        - **h1_pct_positive**: % of species pairs where degree-conservation
          correlation is positive
        - **h2_pct_positive**: % of species pairs where degree-ambiguity
          correlation is positive
        - **mean_n_genes**: average number of genes in the merged dataset

        A finding is robust if the sign and magnitude are consistent across thresholds.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. Multi-Metric Analysis

    The hypotheses can be tested with different network centrality measures
    (not just degree). Here we compare how different hub metrics correlate with
    conservation and ambiguity features.
    """)
    return


@app.cell
def _(
    MERGED_DIR,
    NETWORK_COLS,
    ORTHOLOG_COLS,
    compute_pairwise_correlations,
    mo,
    pl,
    plt,
    plot_correlation_heatmap,
):
    # Average correlations across ALL species pairs at threshold=5
    _all_corrs = []
    for _f in sorted(MERGED_DIR.glob("*_t5.parquet")):
        _merged = pl.read_parquet(_f)
        _corr = compute_pairwise_correlations(_merged, NETWORK_COLS, ORTHOLOG_COLS)
        _all_corrs.append(_corr)

    # Stack and average
    _stacked = pl.concat(_all_corrs)
    avg_corr = (
        _stacked.group_by("network_feature", "ortholog_feature")
        .agg(
            pl.col("correlation").mean().alias("correlation"),
            pl.col("n").mean().alias("n"),
        )
        .sort("network_feature", "ortholog_feature")
    )

    _fig, _ax = plt.subplots(figsize=(9, 6))
    plot_correlation_heatmap(
        avg_corr,
        title="Average Correlation Across All 30 Species Pairs (threshold = 5)",
        ax=_ax,
    )
    _fig.tight_layout()

    mo.md(
        """
        ### Average Network-Ortholog Correlations

        This heatmap averages correlations across all 30 directed species pairs
        at threshold = 5. It shows which network features are most consistently
        associated with which ortholog features.
        """
    )

    _fig
    return (avg_corr,)


@app.cell
def _(avg_corr, mo):
    # Show the raw averaged correlation table
    _avg_wide = avg_corr.pivot(on="ortholog_feature", index="network_feature", values="correlation")

    mo.md(
        f"""
        ### Average Correlation Table

        {mo.as_html(_avg_wide)}
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8. Per-Query-Species Breakdown

    Different query species may show different patterns. Below we break down
    the degree-conservation correlation by which species' coexpression network
    we are analyzing.
    """)
    return


@app.cell
def _(
    MERGED_DIR,
    describe_hub_conservation,
    pl,
    plt,
    species_by_wormhole_code,
):
    # Group H1 results by query species at threshold=5
    _by_query = {}
    for _f in sorted(MERGED_DIR.glob("*_t5.parquet")):
        _pair_code = _f.stem.rsplit("_", 1)[0]
        _query_code = _pair_code[:2]
        _target_code = _pair_code[2:]
        _query = species_by_wormhole_code(_query_code)
        _target = species_by_wormhole_code(_target_code)

        _merged = pl.read_parquet(_f)
        _r = describe_hub_conservation(_merged, _query, _target, 5.0)

        if _query.common_name not in _by_query:
            _by_query[_query.common_name] = []
        _by_query[_query.common_name].append(_r.statistic_value)

    _fig, _ax = plt.subplots(figsize=(8, 5))
    _species_names = sorted(_by_query.keys())
    _positions = range(len(_species_names))
    _bp = _ax.boxplot(
        [_by_query[s] for s in _species_names],
        labels=_species_names,
        patch_artist=True,
    )
    for _patch in _bp["boxes"]:
        _patch.set_facecolor("steelblue")
        _patch.set_alpha(0.6)

    _ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
    _ax.set_ylabel("Spearman r (degree vs max WORMHOLE score)")
    _ax.set_xlabel("Query species (coexpression network)")
    _ax.set_title("H1 Correlation Distribution by Query Species (threshold = 5)")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 9. Full Hypothesis Results Table

    Complete results for all hypothesis tests across all species pairs at
    threshold = 5. This table includes multiple hub metrics crossed with
    multiple conservation and ambiguity metrics.
    """)
    return


@app.cell
def _(MERGED_DIR, describe_all_hypotheses, mo, pl, species_by_wormhole_code):
    _all_hypothesis_rows = []
    for _f in sorted(MERGED_DIR.glob("*_t5.parquet")):
        _pair_code = _f.stem.rsplit("_", 1)[0]
        _query = species_by_wormhole_code(_pair_code[:2])
        _target = species_by_wormhole_code(_pair_code[2:])
        _merged = pl.read_parquet(_f)
        _results = describe_all_hypotheses(_merged, _query, _target, 5.0)
        for _r in _results:
            _all_hypothesis_rows.append(
                {
                    "hypothesis": _r.name,
                    "species": _r.species,
                    "target": _r.target_species,
                    "r": round(_r.statistic_value, 4),
                    "n": _r.n_genes,
                    "summary": _r.summary,
                }
            )

    all_hypotheses_df = pl.DataFrame(_all_hypothesis_rows)

    mo.md(
        f"""
        ### All Hypothesis Results (threshold = 5)

        Total tests: **{len(all_hypotheses_df)}** ({len(all_hypotheses_df.filter(pl.col('hypothesis') == 'hub_conservation'))} conservation + {len(all_hypotheses_df.filter(pl.col('hypothesis') == 'hub_ambiguity'))} ambiguity)

        {mo.as_html(all_hypotheses_df.select('hypothesis', 'species', 'target', 'r', 'n'))}
        """
    )
    return (all_hypotheses_df,)


@app.cell
def _(all_hypotheses_df, mo, pl):
    # Summary: average r by hypothesis
    _hyp_summary = (
        all_hypotheses_df.group_by("hypothesis")
        .agg(
            pl.col("r").mean().alias("mean_r"),
            pl.col("r").median().alias("median_r"),
            pl.col("r").std().alias("std_r"),
            (pl.col("r") > 0).sum().alias("n_positive"),
            pl.len().alias("n_total"),
        )
        .with_columns((pl.col("n_positive") / pl.col("n_total") * 100).round(1).alias("pct_positive"))
        .sort("hypothesis")
    )

    mo.md(
        f"""
        ### Aggregate Summary

        {mo.as_html(_hyp_summary)}

        **Interpretation guide:**
        - `mean_r` / `median_r`: central tendency of correlations across all species pairs and metrics
        - `pct_positive`: what percentage of tests show the hypothesized direction
        - A consistent signal should show high `pct_positive` and non-trivial `mean_r`
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 10. Conclusions

    Review the plots and tables above to assess:

    1. **Is the hub-conservation relationship consistent?**
       Check if the majority of species pairs show positive correlations
       (Section 3) and whether the effect is robust across thresholds (Section 6).

    2. **Is the hub-ambiguity relationship consistent?**
       Same assessment for ortholog count / vote entropy (Section 4).

    3. **Which network metric is most informative?**
       The multi-metric heatmap (Section 7) reveals whether degree, betweenness,
       or eigenvector centrality best predicts conservation features.

    4. **Are there species-specific patterns?**
       The per-query-species boxplot (Section 8) reveals whether certain
       species' networks show stronger signal than others.

    **Caveats:**
    - This is a descriptive analysis. Correlations describe association, not causation.
    - Gene length, expression level, and functional category are potential confounders
      not controlled for in this iteration.
    - P-values are intentionally omitted. The focus is on effect sizes and their
      consistency across species and thresholds.
    """)
    return


if __name__ == "__main__":
    app.run()
