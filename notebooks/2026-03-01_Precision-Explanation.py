import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    np.random.seed(42)

    # --- Define categories ---
    age_groups = ["40-49", "50-59", "60-69", "70-79", "80+"]
    race_ethnicity = [
        "Non-Hispanic White",
        "Non-Hispanic Black",
        "Hispanic",
        "Asian/Pacific Islander",
        "American Indian/Alaska Native",
    ]

    # --- Helper function to generate fake RR and 95% CIs ---
    def fake_rr(center, spread=0.1, n=1):
        """Generate a plausible-looking RR with asymmetric CIs."""
        rr = np.random.normal(center, spread, n)
        rr = np.clip(rr, 0.5, 3.0)
        se = np.random.uniform(0.05, 0.20, n)
        lower = np.exp(np.log(rr) - 1.96 * se)
        upper = np.exp(np.log(rr) + 1.96 * se)
        return rr, lower, upper

    # --- Age group results (reference: 40-49) ---
    age_centers = [1.00, 1.25, 1.55, 1.90, 2.30]  # increasing trend with age
    age_rows = []

    for i, age in enumerate(age_groups):
        if age == "40-49":
            age_rows.append({
                "Variable": "Age Group",
                "Category": age,
                "RR": 1.00,
                "CI_Lower": np.nan,
                "CI_Upper": np.nan,
                "Reference": True,
            })
        else:
            rr, lo, hi = fake_rr(age_centers[i])
            age_rows.append({
                "Variable": "Age Group",
                "Category": age,
                "RR": round(rr[0], 2),
                "CI_Lower": round(lo[0], 2),
                "CI_Upper": round(hi[0], 2),
                "Reference": False,
            })

    # --- Race/ethnicity results (reference: Non-Hispanic White) ---
    race_centers = [1.00, 1.40, 1.20, 0.85, 1.60]
    race_rows = []

    for i, race in enumerate(race_ethnicity):
        if race == "Non-Hispanic White":
            race_rows.append({
                "Variable": "Race/Ethnicity",
                "Category": race,
                "RR": 1.00,
                "CI_Lower": np.nan,
                "CI_Upper": np.nan,
                "Reference": True,
            })
        else:
            rr, lo, hi = fake_rr(race_centers[i])
            race_rows.append({
                "Variable": "Race/Ethnicity",
                "Category": race,
                "RR": round(rr[0], 2),
                "CI_Lower": round(lo[0], 2),
                "CI_Upper": round(hi[0], 2),
                "Reference": False,
            })

    # --- Combine into a single DataFrame ---
    #df = pd.DataFrame(age_rows + race_rows)
    df = pd.DataFrame(race_rows)
    df["95% CI"] = df.apply(
        lambda row: "Ref"
        if row["Reference"]
        else f"({row['CI_Lower']:.2f}, {row['CI_Upper']:.2f})",
        axis=1,
    )

    df
    return df, np, pd, plt


@app.cell
def _(df, plt):
    import matplotlib.patches as mpatches

    def plot_forest(df, title="Forest Plot of Relative Risk"):
        """
        Plot a forest plot from a GLM results DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: 'Variable', 'Category', 'RR', 'CI_Lower', 'CI_Upper', 'Reference'
        title : str
            Title for the plot.
        """

        # --- Assign colors per variable group ---
        groups = df["Variable"].unique()
        palette = ["#2166ac", "#d6604d", "#4dac26"]
        color_map = {group: palette[i % len(palette)] for i, group in enumerate(groups)}

        # --- Build plot rows (add blank separator rows between groups) ---
        rows = []
        prev_var = None
        for _, row in df.iterrows():
            if prev_var is not None and row["Variable"] != prev_var:
                rows.append(None)  # separator
            rows.append(row)
            prev_var = row["Variable"]

        n_rows = len(rows)

        # --- Size figure based on actual data rows only ---
        n_data_rows = sum(1 for r in rows if r is not None)
        fig_height = max(4, n_data_rows * 0.45 + 1.5)

        fig, (ax_table, ax_plot) = plt.subplots(
            1, 2,
            figsize=(12, fig_height),
            gridspec_kw={"width_ratios": [2, 3]}
        )

        # Use tighter y spacing: one unit per row, separator rows get 0.4 units
        y_positions = []
        y = 0
        for row in reversed(rows):
            y_positions.insert(0, y)
            if row is None:
                y += 0.4   # small gap for separators
            else:
                y += 1.0   # full unit per data row

        y_max = y  # total height of y axis

        # ------------------------------------------------------------------ #
        #  LEFT PANEL — text table                                            #
        # ------------------------------------------------------------------ #
        ax_table.set_xlim(0, 1)
        ax_table.set_ylim(-0.5, y_max)
        ax_table.axis("off")

        # Column headers
        ax_table.text(0.02, y_max - 0.3, "Category",
                      fontweight="bold", fontsize=9, va="center")
        ax_table.text(0.72, y_max - 0.3, "RR",
                      fontweight="bold", fontsize=9, va="center", ha="right")
        ax_table.text(0.95, y_max - 0.3, "95% CI",
                      fontweight="bold", fontsize=9, va="center", ha="right")

        for y_pos, row in zip(y_positions, rows):
            if row is None:
                continue

            is_ref = row["Reference"]
            color = color_map[row["Variable"]]

            rr_text = "Ref" if is_ref else f"{row['RR']:.2f}"
            ci_text = "" if is_ref else f"({row['CI_Lower']:.2f}, {row['CI_Upper']:.2f})"

            ax_table.text(0.02, y_pos, row["Category"], fontsize=8.5, va="center",
                          color=color, fontweight="bold" if is_ref else "normal")
            ax_table.text(0.72, y_pos, rr_text, fontsize=8.5, va="center",
                          ha="right", color="black")
            ax_table.text(0.95, y_pos, ci_text, fontsize=8.5, va="center",
                          ha="right", color="dimgray")

        # ------------------------------------------------------------------ #
        #  RIGHT PANEL — forest plot                                          #
        # ------------------------------------------------------------------ #
        all_lo = df.loc[~df["Reference"], "CI_Lower"]
        all_hi = df.loc[~df["Reference"], "CI_Upper"]

        x_min = max(0.3, all_lo.min() * 0.85)
        x_max = all_hi.max() * 1.15

        ax_plot.set_xlim(x_min, x_max)
        ax_plot.set_ylim(-0.5, y_max)
        #ax_plot.set_xscale("log")
        ax_plot.axvline(x=1.0, color="black", linewidth=0.9,
                        linestyle="--", alpha=0.7)

        # Column header
        ax_plot.text(
            0.5, 1.0, "Relative Risk (log scale)",
            fontweight="bold", fontsize=9, va="bottom", ha="center",
            transform=ax_plot.transAxes
        )

        for y_pos, row in zip(y_positions, rows):
            if row is None:
                continue

            is_ref = row["Reference"]
            color = color_map[row["Variable"]]

            if is_ref:
                ax_plot.plot(1.0, y_pos, marker="D", color=color,
                             markersize=6, alpha=0.5, zorder=3)
            else:
                ax_plot.plot(
                    [row["CI_Lower"], row["CI_Upper"]], [y_pos, y_pos],
                    color=color, linewidth=1.5, zorder=2
                )
                for x_cap in [row["CI_Lower"], row["CI_Upper"]]:
                    ax_plot.plot(x_cap, y_pos, marker="|", color=color,
                                 markersize=6, markeredgewidth=1.5, zorder=3)
                ax_plot.plot(row["RR"], y_pos, marker="s", color=color,
                             markersize=7, zorder=4)

        # Alternating group shading
        for i, group in enumerate(groups):
            group_y = [
                y_pos for y_pos, r in zip(y_positions, rows)
                if r is not None and r["Variable"] == group
            ]
            if group_y and i % 2 == 0:
                ax_plot.axhspan(
                    min(group_y) - 0.5, max(group_y) + 0.5,
                    color="lightgray", alpha=0.2, zorder=0
                )

        ax_plot.set_xlabel("Relative Risk", fontsize=9)
        ax_plot.tick_params(axis="y", left=False, labelleft=False)
        ax_plot.tick_params(axis="x", labelsize=8)
        ax_plot.spines[["top", "right", "left"]].set_visible(False)

        # Legend
        legend_handles = [
            mpatches.Patch(color=color_map[g], label=g) for g in groups
        ]
        ax_plot.legend(handles=legend_handles, loc="upper right",
                       fontsize=8, framealpha=0.7)

        #fig.suptitle(title, fontsize=12, fontweight="bold")

        return fig, (ax_plot, ax_table)


    _fig, (_ax_plot, _ax_table) = plot_forest(df, title="Forest Plot of Relative Risk")
    _fig
    return mpatches, plot_forest


@app.cell
def _(df, np, pd):
    def add_ai_use(df, rr, ci_lower, ci_upper):
        """
        Append an 'AI Used' variable to an existing GLM results DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Existing results DataFrame compatible with plot_forest().
        rr : float
            Point estimate (relative risk) for AI Used = Yes.
        ci_lower : float
            Lower bound of the 95% CI for AI Used = Yes.
        ci_upper : float
            Upper bound of the 95% CI for AI Used = Yes.

        Returns
        -------
        pd.DataFrame
            Original DataFrame with two new rows appended for 'AI Used'.
        """
        new_rows = pd.DataFrame([
            {
                "Variable":  "AI Used",
                "Category":  "No",
                "RR":        1.00,
                "CI_Lower":  np.nan,
                "CI_Upper":  np.nan,
                "Reference": True,
                "95% CI":    "Ref",
            },
            {
                "Variable":  "AI Used",
                "Category":  "Yes",
                "RR":        round(rr, 2),
                "CI_Lower":  round(ci_lower, 2),
                "CI_Upper":  round(ci_upper, 2),
                "Reference": False,
                "95% CI":    f"({ci_lower:.2f}, {ci_upper:.2f})",
            },
        ])

        return pd.concat([df, new_rows], ignore_index=True)

    _sig_ci = [1.01, 15.0]
    sig_df = add_ai_use(df, np.mean(_sig_ci), _sig_ci[0], _sig_ci[1])

    _prec_ci = [1.15, 1.25]
    prec_df = add_ai_use(df, np.mean(_prec_ci), _prec_ci[0], _prec_ci[1])
    return prec_df, sig_df


@app.cell
def _(plot_forest, sig_df):
    _fig, (_ax_plot, _ax_table) = plot_forest(sig_df, title="Fake GLM Results\nGoal is Statistical Significance\nn = only as many observations as required to acheive significance")
    _fig
    return


@app.cell
def _(plot_forest, prec_df):
    _fig, (_ax_plot, _ax_table) = plot_forest(prec_df, title="Fake GLM Results\nGoal is Precision\nn = as many observations as we could get our hands on")
    _fig
    return


@app.cell
def _(mpatches):
    def plot_forest_ax(df, ax, title="Forest Plot of Relative Risk"):
        """
        Draw a forest plot onto a provided Axes object, with no table panel.
        The AI Used: Yes row is annotated with its RR and 95% CI directly on the plot.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: 'Variable', 'Category', 'RR', 'CI_Lower', 'CI_Upper', 'Reference'
        ax : matplotlib.axes.Axes
            Axes to draw on.
        title : str
            Title for the plot.
        """

        groups = df["Variable"].unique()
        palette = ["#2166ac", "#d6604d", "#4dac26"]
        color_map = {group: palette[i % len(palette)] for i, group in enumerate(groups)}

        # Build rows with separator Nones between groups
        rows = []
        prev_var = None
        for _, row in df.iterrows():
            if prev_var is not None and row["Variable"] != prev_var:
                rows.append(None)
            rows.append(row)
            prev_var = row["Variable"]

        # Compute y positions
        y_positions = []
        y = 0
        for row in reversed(rows):
            y_positions.insert(0, y)
            y += 0.4 if row is None else 1.0
        y_max = y
    

        # X axis limits
        all_lo = df.loc[~df["Reference"], "CI_Lower"]
        all_hi = df.loc[~df["Reference"], "CI_Upper"]
        x_min = max(0.3, all_lo.min() * 0.85)
        x_max = all_hi.max() * 1.15

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.5, y_max)
        #ax.set_xscale("log")
        ax.axvline(x=1.0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
        ax.set_title(title, fontsize=11, fontweight="bold")

        for y_pos, row in zip(y_positions, rows):
            if row is None:
                continue

            is_ref = row["Reference"]
            color = color_map[row["Variable"]]

            if is_ref:
                ax.plot(1.0, y_pos, marker="D", color=color,
                        markersize=6, alpha=0.5, zorder=3)
            else:
                ax.plot(
                    [row["CI_Lower"], row["CI_Upper"]], [y_pos, y_pos],
                    color=color, linewidth=1.5, zorder=2
                )
                for x_cap in [row["CI_Lower"], row["CI_Upper"]]:
                    ax.plot(x_cap, y_pos, marker="|", color=color,
                            markersize=6, markeredgewidth=1.5, zorder=3)
                ax.plot(row["RR"], y_pos, marker="s", color=color,
                        markersize=7, zorder=4)

                # Annotate only the AI Used: Yes row
                if row["Variable"] == "AI Used" and row["Category"] == "Yes":
                    ax.annotate(
                        f"RR {row['RR']:.2f} ({row['CI_Lower']:.2f}, {row['CI_Upper']:.2f})",
                        xy=(row["CI_Upper"]*1.01, y_pos*1.05),
                        xytext=(8, 0),
                        textcoords="offset points",
                        fontsize=12,
                        va="center",
                        color=color,
                    )

        # Alternating group shading
        for i, group in enumerate(groups):
            group_y = [
                y_pos for y_pos, r in zip(y_positions, rows)
                if r is not None and r["Variable"] == group
            ]
            if group_y and i % 2 == 0:
                ax.axhspan(
                    min(group_y) - 0.5, max(group_y) + 0.5,
                    color="lightgray", alpha=0.2, zorder=0
                )

        ax.set_xlabel("Relative Risk", fontsize=9)
        # Set y ticks at data row positions with category labels
        data_y = [(y_pos, row["Category"]) for y_pos, row in zip(y_positions, rows) if row is not None]
        ax.set_yticks([y for y, _ in data_y])
        ax.set_yticklabels([label for _, label in data_y], fontsize=8.5)
        ax.tick_params(axis="y", left=False)  # hide tick marks but keep labels
        #ax.tick_params(axis="y", left=False, labelleft=False)
        ax.tick_params(axis="x", labelsize=8)
        ax.spines[["top", "right", "left"]].set_visible(False)

        legend_handles = [
            mpatches.Patch(color=color_map[g], label=g) for g in groups
        ]
        ax.legend(handles=legend_handles, loc="upper right",
                  fontsize=8, framealpha=0.7)

        return ax

    return (plot_forest_ax,)


@app.cell
def _(plot_forest_ax, plt, prec_df, sig_df):
    _fig, _axes = plt.subplots(1, 2, figsize=(16, 5))

    plot_forest_ax(sig_df, _axes[0], title="Goal is Significance\nn = only as many observations as required for significance")
    plot_forest_ax(prec_df, _axes[1], title="Goal is Precision\nn = as many observations as we could get our hands on")

    _axes[0].text(
        s='"We see a statistically significant impact of AI.\nThe magnitude is somewhere between a 1% and a 1,500% increase.\nWe frankly do not really care about how big the impact is,\nwe just care that we acheived significance."',
        x=-0.0,
        y=-0.2,
        ha="left",
        va="top",
        transform=_axes[0].transAxes,
        fontsize=12
    )
    _axes[0].annotate(
        "Statistically Significant,\nbut high uncertainty",
        (1.01, 0.1),
        xytext=(8, 3),
        arrowprops={
            "width": 1
        },
        va="bottom",
        ha="center"
    )
    _axes[0].annotate(
        "",
        (15, 0.1),
        xytext=(11, 3),
        arrowprops={
            "width": 1
        }
    )

    _axes[1].text(
        s='"We see a statistically significant impact of AI.\nBut we do not care about significance -- precision is more relevant.\n As a resut, we included as many observations as possible,\nand found that the magnitude is somewhere between a 15% and a 25% increase."',
        x=-0.0,
        y=-0.2,
        ha="left",
        va="top",
        transform=_axes[1].transAxes,
        fontsize=12
    )
    _axes[1].annotate(
        "Statistically Significant,\nand high precision",
        (2, 0.0),
        xytext=(2.2, 0),
        arrowprops={
            "width": 1
        },
        va="center",
        ha="left"
    )

    plt.tight_layout(h_pad=0.1, w_pad=6)

    _fig
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
