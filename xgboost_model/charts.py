"""Chart generation for the XGBoost replacement models.

Colors are taken directly from the validated reference palette (categorical
hues, chrome/ink roles) rather than matplotlib defaults -- fixed hue order,
one hue per magnitude series, no dual axes, recessive gridlines, legend
whenever more than one series is on a chart.
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

from evaluate_fixed import evaluate_fixed_model

CHARTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")

# Reference palette (light mode) -- categorical slots, in fixed order, plus chrome/ink.
COLOR_SURFACE = "#fcfcfb"
COLOR_INK_PRIMARY = "#0b0b0b"
COLOR_INK_SECONDARY = "#52514e"
COLOR_INK_MUTED = "#898781"
COLOR_GRIDLINE = "#e1e0d9"
COLOR_BASELINE = "#c3c2b7"

CATEGORICAL = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
}

# Color follows the entity (engine), never its rank -- fixed across every
# chart, in the palette's documented slot order (1=blue, 2=orange, 3=aqua,
# 4=yellow), so a 4th engine never repaints the first three.
ENGINE_COLORS = {
    "autogluon": CATEGORICAL["blue"],
    "xgboost_naive_quantile": CATEGORICAL["orange"],
    "xgboost_conformal": CATEGORICAL["aqua"],
    "xgboost_conformal_tuned": CATEGORICAL["yellow"],
}
ENGINE_LABELS = {
    "autogluon": "AutoGluon (production)",
    "xgboost_naive_quantile": "XGBoost (naive quantile)",
    "xgboost_conformal": "XGBoost (conformal, baseline features)",
    "xgboost_conformal_tuned": "XGBoost (conformal, tuned+enriched)",
}


def _new_axis(figsize=(9, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_BASELINE)
    ax.tick_params(colors=COLOR_INK_MUTED, labelsize=9)
    ax.grid(axis="y", color=COLOR_GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    return fig, ax


def plot_forecast_band(site_code, horizon_hours, days=45, save_path=None):
    """Actual vs. predicted gage height over the held-out test period, with
    the conformal-calibrated 80% interval shaded behind the forecast line."""
    result = evaluate_fixed_model(site_code, horizon_hours, days=days, return_predictions=True)
    if result["status"] != "ok":
        raise RuntimeError(f"Cannot chart {site_code} h={horizon_hours}: {result['status']}")

    preds = result["predictions"]

    fig, ax = _new_axis(figsize=(10, 5))

    ax.fill_between(
        preds["timestamp"], preds["lower"], preds["upper"],
        color=CATEGORICAL["blue"], alpha=0.15, linewidth=0, zorder=1,
        label=f"{result['ci80_nominal']:.0%} interval (conformal)",
    )
    ax.plot(
        preds["timestamp"], preds["actual"],
        color=COLOR_INK_PRIMARY, linewidth=1.6, zorder=3, label="Actual gage height",
    )
    ax.plot(
        preds["timestamp"], preds["predicted"],
        color=CATEGORICAL["blue"], linewidth=1.6, zorder=2, label="Predicted (XGBoost)",
    )

    ax.set_ylabel("Gage height (ft)", color=COLOR_INK_SECONDARY, fontsize=10)
    ax.set_title(
        f"Pompton River (site {site_code}) — {horizon_hours}h-ahead forecast, held-out test period\n"
        f"MAE {result['mae_ft']:.3f} ft · NSE {result['nse']:.2f} · "
        f"CI coverage {result['ci80_coverage']:.0%} (nominal {result['ci80_nominal']:.0%})",
        color=COLOR_INK_PRIMARY, fontsize=11, loc="left",
    )
    fig.autofmt_xdate()
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=COLOR_INK_SECONDARY)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=COLOR_SURFACE)
        plt.close(fig)
        return save_path
    return fig


def plot_forecast_lead_time(site_code, horizon_hours, days=45, save_path=None):
    """Same data as plot_forecast_band, aligned differently to answer a
    different question: does the forecast actually give advance warning?

    plot_forecast_band plots the prediction at its *target* time (issue_time
    + horizon) right next to the actual reading for that same moment -- the
    standard way to visualize point-forecast accuracy, but it can *never*
    show lead time, because a perfectly accurate forecast and a merely
    persistent one look identical under that alignment (both just match the
    actual line at each moment).

    Here, the actual line stays at its true, natural occurrence times, but
    the predicted line and its interval are shifted back to the *issue*
    time (when the forecast was actually made) instead of the target time.
    If the model had real lead time, its predicted line would show a rise
    horizon_hours to the *left* of the actual line's own rise. If the model
    is reactive (only extrapolating a rise that's already visible in its own
    recent inputs -- see README.md), the predicted line's rise lands at
    essentially the same moment as the actual line's, because "predicted for
    T+horizon, issued at T" ends up close to whatever the reading already
    was at T.
    """
    result = evaluate_fixed_model(site_code, horizon_hours, days=days, return_predictions=True)
    if result["status"] != "ok":
        raise RuntimeError(f"Cannot chart {site_code} h={horizon_hours}: {result['status']}")

    preds = result["predictions"]
    issue_time = preds["timestamp"] - pd.Timedelta(hours=horizon_hours)

    fig, ax = _new_axis(figsize=(10, 5))

    ax.fill_between(
        issue_time, preds["lower"], preds["upper"],
        color=CATEGORICAL["blue"], alpha=0.15, linewidth=0, zorder=1,
        label=f"{result['ci80_nominal']:.0%} interval, shown at issue time",
    )
    ax.plot(
        preds["timestamp"], preds["actual"],
        color=COLOR_INK_PRIMARY, linewidth=1.6, zorder=3, label="Actual gage height (true time)",
    )
    ax.plot(
        issue_time, preds["predicted"],
        color=CATEGORICAL["orange"], linewidth=1.6, zorder=2,
        label=f"Predicted, shown at issue time (i.e. {horizon_hours}h before its target)",
    )

    ax.set_ylabel("Gage height (ft)", color=COLOR_INK_SECONDARY, fontsize=10)
    ax.set_title(
        f"Pompton River (site {site_code}) — {horizon_hours}h-ahead forecast, aligned by issue time\n"
        f"a real early-warning signal would show the orange line rising {horizon_hours}h left of the black line",
        color=COLOR_INK_PRIMARY, fontsize=11, loc="left",
    )
    fig.autofmt_xdate()
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=COLOR_INK_SECONDARY)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=COLOR_SURFACE)
        plt.close(fig)
        return save_path
    return fig


def plot_calibration_reliability(results_df, save_path=None):
    """Grouped bars: empirical CI coverage per engine per horizon, against a
    dashed nominal-coverage reference line. The gap between bar and line *is*
    the finding -- bars below the line under-cover (dangerous), above it
    over-cover (conservative but safe)."""
    engines = [e for e in ENGINE_COLORS if e in results_df["engine"].unique()]
    horizons = sorted(results_df["horizon_hours"].unique())

    fig, ax = _new_axis(figsize=(10, 5))

    bar_width = 0.8 / len(engines)
    x = range(len(horizons))

    for i, engine in enumerate(engines):
        sub = results_df[results_df["engine"] == engine].set_index("horizon_hours")
        # NaN (not a missing list entry) for any horizon this engine has no
        # row for -- matplotlib just skips drawing that bar, but the x
        # position stays reserved so every engine's bars line up across
        # horizons instead of compacting toward missing ones.
        values = [sub.loc[h, "ci80_coverage"] if h in sub.index else float("nan") for h in horizons]
        # Center the group of engines on each horizon's tick: engine i sits
        # at an offset of (i - middle_index) bar-widths from that tick.
        offsets = [xi + (i - (len(engines) - 1) / 2) * bar_width for xi in x]
        ax.bar(
            offsets, values, width=bar_width * 0.9,
            color=ENGINE_COLORS[engine], label=ENGINE_LABELS[engine], zorder=2,
        )

    nominal = results_df["ci80_nominal"].iloc[0]
    ax.axhline(nominal, color=COLOR_INK_SECONDARY, linewidth=1.4, linestyle="--", zorder=3)
    ax.text(
        len(horizons) - 0.5, nominal + 0.02, f"nominal {nominal:.0%}",
        color=COLOR_INK_SECONDARY, fontsize=9, ha="right",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{h}h" for h in horizons])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Empirical CI coverage", color=COLOR_INK_SECONDARY, fontsize=10)
    ax.set_title(
        "Confidence interval calibration by horizon\nbars above the dashed line are conservative (safe); below it, dangerously overconfident",
        color=COLOR_INK_PRIMARY, fontsize=11, loc="left",
    )
    ax.legend(frameon=False, loc="lower left", fontsize=9, labelcolor=COLOR_INK_SECONDARY, ncol=1)
    fig.text(
        0.01, 0.01,
        "Note: AutoGluon/naive-quantile use a 14-day walk-forward window; conformal uses a separate 45-day train/calibrate/test split. Directionally comparable, not the same test rows.",
        color=COLOR_INK_MUTED, fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=COLOR_SURFACE)
        plt.close(fig)
        return save_path
    return fig


def plot_error_by_horizon(results_df, save_path=None):
    """Grouped bars: MAE per engine per horizon -- the point-forecast accuracy
    comparison, same entity-color mapping as the calibration chart."""
    engines = [e for e in ENGINE_COLORS if e in results_df["engine"].unique()]
    horizons = sorted(results_df["horizon_hours"].unique())

    fig, ax = _new_axis(figsize=(10, 5))

    bar_width = 0.8 / len(engines)
    x = range(len(horizons))

    for i, engine in enumerate(engines):
        # Same NaN-for-missing / centered-offset logic as
        # plot_calibration_reliability above.
        sub = results_df[results_df["engine"] == engine].set_index("horizon_hours")
        values = [sub.loc[h, "mae_ft"] if h in sub.index else float("nan") for h in horizons]
        offsets = [xi + (i - (len(engines) - 1) / 2) * bar_width for xi in x]
        ax.bar(
            offsets, values, width=bar_width * 0.9,
            color=ENGINE_COLORS[engine], label=ENGINE_LABELS[engine], zorder=2,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{h}h" for h in horizons])
    ax.set_ylabel("MAE (ft)", color=COLOR_INK_SECONDARY, fontsize=10)
    ax.set_title(
        "Point-forecast error by horizon", color=COLOR_INK_PRIMARY, fontsize=11, loc="left",
    )
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=COLOR_INK_SECONDARY)
    fig.text(
        0.01, 0.01,
        "Note: AutoGluon/naive-quantile use a 14-day walk-forward window; conformal uses a separate 45-day train/calibrate/test split. Directionally comparable, not the same test rows.",
        color=COLOR_INK_MUTED, fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=COLOR_SURFACE)
        plt.close(fig)
        return save_path
    return fig


if __name__ == "__main__":
    os.makedirs(CHARTS_DIR, exist_ok=True)
    site_code = "01388500"

    # <=6h only: NSE goes negative (worse than predicting the mean) at 12h+
    # for every engine tried, regardless of feature set or tuning -- see
    # README.md. Not worth charting a horizon nothing can forecast well.
    HORIZONS = [1, 3, 6]

    print("Rendering forecast-with-band charts...")
    for horizon in HORIZONS:
        path = plot_forecast_band(site_code, horizon, save_path=os.path.join(CHARTS_DIR, f"forecast_h{horizon}.png"))
        print(f"  saved {path}")

    print("Rendering issue-time-aligned lead-time charts...")
    for horizon in HORIZONS:
        path = plot_forecast_lead_time(site_code, horizon, save_path=os.path.join(CHARTS_DIR, f"lead_time_h{horizon}.png"))
        print(f"  saved {path}")

    print("\nAssembling cross-engine comparison table...")
    autogluon_df = pd.read_csv(os.path.join("..", "evaluation", "backtest_results.csv"))
    autogluon_df = autogluon_df[autogluon_df["label"].str.startswith("pompton_h")].copy()
    autogluon_df["engine"] = "autogluon"

    naive_xgb_df = pd.read_csv(os.path.join("..", "evaluation", "xgboost_vs_autogluon_pompton.csv"))
    naive_xgb_df = naive_xgb_df[naive_xgb_df["engine"] == "xgboost"].copy()
    naive_xgb_df["engine"] = "xgboost_naive_quantile"

    # Baseline features (own-site lags only) with conformal calibration --
    # the main deliverable before the upstream+weather experiment.
    conformal_results = [evaluate_fixed_model(site_code, h, days=45) for h in HORIZONS]
    conformal_df = pd.DataFrame([r for r in conformal_results if r["status"] == "ok"])
    conformal_df["engine"] = "xgboost_conformal"

    # Enriched features (upstream gages + weather) *with* regularization/
    # feature-count tuning -- the untuned enriched variant is deliberately
    # left out of this chart since tuning.py's search strictly improves on
    # it (see README's tuning section); showing both would just clutter the
    # comparison without changing the conclusion.
    tuned_results = [evaluate_fixed_model(site_code, h, days=45, feature_set="enriched", tune=True) for h in HORIZONS]
    tuned_df = pd.DataFrame([r for r in tuned_results if r["status"] == "ok"])
    tuned_df["engine"] = "xgboost_conformal_tuned"

    combined = pd.concat([
        autogluon_df[["engine", "horizon_hours", "mae_ft", "ci80_coverage", "ci80_nominal"]],
        naive_xgb_df[["engine", "horizon_hours", "mae_ft", "ci80_coverage", "ci80_nominal"]],
        conformal_df[["engine", "horizon_hours", "mae_ft", "ci80_coverage", "ci80_nominal"]],
        tuned_df[["engine", "horizon_hours", "mae_ft", "ci80_coverage", "ci80_nominal"]],
    ], ignore_index=True)
    # Drop rows where the model failed to load/predict (e.g. AutoGluon's
    # broken h48) -- they carry NaN horizon_hours, which otherwise sorts into
    # its own bogus "nanh" tick on the x-axis. Also restrict AutoGluon/naive-
    # quantile (which still carry their own 12h+ rows from earlier runs) down
    # to the same <=6h focus as everything else.
    combined = combined.dropna(subset=["horizon_hours"])
    combined["horizon_hours"] = combined["horizon_hours"].astype(int)
    combined = combined[combined["horizon_hours"].isin(HORIZONS)]
    combined.to_csv(os.path.join(CHARTS_DIR, "combined_comparison.csv"), index=False)

    print("Rendering comparison charts...")
    path = plot_calibration_reliability(combined, save_path=os.path.join(CHARTS_DIR, "calibration_reliability.png"))
    print(f"  saved {path}")
    path = plot_error_by_horizon(combined, save_path=os.path.join(CHARTS_DIR, "error_by_horizon.png"))
    print(f"  saved {path}")
