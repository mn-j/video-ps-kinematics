"""PCA-varimax feature structure analysis."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_pca_varimax_analysis(
    merged_df,
    feature_cols=None,
    output_dir=".",
    save_plots=True,
    show_plots=False,
    max_components=12,
    min_eigenvalue=1.0,
):
    """Principal component analysis with varimax rotation on kinematic features.

    Replicates Zarrat Ehsan et al. (2024) Figure 4:
    * Scree plot: individual + cumulative explained variance.
    * Varimax-rotated loading heatmap.

    Parameters
    ----------
    merged_df : pd.DataFrame
        DataFrame containing kinematic features and 'score_clean'.
    feature_cols : list of str, optional
        Feature columns to include. Defaults to all numeric kinematic columns
        that contain at least 10 non-NaN values.
    output_dir : str
        Directory for saving plots.
    save_plots : bool
        Whether to save figures to disk.
    show_plots : bool
        Whether to display figures interactively.
    max_components : int
        Maximum number of PCA components to compute (<=n_features).
    min_eigenvalue : float
        Kaiser criterion: retain components with eigenvalue >= this (default 1.0).

    Returns
    -------
    dict with keys 'n_components', 'explained_variance_ratio', 'loadings_df', 'feature_cols'
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Select feature columns
    if feature_cols is None:
        _exclude = {"score_clean", "record_type", "video_path", "output_video"}
        feature_cols = [
            c
            for c in merged_df.select_dtypes(include="number").columns
            if c not in _exclude and merged_df[c].notna().sum() >= 10
        ]

    # Drop rows with any NaN in selected features
    _df = merged_df[feature_cols].dropna()
    if len(_df) < 5 or len(feature_cols) < 2:
        print("PCA-varimax: insufficient data \u2014 skipping.")
        return {}

    print(f"\n{'='*60}")
    print("PCA-VARIMAX FEATURE STRUCTURE ANALYSIS")
    print(f"{'='*60}")
    print(f"  Features: {len(feature_cols)}, Observations: {len(_df)}")

    # Standardise
    _scaler = StandardScaler()
    _X = _scaler.fit_transform(_df.values)

    # PCA
    _n_comp = min(max_components, len(feature_cols), len(_df) - 1)
    _pca = PCA(n_components=_n_comp, random_state=42)
    _pca.fit(_X)

    # Kaiser criterion: retain components with eigenvalue >= min_eigenvalue
    _eigenvalues = _pca.explained_variance_
    _n_retain = max(1, int(np.sum(_eigenvalues >= min_eigenvalue)))
    _n_retain = min(_n_retain, _n_comp)

    _ev_ratio = _pca.explained_variance_ratio_
    _cumvar = np.cumsum(_ev_ratio) * 100.0
    print(f"  Components retained (eigenvalue \u2265 {min_eigenvalue:.1f}): {_n_retain}")
    print(f"  Cumulative variance explained: {_cumvar[_n_retain-1]:.1f}%")

    # --- Scree plot ---
    fig_scree, ax_scree = plt.subplots(figsize=(8, 5))
    _xvals = np.arange(1, _n_comp + 1)
    ax_scree.bar(
        _xvals, _ev_ratio * 100.0, color="steelblue", alpha=0.7, label="Individual variance (%)"
    )
    ax_scree_r = ax_scree.twinx()
    ax_scree_r.plot(_xvals, _cumvar, "k-o", markersize=5, label="Cumulative variance (%)")
    for _ci, _cv in enumerate(_cumvar):
        if (_ci + 1) <= _n_retain:
            ax_scree_r.annotate(
                f"{_cv:.1f}%",
                xy=(_ci + 1, _cv),
                xytext=(2, 4),
                textcoords="offset points",
                fontsize=7,
                color="black",
            )
    ax_scree.axvline(
        _n_retain + 0.5,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"Retain {_n_retain} (Kaiser)",
    )
    ax_scree.set_xlabel("Principal Component")
    ax_scree.set_ylabel("Explained Variance (%)")
    ax_scree_r.set_ylabel("Cumulative Variance (%)")
    ax_scree_r.set_ylim(0, 105)
    ax_scree.set_title("PCA Scree Plot \u2014 Kinematic Features")
    lines1, labs1 = ax_scree.get_legend_handles_labels()
    lines2, labs2 = ax_scree_r.get_legend_handles_labels()
    ax_scree.legend(lines1 + lines2, labs1 + labs2, loc="center right", fontsize=9)
    plt.tight_layout()
    if save_plots:
        _sp = os.path.join(output_dir, "pca_varimax_scree.png")
        fig_scree.savefig(_sp, dpi=150, bbox_inches="tight")
        print(f"  Saved scree plot \u2192 {_sp}")
    if show_plots:
        plt.show()
    else:
        plt.close(fig_scree)

    # --- Varimax rotation ---
    _loadings_raw = _pca.components_[:_n_retain].T  # shape: (n_features, n_retain)

    def _varimax(Phi, gamma=1.0, q=20, tol=1e-6):
        """Varimax rotation (Kaiser 1958)."""
        p, k = Phi.shape
        R = np.eye(k)
        d_old = 0.0
        for _ in range(q):
            Lambda = Phi @ R
            u, s, vh = np.linalg.svd(
                Phi.T @ (Lambda**3 - (gamma / p) * Lambda @ np.diag(np.diag(Lambda.T @ Lambda)))
            )
            R = u @ vh
            d_new = np.sum(s)
            if abs(d_new - d_old) < tol:
                break
            d_old = d_new
        return Phi @ R

    try:
        from factor_analyzer.rotator import Rotator

        _rot = Rotator(method="varimax")
        _loadings_rot = _rot.fit_transform(_loadings_raw)
        print("  Varimax rotation: using factor_analyzer library.")
    except ImportError:
        _loadings_rot = _varimax(_loadings_raw)
        print(
            "  Varimax rotation: using built-in implementation (install factor-analyzer for faster convergence)."
        )

    _comp_labels = [f"PC{i+1}" for i in range(_n_retain)]
    _loadings_df = pd.DataFrame(_loadings_rot, index=feature_cols, columns=_comp_labels)

    # --- Loading heatmap ---
    fig_heat, ax_heat = plt.subplots(
        figsize=(max(6, _n_retain * 1.0), max(8, len(feature_cols) * 0.4))
    )

    _vmax = 1.0
    _cmap = "RdBu_r"
    _im = ax_heat.imshow(_loadings_df.values, cmap=_cmap, vmin=-_vmax, vmax=_vmax, aspect="auto")
    plt.colorbar(_im, ax=ax_heat, fraction=0.03, pad=0.04)
    ax_heat.set_xticks(range(_n_retain))
    ax_heat.set_xticklabels(_comp_labels, fontsize=9)
    ax_heat.set_yticks(range(len(feature_cols)))
    ax_heat.set_yticklabels(feature_cols, fontsize=8)
    # Annotate cell values
    for _ri in range(len(feature_cols)):
        for _ci in range(_n_retain):
            _v = _loadings_df.iloc[_ri, _ci]
            _tc = "white" if abs(_v) > 0.6 else "black"
            ax_heat.text(_ci, _ri, f"{_v:.2f}", ha="center", va="center", fontsize=7, color=_tc)
    ax_heat.set_title(
        f"Varimax-Rotated PCA Loadings ({_n_retain} components, "
        f"{_cumvar[_n_retain-1]:.1f}% variance)"
    )
    plt.tight_layout()
    if save_plots:
        _hp = os.path.join(output_dir, "pca_varimax_loadings.png")
        fig_heat.savefig(_hp, dpi=150, bbox_inches="tight")
        print(f"  Saved loadings heatmap \u2192 {_hp}")
    if show_plots:
        plt.show()
    else:
        plt.close(fig_heat)

    # Print top loadings per component
    for _ci, _cp in enumerate(_comp_labels):
        _col_load = _loadings_df[_cp].abs().sort_values(ascending=False)
        _top = _col_load.head(4)
        print(
            f"  {_cp}: " + ", ".join(f"{f} ({_loadings_df.loc[f, _cp]:+.2f})" for f in _top.index)
        )

    return {
        "n_components": _n_retain,
        "explained_variance_ratio": _ev_ratio[:_n_retain].tolist(),
        "cumulative_variance_pct": float(_cumvar[_n_retain - 1]),
        "loadings_df": _loadings_df,
        "feature_cols": feature_cols,
    }
