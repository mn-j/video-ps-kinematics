"""
scripts/classify_nn.py — Neural-network PD motor severity classification.

Architectures
-------------
ResidualMLP
    Feed-forward network with residual blocks, batch normalisation and dropout.
    Trained with class-weighted softmax cross-entropy.  Serves as a strong
    tabular-NN baseline.

CoralMLP
    Same backbone, but with a CORAL ordinal-regression head.

    Reference: Cao et al. 2020, "Rank Consistent Ordinal Regression for Neural
    Networks with Application to Age Estimation."  ArXiv:1901.07884.

    CORAL replaces the K-way softmax head with K−1 binary classifiers that
    share a weight vector but have independent bias terms.  The loss is a sum
    of binary cross-entropies over the K−1 rank thresholds.  This directly
    encodes Mild < Moderate < Severe — a constraint that vanilla cross-entropy
    completely ignores and that tree-based models cannot express natively.

Evaluation
----------
  Leave-one-subject-out CV — identical protocol to classify.py.
  Metrics: accuracy, balanced accuracy, macro precision, macro F1.
  Gradient saliency (mean |∂loss/∂input|) replaces gain-based feature
  importance for the final full-dataset model.

Usage
-----
  python scripts/classify_nn.py \\
      --kinematics tracking_logs.csv \\
      --scores scores.csv \\
      --id2vid id2vid.csv \\
      --output results/classify_nn/

Dependencies
------------
  torch        (pip install torch)
  scikit-learn (pip install scikit-learn)
  All other deps inherited from classify.py / ps_kinematics.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make sibling scripts importable (e.g. `from classify import _load_and_merge`)
# regardless of how this file is invoked. The ps_kinematics package itself
# is resolved via the editable install (see `pip install -e .`).
_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

# ── Shared helpers from classify.py ──────────────────────────────────────────
from classify import (  # noqa: E402
    CLASS_NAMES,
    CLASSIFICATION_FEATURES,
    _compute_metrics,
    _load_and_merge,
    _plot_confusion_matrix,
)

# ── PyTorch (optional — hard failure if absent at runtime) ───────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset  # noqa: F401

    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

# ── Progress bars (tqdm — degrades gracefully if not installed) ───────────────
try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:

    def _tqdm(iterable=None, **kwargs):  # type: ignore[misc]
        """No-op fallback when tqdm is not installed."""
        return iterable if iterable is not None else range(kwargs.get("total", 0))


# ─────────────────────────────────────────────────────────────────────────────
# Default hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
_HIDDEN = 128  # hidden-layer width (wider for more feature interaction capacity)
_N_BLOCKS = 2  # number of residual blocks
_DROPOUT = 0.4  # dropout probability
_EPOCHS = 100  # maximum training epochs per fold
_LR = 1e-3  # AdamW initial learning rate
_WEIGHT_DECAY = 1e-4  # L2 regularisation
_BATCH_SIZE = 256  # mini-batch size (larger = fewer kernel launches = better GPU utilisation)
_PATIENCE = 15  # early-stopping patience (epochs without val improvement)
_VAL_FRAC = 0.15  # fraction of train-val pool held out for early stopping
_N_SEEDS = 5  # ensemble seeds per LOSO fold — train N models, average probs → lower variance
_INPUT_NOISE_STD = (
    0.05  # σ of Gaussian noise added to scaled inputs during training (regularisation)
)
_LABEL_SMOOTHING = 0.1  # label-smoothing ε for CE loss (prevents overconfident predictions)


# ─────────────────────────────────────────────────────────────────────────────
# Neural-network architectures
# ─────────────────────────────────────────────────────────────────────────────


class _ResidualBlock(nn.Module):
    """Two-layer residual block: Linear → BN → ReLU → Dropout → Linear → BN,
    with a skip connection and a final ReLU."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.ReLU()

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.act(x + self.net(x))


class ResidualMLP(nn.Module):
    """Tabular feed-forward network with residual blocks.

    Output: ``(batch, num_classes)`` raw logits.
    Use ``F.softmax`` or ``F.cross_entropy`` on these.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 3,
        hidden: int = _HIDDEN,
        n_blocks: int = _N_BLOCKS,
        dropout: float = _DROPOUT,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[_ResidualBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.head(self.blocks(self.stem(x)))


class CoralMLP(nn.Module):
    """Ordinal regression via CORAL (Cao et al. 2020).

    Architecture
    ------------
    Same stem + residual blocks as :class:`ResidualMLP`, but the output head
    uses K−1 shared-weight binary classifiers with independent bias terms::

        logit_k(x) = w · h(x) + b_k      for k = 0 … K−2

    where ``h(x)`` is the hidden representation.  The K class probabilities
    are recovered from cumulative sigmoid probabilities:

        P(Y ≥ k)  = σ(logit_{k-1}(x))
        P(Y = k)  = P(Y ≥ k) − P(Y ≥ k+1)

    Output of ``forward()``: ``(batch, K−1)`` threshold logits — use
    :meth:`predict_proba` for class probabilities.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 3,
        hidden: int = _HIDDEN,
        n_blocks: int = _N_BLOCKS,
        dropout: float = _DROPOUT,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.stem = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[_ResidualBlock(hidden, dropout) for _ in range(n_blocks)])
        self.fc = nn.Linear(hidden, 1, bias=False)  # shared weight
        # K−1 independent bias terms, initialised in strictly descending order
        # so that P(Y>=1) > P(Y>=2) holds from the very first forward pass.
        # Zero-initialisation would give P(Y>=1) = P(Y>=2) = 0.5 at init,
        # immediately violating rank monotonicity and making P(Moderate)=0.
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))
        with torch.no_grad():
            self.bias.copy_(torch.linspace(1.0, -1.0, num_classes - 1))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Return ``(batch, K−1)`` threshold logits."""
        h = self.blocks(self.stem(x))
        return self.fc(h) + self.bias  # (batch,1) + (K-1,) → (batch, K-1)

    def predict_proba(self, x: "torch.Tensor") -> "torch.Tensor":
        """Return ``(batch, K)`` class probabilities (no grad)."""
        with torch.no_grad():
            logits = self.forward(x)
        cumprobs = torch.sigmoid(logits)  # P(Y >= k+1), k=0..K-2
        b = x.shape[0]
        ones = torch.ones(b, 1, device=x.device)
        zeros = torch.zeros(b, 1, device=x.device)
        aug = torch.cat([ones, cumprobs, zeros], dim=1)  # (batch, K+1)
        probs = (aug[:, :-1] - aug[:, 1:]).clamp(min=0.0)  # (batch, K)
        return probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────


def _ce_loss(
    logits: "torch.Tensor",
    targets: "torch.Tensor",
    class_weights: "torch.Tensor",
    label_smoothing: float = _LABEL_SMOOTHING,
) -> "torch.Tensor":
    """Class-weighted softmax cross-entropy with label smoothing for :class:`ResidualMLP`.

    Label smoothing (Szegedy et al. 2016) replaces hard one-hot targets with
    soft targets:  ``(1 - ε) * one_hot + ε / K``.  This prevents the model
    from becoming overconfident on small training sets — a common failure mode
    that manifests as high training accuracy but poor generalisation.
    PyTorch's built-in ``F.cross_entropy`` supports this natively via the
    ``label_smoothing`` argument (torch >= 1.10).
    """
    return F.cross_entropy(logits, targets, weight=class_weights, label_smoothing=label_smoothing)


def _coral_loss(
    logits: "torch.Tensor",
    targets: "torch.Tensor",
    class_weights: "torch.Tensor",
    num_classes: int = 3,
) -> "torch.Tensor":
    """CORAL loss: sample-weighted sum of K−1 binary cross-entropies plus a
    rank-monotonicity penalty.

    Binary label at threshold k: ``1`` iff ``target > k``.
    Sample weights are taken from ``class_weights[target]`` so that under-
    represented severity classes receive higher gradient signal.

    Monotonicity penalty
    --------------------
    CORAL requires ``logit_k >= logit_{k+1}`` (i.e. ``P(Y>=k) >= P(Y>=k+1)``).
    Violations are penalised with ``relu(logit_{k+1} - logit_k)`` so the
    bias parameters are nudged back into a valid ordering during training.
    Without this, ``P(Moderate)`` can be clipped to zero for many epochs.
    """
    binary_labels = torch.stack(
        [targets > k for k in range(num_classes - 1)], dim=1
    ).float()  # (batch, K-1)

    per_element = F.binary_cross_entropy_with_logits(
        logits, binary_labels, reduction="none"
    )  # (batch, K-1)

    sample_w = class_weights[targets]  # (batch,)
    denom = sample_w.sum().clamp(min=1e-8)
    bce_loss = (per_element.mean(dim=1) * sample_w).sum() / denom

    # Penalise any threshold pair where logit_{k+1} > logit_{k}
    mono_penalty = 0.0
    if logits.shape[1] > 1:
        mono_penalty = F.relu(logits[:, 1:] - logits[:, :-1]).mean()

    return bce_loss + 0.1 * mono_penalty


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────


def _balanced_class_weights(
    y_np: np.ndarray, num_classes: int, device: "torch.device"
) -> "torch.Tensor":
    """Sklearn-style balanced class weights as a PyTorch tensor."""
    counts = np.bincount(y_np, minlength=num_classes).astype(float)
    w = len(y_np) / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(w, dtype=torch.float32, device=device)


def _train_fold(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    loss_fn,
    device: "torch.device",
    epochs: int = _EPOCHS,
    lr: float = _LR,
    weight_decay: float = _WEIGHT_DECAY,
    batch_size: int = _BATCH_SIZE,
    patience: int = _PATIENCE,
    input_noise_std: float = _INPUT_NOISE_STD,
    seed: int = 0,
) -> None:
    """Train *model* in-place for one LOSO fold.

    Uses AdamW + cosine-annealing LR and restores the checkpoint with the
    lowest validation loss (early stopping).

    Parameters
    ----------
    input_noise_std : float
        Standard deviation of Gaussian noise added to training batches.
        Equivalent to Gaussian data augmentation — shifts the decision boundary
        away from the (sparse) training points, improving generalisation.
        Noise is added only during training; validation and test use clean inputs.
    seed : int
        Controls torch RNG for this specific training run, so that multiple
        calls in a multi-seed ensemble each start from a different random
        parameter initialisation.
    """
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    cw = _balanced_class_weights(y_train, num_classes=3, device=device)

    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.long, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.long, device=device)

    # BatchNorm1d requires ≥ 2 samples per batch during training.
    # Batching directly on GPU via torch.randperm avoids Python DataLoader
    # overhead (collation, dispatch) — critical when training 300+ tiny folds.
    # Equivalent to DataLoader(shuffle=True, drop_last=True) but stays on GPU.
    n_train_t = len(yt)
    eff_bs = max(2, min(batch_size, n_train_t))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float("inf")
    best_ckpt = {k: v.clone() for k, v in model.state_dict().items()}
    stagnant = 0

    epoch_bar = _tqdm(
        range(epochs),
        desc="    epoch",
        leave=False,
        unit="ep",
        bar_format="{l_bar}{bar:18}{r_bar}",
    )
    for _ in epoch_bar:
        model.train()
        perm = torch.randperm(n_train_t, device=device)
        for start in range(0, n_train_t - eff_bs + 1, eff_bs):  # drop_last
            idx = perm[start : start + eff_bs]
            xb, yb = Xt[idx], yt[idx]
            # Gaussian input noise — adds σ-noise to scaled features, acting
            # as a continuous analogue of random feature dropout.  Only during
            # training; noise is NOT added at validation or inference time.
            if input_noise_std > 0.0:
                xb = xb + torch.randn_like(xb) * input_noise_std
            optimizer.zero_grad()
            loss_fn(model(xb), yb, cw).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv, cw).item()

        epoch_bar.set_postfix(val=f"{val_loss:.4f}", stale=stagnant)

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_ckpt = {k: v.clone() for k, v in model.state_dict().items()}
            stagnant = 0
        else:
            stagnant += 1
            if stagnant >= patience:
                epoch_bar.set_description("    epoch [early stop]")
                break

    model.load_state_dict(best_ckpt)


# ─────────────────────────────────────────────────────────────────────────────
# Leave-one-subject-out cross-validation
# ─────────────────────────────────────────────────────────────────────────────


def _loso_cv_nn(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_class,
    model_kwargs: dict,
    loss_fn,
    device: "torch.device",
    val_frac: float = _VAL_FRAC,
    n_seeds: int = _N_SEEDS,
    input_noise_std: float = _INPUT_NOISE_STD,
    desc: str = "",
    **train_kwargs,
) -> dict:
    """Leave-one-subject-out CV for a PyTorch model with multi-seed ensembling.

    For each LOSO fold ``n_seeds`` independent models are trained from different
    random initialisations.  Their softmax / CORAL probabilities are averaged
    before taking the argmax.  Ensemble averaging reduces variance introduced
    by unlucky weight initialisation — particularly important here because many
    folds have fewer than ~20 training samples.

    Parameters
    ----------
    n_seeds : int
        Number of ensemble members per fold.  ``n_seeds=1`` reproduces the
        original single-model behaviour.
    input_noise_std : float
        Passed through to :func:`_train_fold`.  Set to ``0.0`` to disable.

    Returns
    -------
    dict
        ``y_true``, ``y_pred``, ``y_prob`` (averaged class probabilities),
        ``subjects``.
    """
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    unique_subjects = np.unique(groups)
    y_true_all, y_pred_all, y_prob_all, subj_all = [], [], [], []

    fold_bar = _tqdm(
        enumerate(unique_subjects),
        total=len(unique_subjects),
        desc=f"  LOSO {desc}",
        unit="fold",
    )
    for fold_idx, subj in fold_bar:
        test_mask = groups == subj
        trainval_mask = ~test_mask

        n_tv = int(trainval_mask.sum())
        n_test = int(test_mask.sum())
        if n_tv < 10 or n_test < 1:
            continue

        X_tv, y_tv = X[trainval_mask], y[trainval_mask]

        # Deterministic train / validation split — seed varies per fold so
        # different folds do not always select the same relative row indices.
        rng = np.random.RandomState(42 + fold_idx)
        n_val = max(1, int(n_tv * val_frac))
        val_idx = rng.choice(n_tv, size=n_val, replace=False)
        trn_idx = np.setdiff1d(np.arange(n_tv), val_idx)

        X_train, y_train = X_tv[trn_idx], y_tv[trn_idx]
        X_val, y_val = X_tv[val_idx], y_tv[val_idx]
        X_test = X[test_mask]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

        X_t = torch.tensor(X_test, dtype=torch.float32, device=device)

        # ── Multi-seed ensemble ───────────────────────────────────────────────
        # Train n_seeds models from different initialisations, collect their
        # probability tensors, then average.  Each seed uses a globally unique
        # integer so that no two (fold, seed) pairs share the same RNG state.
        seed_probs = []
        for s in range(n_seeds):
            global_seed = fold_idx * 1000 + s  # unique across all (fold, seed) pairs
            fold_bar.set_description(f"  LOSO {desc} [seed {s + 1}/{n_seeds}]")
            model = model_class(**model_kwargs).to(device)
            # Re-initialise model weights with this seed's RNG state so every
            # member of the ensemble starts from a genuinely different point.
            torch.manual_seed(global_seed)
            for m in model.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _train_fold(
                    model,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    loss_fn,
                    device,
                    input_noise_std=input_noise_std,
                    seed=global_seed,
                    **train_kwargs,
                )

            model.eval()
            if isinstance(model, CoralMLP):
                p = model.predict_proba(X_t)  # no_grad internally
            else:
                with torch.no_grad():
                    p = F.softmax(model(X_t), dim=1)
            seed_probs.append(p)

        # Average probabilities across seeds (on CPU to avoid GPU alloc waste)
        probs = torch.stack(seed_probs, dim=0).mean(dim=0)  # (n_test, K)
        fold_bar.set_description(f"  LOSO {desc}")

        preds = probs.argmax(dim=1)
        y_true_all.extend(y[test_mask].tolist())
        y_pred_all.extend(preds.cpu().numpy().tolist())
        y_prob_all.extend(probs.cpu().detach().numpy().tolist())
        subj_all.extend([subj] * n_test)

        # Update fold bar with running balanced accuracy
        if len(y_true_all) >= 2:
            _ba = balanced_accuracy_score(y_true_all, y_pred_all) * 100
            fold_bar.set_postfix(BA=f"{_ba:.1f}%", n=len(y_true_all))

    return {
        "y_true": np.array(y_true_all),
        "y_pred": np.array(y_pred_all),
        "y_prob": np.array(y_prob_all),
        "subjects": subj_all,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance — gradient saliency
# ─────────────────────────────────────────────────────────────────────────────


def _gradient_saliency(
    model: nn.Module,
    X_scaled: np.ndarray,
    y_np: np.ndarray,
    loss_fn,
    device: "torch.device",
) -> np.ndarray:
    """Mean absolute input gradient as feature importance proxy.

    Computes ``E[|∂loss/∂x_j|]`` over all training samples.  A high value
    means the loss is sensitive to that feature — a non-linear analogue of
    linear coefficients.

    Parameters
    ----------
    X_scaled : pre-scaled ``(n_samples, n_features)`` array (same scaler
        that was used during training).
    """
    X_t = torch.tensor(
        X_scaled.astype(np.float32), dtype=torch.float32, device=device, requires_grad=True
    )
    y_t = torch.tensor(y_np, dtype=torch.long, device=device)
    cw = _balanced_class_weights(y_np, num_classes=3, device=device)

    model.train()  # use batch statistics for stable gradients on full data
    loss = loss_fn(model(X_t), y_t, cw)
    loss.backward()

    importance = X_t.grad.abs().mean(dim=0).detach().cpu().numpy()

    # Clear accumulated parameter gradients so the model is not left in a
    # state where a future optimizer.step() would silently apply them.
    model.zero_grad()
    model.eval()
    return importance


def _plot_nn_feature_importance(
    importance: np.ndarray,
    feature_names: list,
    model_name: str,
    output_dir: str,
    save_plots: bool,
    show_plots: bool,
) -> None:
    """Horizontal bar chart of gradient-saliency feature importances (top 20)."""
    idx = np.argsort(importance)[::-1][:20]
    names = [feature_names[i] for i in idx]
    vals = importance[idx]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))
    ax.barh(range(len(names)), vals[::-1], color="steelblue", alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel("Mean |∂Loss / ∂Feature|", fontsize=11)
    ax.set_title(f"{model_name} — Gradient Saliency (top features)", fontsize=12)
    plt.tight_layout()

    if save_plots:
        safe = model_name.replace(" ", "_")
        path = os.path.join(output_dir, f"nn_feature_importance_{safe}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"    Saved feature importance → {path}")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────


def run_nn_classification(
    kinematics_csv_path: str,
    score_csv_path: str,
    id2vid_csv_path: str,
    output_dir: str = ".",
    score_column: str = "score_clean",
    signal_quality_threshold: float = 0.0,
    signal_quality_sub_thresholds: dict | None = None,
    min_cycles: int = 0,
    min_quality_cycles: int = 0,
    min_inter_mcp_span_px: float = 0.0,
    min_detection_rate: float = 0.0,
    recording_angle_csv_path: str | None = None,
    selected_recording_angles: list | None = None,
    video_quality_labels_csv_path: str | None = None,
    video_quality_threshold: int = 3,
    save_plots: bool = True,
    show_plots: bool = False,
    # NN hyper-parameters (tunable via CLI)
    hidden: int = _HIDDEN,
    n_blocks: int = _N_BLOCKS,
    dropout: float = _DROPOUT,
    epochs: int = _EPOCHS,
    lr: float = _LR,
    batch_size: int = _BATCH_SIZE,
    patience: int = _PATIENCE,
    n_seeds: int = _N_SEEDS,
    input_noise_std: float = _INPUT_NOISE_STD,
) -> dict:
    """Run the neural-network classification pipeline (LOSO-CV).

    Parameters mirror :func:`classify.run_classification`.  NN-specific
    hyper-parameters are exposed for easy CLI override.

    Returns
    -------
    dict
        ``metrics_table`` (pd.DataFrame) and ``per_model`` (per-classifier
        LOSO results including ``y_true``, ``y_pred``, ``y_prob``).
    """
    if not _TORCH_OK:
        raise RuntimeError("PyTorch is required.  Install with:  pip install torch")

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print("=" * 60)
    print("NEURAL NETWORK CLASSIFICATION PIPELINE")
    print("=" * 60)
    print(f"  Device   : {device}")
    print(f"  hidden={hidden}  blocks={n_blocks}  dropout={dropout}")
    print(f"  epochs={epochs}  lr={lr}  batch={batch_size}  patience={patience}")
    print(
        f"  n_seeds={n_seeds}  input_noise_std={input_noise_std}  label_smoothing={_LABEL_SMOOTHING}"
    )

    # ── Load & merge ──────────────────────────────────────────────────────────
    print("\nLoading and merging data...")
    merged = _load_and_merge(
        kinematics_csv_path,
        score_csv_path,
        id2vid_csv_path,
        score_column=score_column,
        signal_quality_threshold=signal_quality_threshold,
        signal_quality_sub_thresholds=signal_quality_sub_thresholds,
        min_cycles=min_cycles,
        min_quality_cycles=min_quality_cycles,
        min_inter_mcp_span_px=min_inter_mcp_span_px,
        min_detection_rate=min_detection_rate,
        recording_angle_csv_path=recording_angle_csv_path,
        selected_recording_angles=selected_recording_angles,
        video_quality_labels_csv_path=video_quality_labels_csv_path,
        video_quality_threshold=video_quality_threshold,
    )
    print(f"  Total samples after merge: {len(merged)}")
    print("  Severity distribution:")
    for lbl in CLASS_NAMES:
        print(f"    {lbl}: {int((merged['severity_label'] == lbl).sum())}")

    # ── Feature matrix ────────────────────────────────────────────────────────
    feat_cols = [c for c in CLASSIFICATION_FEATURES if c in merged.columns]
    missing = [c for c in CLASSIFICATION_FEATURES if c not in merged.columns]
    if missing:
        print(f"\n  ⚠ Missing feature columns (excluded): {missing}")
    print(f"  Using {len(feat_cols)} feature columns.")

    valid = merged.dropna(subset=feat_cols + ["severity_int", "subject_id"]).copy()
    print(f"  Samples with complete features: {len(valid)}")

    if len(valid) < 20:
        print("  ⚠ Too few complete samples for NN classification. Exiting.")
        return {}

    X = valid[feat_cols].values.astype(np.float32)
    y = valid["severity_int"].values.astype(np.int64)
    groups = valid["subject_id"].values

    n_subjects = len(np.unique(groups))
    print(f"  Unique subjects (LOSO folds): {n_subjects}")

    # ── Architecture kwargs (shared between both models) ─────────────────────
    arch_kw = dict(
        in_features=X.shape[1],
        num_classes=3,
        hidden=hidden,
        n_blocks=n_blocks,
        dropout=dropout,
    )

    # CoralMLP needs num_classes in the loss closure
    _coral_loss_fn = lambda logits, t, cw: _coral_loss(logits, t, cw, num_classes=3)

    model_registry = {
        "ResidualMLP": (ResidualMLP, _ce_loss, "Multi"),
        "CoralMLP": (CoralMLP, _coral_loss_fn, "Ordinal"),
    }

    train_kw = dict(
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        # input_noise_std is forwarded via loso_kw below, not train_kw,
        # because _loso_cv_nn owns the per-seed training loop.
    )

    # ── LOSO-CV ───────────────────────────────────────────────────────────────
    all_rows = []
    per_model = {}

    for model_name, (model_cls, loss_fn, mode_label) in model_registry.items():
        print(f"\n  [{model_name}]  mode={mode_label}")
        cv = _loso_cv_nn(
            X,
            y,
            groups,
            model_cls,
            arch_kw,
            loss_fn,
            device,
            n_seeds=n_seeds,
            input_noise_std=input_noise_std,
            desc=model_name,
            **train_kw,
        )

        if len(cv["y_true"]) == 0:
            print("    No predictions — skipping.")
            continue

        m = _compute_metrics(cv["y_true"], cv["y_pred"])
        print(f"    Accuracy:          {m['accuracy']:.2f}%")
        print(f"    Balanced Accuracy: {m['balanced_accuracy']:.2f}%")
        print(f"    Macro Precision:   {m['macro_precision']:.2f}%")
        print(f"    Macro F1:          {m['macro_f1']:.2f}%")

        all_rows.append(
            {
                "Model": model_name,
                "Mode": mode_label,
                "Accuracy": round(m["accuracy"], 2),
                "Balanced Acc.": round(m["balanced_accuracy"], 2),
                "Macro Precision": round(m["macro_precision"], 2),
                "Macro F1": round(m["macro_f1"], 2),
                "n": len(cv["y_true"]),
            }
        )
        per_model[model_name] = {**m, "cv_results": cv}

        _plot_confusion_matrix(
            cv["y_true"],
            cv["y_pred"],
            model_name,
            output_dir,
            save_plots,
            show_plots,
        )

    # ── Baselines (reproduced for direct comparison) ──────────────────────────
    for bname, bmet in [
        (
            "Random Guess",
            {
                "accuracy": 100.0 / 3,
                "balanced_accuracy": 100.0 / 3,
                "macro_precision": 100.0 / 3,
                "macro_f1": 100.0 / 3,
            },
        ),
        (
            "Majority Class",
            {
                "accuracy": float(np.max(np.bincount(y)) / len(y) * 100),
                "balanced_accuracy": 100.0 / 3,
                "macro_precision": float(np.max(np.bincount(y)) / len(y) * 100 / 3),
                "macro_f1": float(np.max(np.bincount(y)) / len(y) * 100 / 3),
            },
        ),
    ]:
        all_rows.append(
            {
                "Model": bname,
                "Mode": "—",
                "Accuracy": round(bmet["accuracy"], 2),
                "Balanced Acc.": round(bmet["balanced_accuracy"], 2),
                "Macro Precision": round(bmet["macro_precision"], 2),
                "Macro F1": round(bmet["macro_f1"], 2),
                "n": len(y),
            }
        )

    # ── Feature importance (retrain each model on full dataset) ───────────────
    from sklearn.preprocessing import StandardScaler

    for model_name, (model_cls, loss_fn, _) in model_registry.items():
        if model_name not in per_model:
            continue
        print(f"\n  Computing gradient saliency for {model_name} (full dataset)...")

        scaler_full = StandardScaler()
        X_scaled = scaler_full.fit_transform(X).astype(np.float32)

        final_model = model_cls(**arch_kw).to(device)
        Xf_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
        yf_t = torch.tensor(y, dtype=torch.long, device=device)
        cw_f = _balanced_class_weights(y, num_classes=3, device=device)

        n_full = len(yf_t)
        eff_bs_f = max(2, min(batch_size, n_full))
        opt_f = optim.AdamW(final_model.parameters(), lr=lr, weight_decay=_WEIGHT_DECAY)
        final_model.train()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(min(epochs, 100)):
                perm_f = torch.randperm(n_full, device=device)
                for start in range(0, n_full - eff_bs_f + 1, eff_bs_f):
                    idx = perm_f[start : start + eff_bs_f]
                    xb, yb = Xf_t[idx], yf_t[idx]
                    opt_f.zero_grad()
                    loss_fn(final_model(xb), yb, cw_f).backward()
                    nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
                    opt_f.step()

        importance = _gradient_saliency(final_model, X_scaled, y, loss_fn, device)
        _plot_nn_feature_importance(
            importance,
            feat_cols,
            model_name,
            output_dir,
            save_plots,
            show_plots,
        )

    # ── Summary table ─────────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(all_rows)
    print(f"\n{'='*60}")
    print("NEURAL NETWORK PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    print(metrics_df.to_string(index=False))

    if save_plots:
        csv_path = os.path.join(output_dir, "nn_classification_results.csv")
        metrics_df.to_csv(csv_path, index=False)
        print(f"\n  Results saved → {csv_path}")

    return {"metrics_table": metrics_df, "per_model": per_model}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Neural-network PD severity classification (LOSO-CV).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--kinematics", required=True, help="Path to tracking_logs.csv")
    p.add_argument("--scores", required=True, help="Path to clinical scores CSV")
    p.add_argument("--id2vid", default="", help="Path to id2vid.csv (optional)")
    p.add_argument("--output", default="results/classify_nn")
    p.add_argument("--score-column", default="ProS")
    # Filters (mirror analyze.py / classify.py)
    p.add_argument(
        "--signal-quality",
        type=float,
        default=0.0,
        help="Minimum Signal Quality threshold (0 = no filter)",
    )
    p.add_argument(
        "--min-detection-rate",
        type=float,
        default=0.0,
        help="Minimum VQ_detection_rate (0 = no filter)",
    )
    p.add_argument(
        "--min-inter-mcp-span-px",
        type=float,
        default=0.0,
        help="Minimum VQ_inter_mcp_span_px in pixels (0 = no filter)",
    )
    p.add_argument(
        "--min-cycles", type=int, default=0, help="Minimum Total Cycles required (0 = no filter)"
    )
    p.add_argument(
        "--min-quality-cycles",
        type=int,
        default=0,
        help="Minimum Quality Cycles required (0 = no filter)",
    )
    p.add_argument(
        "--recording-angle-csv",
        default="",
        help="Path to recording-angle labels CSV/Excel (optional)",
    )
    p.add_argument(
        "--selected-recording-angles",
        nargs="*",
        default=None,
        help="Allow-list of recording angles to keep, e.g. front angled",
    )
    p.add_argument(
        "--video-quality-labels-csv",
        default="",
        help="Path to manual video-quality labels CSV with video_path and quality_label",
    )
    p.add_argument(
        "--video-quality-threshold",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Keep videos with quality_label <= threshold (1=best, 3=worst)",
    )
    # NN hyper-parameters
    p.add_argument("--hidden", type=int, default=_HIDDEN)
    p.add_argument("--n-blocks", type=int, default=_N_BLOCKS)
    p.add_argument("--dropout", type=float, default=_DROPOUT)
    p.add_argument("--epochs", type=int, default=_EPOCHS)
    p.add_argument("--lr", type=float, default=_LR)
    p.add_argument("--batch-size", type=int, default=_BATCH_SIZE)
    p.add_argument("--patience", type=int, default=_PATIENCE)
    p.add_argument(
        "--n-seeds",
        type=int,
        default=_N_SEEDS,
        help="Ensemble seeds per LOSO fold (more seeds = lower variance, more compute)",
    )
    p.add_argument(
        "--input-noise",
        type=float,
        default=_INPUT_NOISE_STD,
        dest="input_noise",
        help="σ of Gaussian noise added to scaled inputs during training (0 = off)",
    )
    p.add_argument("--show-plots", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    run_nn_classification(
        kinematics_csv_path=_args.kinematics,
        score_csv_path=_args.scores,
        id2vid_csv_path=_args.id2vid,
        output_dir=_args.output,
        score_column=_args.score_column,
        signal_quality_threshold=_args.signal_quality,
        min_detection_rate=_args.min_detection_rate,
        min_inter_mcp_span_px=_args.min_inter_mcp_span_px,
        min_cycles=_args.min_cycles,
        min_quality_cycles=_args.min_quality_cycles,
        recording_angle_csv_path=_args.recording_angle_csv or None,
        selected_recording_angles=_args.selected_recording_angles,
        video_quality_labels_csv_path=_args.video_quality_labels_csv or None,
        video_quality_threshold=_args.video_quality_threshold,
        save_plots=True,
        show_plots=_args.show_plots,
        hidden=_args.hidden,
        n_blocks=_args.n_blocks,
        dropout=_args.dropout,
        epochs=_args.epochs,
        lr=_args.lr,
        batch_size=_args.batch_size,
        patience=_args.patience,
        n_seeds=_args.n_seeds,
        input_noise_std=_args.input_noise,
    )
