"""
Linear probing layer-wise con LogisticRegression.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - dipendenza opzionale
    torch = None

if TYPE_CHECKING:
    from sklearn.linear_model import LogisticRegression


def train_linear_probe(
    representations: "torch.Tensor | np.ndarray",
    labels: "torch.Tensor | np.ndarray",
    *,
    solver: str = "lbfgs",
    max_iter: int = 1000,
    random_state: int = 42,
) -> "LogisticRegression":
    """
    Addestra un classificatore LogisticRegression su un singolo layer.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover
        raise ImportError("scikit-learn non installato. Esegui: pip install scikit-learn") from exc

    X = _to_numpy_2d(representations)
    y = _to_numpy_1d(labels)
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Numero campioni incoerente: X={X.shape[0]} vs y={y.shape[0]}"
        )

    clf = LogisticRegression(
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
    )
    clf.fit(X, y)
    return clf


def evaluate_linear_probe(
    model: "LogisticRegression",
    representations: "torch.Tensor | np.ndarray",
    labels: "torch.Tensor | np.ndarray",
) -> float:
    """
    Valuta il classificatore su held-out e restituisce acc(l).
    """
    X = _to_numpy_2d(representations)
    y = _to_numpy_1d(labels)
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Numero campioni incoerente: X={X.shape[0]} vs y={y.shape[0]}"
        )
    y_hat = model.predict(X)
    return float(np.mean(y_hat == y))


def layerwise_linear_probing(
    activations_per_layer: Iterable["torch.Tensor | np.ndarray"],
    labels: "torch.Tensor | np.ndarray",
    train_mask: "torch.Tensor | np.ndarray",
    val_mask: "torch.Tensor | np.ndarray",
    *,
    solver: str = "lbfgs",
    max_iter: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    """
    Esegue il ciclo completo di linear probing layer-wise.
    """
    y = _to_numpy_1d(labels)
    train_idx = _mask_to_indices(train_mask, y.shape[0], "train_mask")
    val_idx = _mask_to_indices(val_mask, y.shape[0], "val_mask")
    if train_idx.size == 0 or val_idx.size == 0:
        raise ValueError("train_mask e val_mask devono selezionare almeno un campione")

    scores: list[float] = []
    for layer_acts in activations_per_layer:
        X = _to_numpy_2d(layer_acts)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Layer con campioni incoerenti: X={X.shape[0]} vs y={y.shape[0]}"
            )

        model = train_linear_probe(
            X[train_idx],
            y[train_idx],
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
        )
        acc = evaluate_linear_probe(model, X[val_idx], y[val_idx])
        scores.append(acc)

    return np.asarray(scores, dtype=np.float64)


def _to_numpy_2d(x: "torch.Tensor | np.ndarray") -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"Atteso array 2D [n, d], ricevuto shape={arr.shape}")
    return arr


def _to_numpy_1d(x: "torch.Tensor | np.ndarray") -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"Atteso array 1D [n], ricevuto shape={arr.shape}")
    return arr


def _mask_to_indices(
    mask_or_idx: "torch.Tensor | np.ndarray",
    n_samples: int,
    name: str,
) -> np.ndarray:
    if torch is not None and isinstance(mask_or_idx, torch.Tensor):
        arr = mask_or_idx.detach().cpu().numpy()
    else:
        arr = np.asarray(mask_or_idx)

    if arr.ndim != 1:
        raise ValueError(f"{name} deve essere 1D, ricevuto shape={arr.shape}")

    if arr.dtype == bool:
        if arr.shape[0] != n_samples:
            raise ValueError(
                f"{name} booleano deve avere lunghezza {n_samples}, trovato {arr.shape[0]}"
            )
        return np.flatnonzero(arr)

    idx = arr.astype(np.int64, copy=False)
    if np.any(idx < 0) or np.any(idx >= n_samples):
        raise ValueError(f"{name} contiene indici fuori range [0, {n_samples})")
    return idx

