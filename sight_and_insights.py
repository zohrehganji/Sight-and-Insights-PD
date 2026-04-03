
"""
Sight & Insights — Fundus-only pipeline for participant-level Parkinson's disease risk stratification.

Aligned components:
- bilateral color fundus photographs
- shared-weight Siamese EfficientNetV2-B1
- retina-aware preprocessing (LAB + CLAHE + vessel masking)
- channel + spatial attention
- multi-scale convolutional block
- participant-level averaging
- focal loss
- optional overlap weighting (age/sex propensity)
- stratified group CV on TRAIN
- isotonic calibration on OOF predictions
- frozen threshold selection
- Grad-CAM and vessel-aware LIME

Expected layout:
data/
  Healthy/
  Parkinson/

Recommended filename patterns:
  <participant_id>_OD.png
  <participant_id>_OS.png

Optional metadata file may contain:
  participant_id, age, sex
Used only for overlap weighting, not as predictive inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from lime import lime_image
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from skimage.filters import frangi
from skimage.segmentation import slic
from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetV2B1
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.layers import (
    Average,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    Layer,
    Multiply,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    data_dir: str = "./data"
    output_dir: str = "./outputs_fundus"
    metadata_path: Optional[str] = None

    image_size: Tuple[int, int] = (256, 256)
    batch_size: int = 8
    epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay_l2: float = 1e-5
    dropout_rate: float = 0.4
    freeze_layers: int = 200

    random_seed: int = 42
    holdout_split: float = 0.2
    folds: int = 5

    min_sensitivity: float = 0.70
    min_specificity: float = 0.70

    rotation_deg: float = 45.0
    shift_frac: float = 0.20
    shear_frac: float = 0.20
    zoom_frac: float = 0.30
    brightness_low: float = 0.70
    brightness_high: float = 1.30

    xai_samples: int = 5
    lime_samples: int = 1000
    vessel_threshold: float = 0.10

    mixed_precision: bool = True
    class_names: Tuple[str, str] = ("Healthy", "Parkinson")


# =============================================================================
# Utilities
# =============================================================================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    raise ValueError(f"Unsupported metadata format: {p.suffix}")


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1.0 + (z**2 / n)
    center = (phat + z**2 / (2.0 * n)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) + z**2 / (4.0 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


# =============================================================================
# Data parsing
# =============================================================================

def parse_participant_id_and_eye(filename: str) -> Tuple[str, Optional[str]]:
    stem = Path(filename).stem
    eye = None

    if re.search(r"(?:^|[_\-\s])(OD|od|R|right)(?:$|[_\-\s])", stem):
        eye = "OD"
    elif re.search(r"(?:^|[_\-\s])(OS|os|L|left)(?:$|[_\-\s])", stem):
        eye = "OS"

    cleaned = re.sub(r"(?:[_\-\s]?)(?:OD|od|OS|os|R|r|L|l|right|left)(?:[_\-\s]?)", "_", stem)
    cleaned = re.sub(r"[_\-\s]+", "_", cleaned).strip("_")
    tokens = cleaned.split("_")
    participant_id = tokens[0] if tokens else cleaned
    return participant_id, eye


def build_image_dataframe(data_dir: str, class_names: Sequence[str]) -> pd.DataFrame:
    rows: List[dict] = []
    data_root = Path(data_dir)

    class_map = {
        class_names[0].lower(): 0,
        class_names[1].lower(): 1,
        "healthy": 0,
        "parkinson": 1,
        "hc": 0,
        "pd": 1,
    }

    for cls_dir in data_root.iterdir():
        if not cls_dir.is_dir():
            continue
        key = cls_dir.name.lower()
        if key not in class_map:
            continue
        label = class_map[key]

        for file_path in cls_dir.rglob("*"):
            if file_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                continue
            pid, eye = parse_participant_id_and_eye(file_path.name)
            rows.append(
                {
                    "participant_id": str(pid),
                    "eye": eye,
                    "label": label,
                    "path": str(file_path),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No fundus images found under: {data_dir}")
    return df


def pair_bilateral_images(df_eye: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, g in df_eye.groupby("participant_id", sort=False):
        label = int(g["label"].iloc[0])
        od_path, os_path = None, None
        for _, r in g.iterrows():
            if r.get("eye") == "OD":
                od_path = r["path"]
            elif r.get("eye") == "OS":
                os_path = r["path"]

        if od_path is None and os_path is None and len(g) == 2:
            od_path = g.iloc[0]["path"]
            os_path = g.iloc[1]["path"]

        rows.append(
            {
                "participant_id": pid,
                "label": label,
                "od_path": od_path,
                "os_path": os_path,
                "n_eyes": int(pd.notna(od_path)) + int(pd.notna(os_path)),
            }
        )
    out = pd.DataFrame(rows)
    out["n_eyes"] = out["n_eyes"].astype(int)
    return out


def attach_optional_metadata(df_part: pd.DataFrame, metadata_path: Optional[str]) -> pd.DataFrame:
    if metadata_path is None:
        return df_part
    meta = load_table(metadata_path)
    if "participant_id" not in meta.columns:
        raise ValueError("Metadata file must contain a participant_id column.")
    meta = meta.copy()
    meta["participant_id"] = meta["participant_id"].astype(str)
    meta = meta.drop_duplicates("participant_id")
    merged = df_part.merge(meta, on="participant_id", how="left", suffixes=("", "_meta"))
    if "label_meta" in merged.columns:
        merged["label"] = merged["label_meta"].fillna(merged["label"]).astype(int)
        merged = merged.drop(columns=["label_meta"])
    return merged


# =============================================================================
# Preprocessing
# =============================================================================

def retina_preprocess(img_bgr: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    try:
        img_bgr = img_bgr.astype(np.uint8)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab2 = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

        green = img[:, :, 1]
        blurred = cv2.GaussianBlur(green, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = np.repeat(thresh[:, :, None], 3, axis=2)
        img = cv2.bitwise_and(img, mask)

        img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
        return img.astype(np.float32) / 255.0
    except Exception:
        return np.zeros((*image_size, 3), dtype=np.float32)


def load_image(path: Optional[str], image_size: Tuple[int, int]) -> np.ndarray:
    if path is None or (isinstance(path, float) and np.isnan(path)):
        return np.zeros((*image_size, 3), dtype=np.float32)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((*image_size, 3), dtype=np.float32)
    return retina_preprocess(img, image_size)


def apply_same_affine(img1: np.ndarray, img2: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img1.shape[:2]
    angle = random.uniform(-cfg.rotation_deg, cfg.rotation_deg)
    tx = random.uniform(-cfg.shift_frac, cfg.shift_frac) * w
    ty = random.uniform(-cfg.shift_frac, cfg.shift_frac) * h
    shear = random.uniform(-cfg.shear_frac, cfg.shear_frac)
    zoom = 1.0 + random.uniform(-cfg.zoom_frac, cfg.zoom_frac)

    center = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(center, angle, zoom)
    m[0, 1] += shear
    m[1, 0] += shear
    m[0, 2] += tx
    m[1, 2] += ty

    img1 = cv2.warpAffine(img1, m, dsize=(w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    img2 = cv2.warpAffine(img2, m, dsize=(w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    if random.random() < 0.5:
        img1 = cv2.flip(img1, 1)
        img2 = cv2.flip(img2, 1)

    if random.random() < 0.2:
        img1 = cv2.flip(img1, 0)
        img2 = cv2.flip(img2, 0)

    brightness = random.uniform(cfg.brightness_low, cfg.brightness_high)
    img1 = np.clip(img1 * brightness, 0.0, 1.0)
    img2 = np.clip(img2 * brightness, 0.0, 1.0)

    if random.random() < 0.35:
        n1 = np.random.normal(0.0, 0.02, img1.shape).astype(np.float32)
        n2 = np.random.normal(0.0, 0.02, img2.shape).astype(np.float32)
        img1 = np.clip(img1 + n1, 0.0, 1.0)
        img2 = np.clip(img2 + n2, 0.0, 1.0)

    return img1.astype(np.float32), img2.astype(np.float32)


# =============================================================================
# Dataset sequence
# =============================================================================

class BilateralFundusSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: Tuple[int, int],
        batch_size: int,
        shuffle: bool = True,
        augment: bool = False,
        config: Optional[Config] = None,
        sample_weights: Optional[np.ndarray] = None,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.config = config or Config()
        self.sample_weights = sample_weights
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(math.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx: int):
        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch = self.df.iloc[batch_idx]

        od_imgs, os_imgs, labels, sw = [], [], [], []
        for row_idx, (_, row) in zip(batch_idx, batch.iterrows()):
            od = load_image(row.get("od_path"), self.image_size)
            os_img = load_image(row.get("os_path"), self.image_size)
            if self.augment:
                od, os_img = apply_same_affine(od, os_img, self.config)

            od_imgs.append(od)
            os_imgs.append(os_img)
            labels.append(float(row["label"]))
            if self.sample_weights is not None:
                sw.append(float(self.sample_weights[row_idx]))

        x = {
            "od_input": np.asarray(od_imgs, dtype=np.float32),
            "os_input": np.asarray(os_imgs, dtype=np.float32),
        }
        y = np.asarray(labels, dtype=np.float32)
        if self.sample_weights is not None:
            return x, y, np.asarray(sw, dtype=np.float32)
        return x, y


# =============================================================================
# Overlap weighting
# =============================================================================

def compute_overlap_weights(df: pd.DataFrame, age_col: str = "age", sex_col: str = "sex") -> np.ndarray:
    if age_col not in df.columns:
        return np.ones(len(df), dtype=np.float32)

    work = df.copy()
    age = pd.to_numeric(work[age_col], errors="coerce").astype(float)
    age = age.fillna(age.median())
    z = (age - age.mean()) / (age.std() + 1e-8)

    X = pd.DataFrame({"z": z, "z2": z**2})
    if sex_col in work.columns:
        sex = work[sex_col].astype(str).str.lower().map({"m": 1, "male": 1, "1": 1, "f": 0, "female": 0, "0": 0})
        sex = sex.fillna(sex.median() if not sex.dropna().empty else 0)
        X["sex"] = sex
        X["z_sex"] = z * sex

    y = work["label"].astype(int).values
    lr = LogisticRegression(max_iter=2000, solver="lbfgs")
    lr.fit(X.values, y)
    e = lr.predict_proba(X.values)[:, 1]
    w = np.where(y == 1, 1.0 - e, e)
    w = np.clip(w, 1e-3, None)
    w = w / np.mean(w)
    return w.astype(np.float32)


# =============================================================================
# Model blocks
# =============================================================================

class ChannelAttention(Layer):
    def __init__(self, reduction: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        channels = int(input_shape[-1])
        hidden = max(8, channels // self.reduction)
        self.fc1 = Dense(hidden, activation="relu", kernel_initializer="he_normal")
        self.fc2 = Dense(channels, activation="sigmoid", kernel_initializer="he_normal")
        super().build(input_shape)

    def call(self, x, training=None):
        avg = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        mx = tf.reduce_max(x, axis=[1, 2], keepdims=True)
        scale = self.fc2(self.fc1(avg)) + self.fc2(self.fc1(mx))
        return x * scale, scale


class SpatialAttention(Layer):
    def __init__(self, kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = Conv2D(1, kernel_size=self.kernel_size, padding="same", activation="sigmoid", use_bias=False)
        super().build(input_shape)

    def call(self, x, training=None):
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        attn = self.conv(tf.concat([avg_pool, max_pool], axis=-1))
        return x * attn, attn


class MultiScaleBlock(Layer):
    def __init__(self, filters: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.c1 = Conv2D(self.filters, (1, 1), padding="same", activation="swish", kernel_initializer="he_normal")
        self.c3 = Conv2D(self.filters, (3, 3), padding="same", activation="swish", kernel_initializer="he_normal")
        self.c5 = Conv2D(self.filters, (5, 5), padding="same", activation="swish", kernel_initializer="he_normal")
        self.fuse = Conv2D(self.filters * 2, (3, 3), padding="same", activation="swish", kernel_initializer="he_normal")
        self.bn = BatchNormalization()
        super().build(input_shape)

    def call(self, x, training=None):
        x = Concatenate()([self.c1(x), self.c3(x), self.c5(x)])
        x = self.fuse(x)
        x = self.bn(x, training=training)
        return x


# =============================================================================
# Model builders
# =============================================================================

def build_eye_encoder(cfg: Config, name: str = "shared_eye_encoder") -> Model:
    inputs = Input(shape=(*cfg.image_size, 3), name=f"{name}_input")
    base = EfficientNetV2B1(include_top=False, weights="imagenet", input_shape=(*cfg.image_size, 3))

    for layer in base.layers[: cfg.freeze_layers]:
        layer.trainable = False

    x = base(inputs)
    x, _ = ChannelAttention(reduction=8, name=f"{name}_ca1")(x)
    x, _ = SpatialAttention(kernel_size=7, name=f"{name}_sa1")(x)
    x = MultiScaleBlock(filters=128, name=f"{name}_ms")(x)
    x, _ = ChannelAttention(reduction=8, name=f"{name}_ca2")(x)
    x = GlobalAveragePooling2D(name=f"{name}_gap")(x)
    x = Dropout(cfg.dropout_rate, name=f"{name}_drop1")(x)
    x = Dense(256, activation="swish", kernel_regularizer=l2(cfg.weight_decay_l2), name=f"{name}_dense1")(x)
    x = BatchNormalization(name=f"{name}_bn1")(x)
    x = Dropout(cfg.dropout_rate, name=f"{name}_drop2")(x)
    outputs = Dense(1, activation="sigmoid", name=f"{name}_eye_prob")(x)
    return Model(inputs=inputs, outputs=outputs, name=name)


def build_bilateral_model(cfg: Config) -> Tuple[Model, Model]:
    eye_encoder = build_eye_encoder(cfg, name="shared_eye_encoder")

    od_input = Input(shape=(*cfg.image_size, 3), name="od_input")
    os_input = Input(shape=(*cfg.image_size, 3), name="os_input")

    od_prob = eye_encoder(od_input)
    os_prob = eye_encoder(os_input)

    participant_prob = Average(name="participant_average")([od_prob, os_prob])
    model = Model(
        inputs={"od_input": od_input, "os_input": os_input},
        outputs=participant_prob,
        name="Bilateral_Fundus_PD",
    )
    return model, eye_encoder


# =============================================================================
# Loss / calibration / metrics
# =============================================================================

def focal_loss(alpha: float = 0.65, gamma: float = 2.0):
    alpha = float(alpha)
    gamma = float(gamma)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        at = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)
        return tf.reduce_mean(-at * tf.pow(1.0 - pt, gamma) * tf.math.log(pt))

    return loss_fn


class IsotonicCalibrator:
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, y_score: np.ndarray, y_true: np.ndarray):
        self.iso.fit(np.asarray(y_score).ravel(), np.asarray(y_true).ravel())
        return self

    def predict(self, y_score: np.ndarray) -> np.ndarray:
        return self.iso.transform(np.asarray(y_score).ravel())


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, min_sens: float, min_spec: float) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    spec = 1.0 - fpr
    valid = np.where((tpr >= min_sens) & (spec >= min_spec))[0]
    if len(valid) > 0:
        hmean = 2 * (tpr[valid] * spec[valid]) / (tpr[valid] + spec[valid] + 1e-12)
        best = valid[np.argmax(hmean)]
        return float(thresholds[best])
    youden = tpr - fpr
    return float(thresholds[np.argmax(youden)])


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float((tp + tn) / max(1, len(y_true))),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "threshold": float(threshold),
        "ppr": float(np.mean(y_pred)),
    }


# =============================================================================
# Plotting
# =============================================================================

def roc_curve_point(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Tuple[float, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return fpr, tpr


def plot_performance_panels(
    y_true: np.ndarray,
    y_prob_raw: np.ndarray,
    y_prob_cal: np.ndarray,
    threshold: float,
    title_prefix: str,
    out_dir: Path,
    prefix: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    y_pred = (y_prob_cal >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    cm = np.array([[tn, fp], [fn, tp]])
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"{title_prefix} Confusion Matrix")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["HC", "PD"])
    ax.set_yticklabels(["HC", "PD"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, y_prob_cal)
    prec, rec, _ = precision_recall_curve(y_true, y_prob_cal)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_true, y_prob_cal):.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].scatter(*roc_curve_point(y_true, y_prob_cal, threshold), s=60, label=f"thr={threshold:.3f}")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"{title_prefix} ROC")
    axes[0].legend(loc="lower right")

    axes[1].plot(rec, prec, label=f"AP = {average_precision_score(y_true, y_prob_cal):.3f}")
    prevalence = float(np.mean(y_true))
    axes[1].plot([0, 1], [prevalence, prevalence], linestyle="--", color="gray", label=f"Prevalence = {prevalence:.2f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"{title_prefix} PR")
    axes[1].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_roc_pr.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(y_prob_cal[y_true == 0], bins=20, alpha=0.6, label="HC")
    ax.hist(y_prob_cal[y_true == 1], bins=20, alpha=0.6, label="PD")
    ax.axvline(threshold, linestyle="--")
    ax.set_xlabel("Calibrated probability")
    ax.set_ylabel("Count")
    ax.set_title(f"{title_prefix} Scores")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_score_hist.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    frac_pos_raw, mean_pred_raw = calibration_curve(y_true, y_prob_raw, n_bins=10, strategy="quantile")
    frac_pos_cal, mean_pred_cal = calibration_curve(y_true, y_prob_cal, n_bins=10, strategy="quantile")
    brier_raw = brier_score_loss(y_true, y_prob_raw)
    brier_cal = brier_score_loss(y_true, y_prob_cal)

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    ax.plot(mean_pred_raw, frac_pos_raw, marker="o", label=f"Raw (Brier={brier_raw:.3f})")
    ax.plot(mean_pred_cal, frac_pos_cal, marker="o", label=f"Calibrated (Brier={brier_cal:.3f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction positive")
    ax.set_title(f"{title_prefix} Calibration")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# XAI
# =============================================================================

def get_last_conv_layer_name(eye_encoder: Model) -> str:
    for layer in reversed(eye_encoder.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in eye encoder for Grad-CAM.")


def gradcam_for_eye(eye_encoder: Model, eye_img: np.ndarray) -> np.ndarray:
    last_conv_layer_name = get_last_conv_layer_name(eye_encoder)
    conv_layer = eye_encoder.get_layer(last_conv_layer_name)
    grad_model = Model(inputs=eye_encoder.input, outputs=[conv_layer.output, eye_encoder.output])
    img = np.expand_dims(eye_img.astype(np.float32), axis=0)

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = cv2.resize(heatmap.numpy(), (eye_img.shape[1], eye_img.shape[0]))
    return heatmap


def extract_vessel_mask(image_rgb: np.ndarray, vessel_threshold: float = 0.10) -> np.ndarray:
    gray = cv2.cvtColor((image_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    vessels = frangi(gray, scale_range=(1, 3), scale_step=2, beta1=0.5, beta2=15, black_ridges=False)
    return (vessels > vessel_threshold).astype(np.uint8)


def save_gradcam_panel(eye_img: np.ndarray, heatmap: np.ndarray, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].imshow(eye_img)
    axes[0].set_title("Original eye image")
    axes[0].axis("off")
    axes[1].imshow(eye_img)
    axes[1].imshow(heatmap, cmap="jet", alpha=0.45)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def vessel_aware_lime_explain(
    model: Model,
    eye_rgb: np.ndarray,
    vessel_mask: np.ndarray,
    save_path: str,
    num_samples: int = 1000,
):
    explainer = lime_image.LimeImageExplainer()

    def predict_fn(img_batch: np.ndarray) -> np.ndarray:
        od = img_batch.astype(np.float32)
        os_zeros = np.zeros_like(od, dtype=np.float32)
        preds = model.predict({"od_input": od, "os_input": os_zeros}, verbose=0)
        return np.hstack([1.0 - preds, preds])

    def segmentation_fn(img: np.ndarray):
        vessel_channel = cv2.resize(vessel_mask, (img.shape[1], img.shape[0]))
        four_channel = np.dstack([img, vessel_channel])
        return slic(four_channel, n_segments=50, compactness=10, sigma=1, start_label=0)

    explanation = explainer.explain_instance(
        eye_rgb,
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=num_samples,
        segmentation_fn=segmentation_fn,
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=10,
        hide_rest=False,
    )

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(eye_rgb)
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].imshow(vessel_mask, cmap="gray")
    ax[1].set_title("Vessel mask")
    ax[1].axis("off")

    ax[2].imshow(lime_image.mark_boundaries(temp / 255.0, mask))
    ax[2].set_title("Vessel-aware LIME")
    ax[2].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return explanation


# =============================================================================
# Training helpers
# =============================================================================

def build_sequences(
    df: pd.DataFrame,
    cfg: Config,
    augment: bool,
    shuffle: bool = True,
    sample_weights: Optional[np.ndarray] = None,
) -> BilateralFundusSequence:
    return BilateralFundusSequence(
        df=df,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        augment=augment,
        config=cfg,
        sample_weights=sample_weights,
    )


def compile_model(model: Model, cfg: Config, alpha: float = 0.65, gamma: float = 2.0) -> Model:
    model.compile(
        optimizer=Adam(learning_rate=cfg.learning_rate),
        loss=focal_loss(alpha=alpha, gamma=gamma),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def train_single_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: Config,
    alpha: float = 0.65,
    gamma: float = 2.0,
) -> Tuple[Model, Model, Dict[str, float], np.ndarray, np.ndarray, int]:
    train_weights = compute_overlap_weights(train_df) if "age" in train_df.columns else np.ones(len(train_df), dtype=np.float32)

    train_seq = build_sequences(train_df, cfg, augment=True, shuffle=True, sample_weights=train_weights)
    val_seq = build_sequences(val_df, cfg, augment=False, shuffle=False)

    model, eye_encoder = build_bilateral_model(cfg)
    model = compile_model(model, cfg, alpha=alpha, gamma=gamma)

    callbacks = [
        EarlyStopping(monitor="val_pr_auc", mode="max", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=3, min_lr=1e-6),
        TerminateOnNaN(),
    ]

    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=cfg.epochs,
        verbose=1,
        callbacks=callbacks,
    )

    y_val = val_df["label"].astype(int).values
    y_prob = model.predict(val_seq, verbose=0).ravel()
    best_epoch = int(np.argmax(history.history.get("val_pr_auc", [0.0])) + 1)

    return model, eye_encoder, {k: float(v[-1]) for k, v in history.history.items()}, y_val, y_prob, best_epoch


def cross_validated_oof_predictions(
    train_df: pd.DataFrame,
    cfg: Config,
    alpha: float = 0.65,
    gamma: float = 2.0,
):
    y = train_df["label"].astype(int).values
    groups = train_df["participant_id"].values
    splitter = StratifiedGroupKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.random_seed)

    oof_prob = np.zeros(len(train_df), dtype=np.float32)
    fold_best_epochs: List[int] = []
    fold_val_ap: List[float] = []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(train_df, y, groups=groups), start=1):
        tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        va_df = train_df.iloc[va_idx].reset_index(drop=True)

        model, eye_encoder, history_last, y_val, y_prob, best_epoch = train_single_fold(
            tr_df, va_df, cfg, alpha=alpha, gamma=gamma
        )
        oof_prob[va_idx] = y_prob
        fold_best_epochs.append(best_epoch)
        fold_val_ap.append(float(average_precision_score(y_val, y_prob)))
        print(f"Fold {fold}: best_epoch={best_epoch}, val_PR_AUC={fold_val_ap[-1]:.4f}")

    return y, oof_prob, fold_best_epochs, fold_val_ap


def fit_final_model(
    train_df: pd.DataFrame,
    cfg: Config,
    epochs: int,
    alpha: float = 0.65,
    gamma: float = 2.0,
) -> Tuple[Model, Model]:
    train_weights = compute_overlap_weights(train_df) if "age" in train_df.columns else np.ones(len(train_df), dtype=np.float32)
    train_seq = build_sequences(train_df, cfg, augment=True, shuffle=True, sample_weights=train_weights)

    model, eye_encoder = build_bilateral_model(cfg)
    model = compile_model(model, cfg, alpha=alpha, gamma=gamma)

    model.fit(
        train_seq,
        epochs=max(1, int(epochs)),
        verbose=1,
        callbacks=[
            ReduceLROnPlateau(monitor="loss", mode="min", factor=0.5, patience=3, min_lr=1e-6),
            TerminateOnNaN(),
        ],
    )
    return model, eye_encoder


# =============================================================================
# Pipeline
# =============================================================================

def load_participant_dataframe(cfg: Config) -> pd.DataFrame:
    df_eye = build_image_dataframe(cfg.data_dir, cfg.class_names)
    df_part = pair_bilateral_images(df_eye)
    df_part = attach_optional_metadata(df_part, cfg.metadata_path)
    return df_part


def split_train_holdout(df_part: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, holdout_df = train_test_split(
        df_part,
        test_size=cfg.holdout_split,
        stratify=df_part["label"],
        random_state=cfg.random_seed,
    )
    return train_df.reset_index(drop=True), holdout_df.reset_index(drop=True)


def run_pipeline(cfg: Config) -> Dict[str, dict]:
    set_global_seed(cfg.random_seed)

    if cfg.mixed_precision:
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
        except Exception:
            pass

    out_dir = ensure_dir(cfg.output_dir)
    ensure_dir(out_dir / "models")
    ensure_dir(out_dir / "xai")
    ensure_dir(out_dir / "figures")

    print("Loading data ...")
    df_part = load_participant_dataframe(cfg)
    print(f"Participants: {len(df_part)}")
    print(df_part["label"].value_counts().sort_index())

    train_df, holdout_df = split_train_holdout(df_part, cfg)
    print(f"TRAIN: {len(train_df)} participants | HOLDOUT: {len(holdout_df)} participants")

    # OOF calibration on TRAIN
    print("\nGenerating TRAIN out-of-fold predictions ...")
    y_train, oof_prob, fold_best_epochs, fold_val_ap = cross_validated_oof_predictions(train_df, cfg)

    calibrator = IsotonicCalibrator().fit(oof_prob, y_train)
    oof_prob_cal = calibrator.predict(oof_prob)

    threshold = find_optimal_threshold(
        y_train,
        oof_prob_cal,
        min_sens=cfg.min_sensitivity,
        min_spec=cfg.min_specificity,
    )

    oof_metrics = evaluate_predictions(y_train, oof_prob_cal, threshold)
    oof_metrics["sensitivity_ci_95"] = wilson_ci(oof_metrics["tp"], oof_metrics["tp"] + oof_metrics["fn"])
    oof_metrics["specificity_ci_95"] = wilson_ci(oof_metrics["tn"], oof_metrics["tn"] + oof_metrics["fp"])
    oof_metrics["ppv_ci_95"] = wilson_ci(oof_metrics["tp"], oof_metrics["tp"] + oof_metrics["fp"])
    oof_metrics["npv_ci_95"] = wilson_ci(oof_metrics["tn"], oof_metrics["tn"] + oof_metrics["fn"])

    print("TRAIN OOF metrics:", oof_metrics)

    # Final model on full TRAIN
    final_epochs = int(np.clip(np.median(fold_best_epochs), 5, cfg.epochs))
    print(f"\nTraining final model on full TRAIN for {final_epochs} epochs ...")
    final_model, eye_encoder = fit_final_model(train_df, cfg, epochs=final_epochs)

    # Hold-out evaluation
    hold_seq = build_sequences(holdout_df, cfg, augment=False, shuffle=False)
    y_hold = holdout_df["label"].astype(int).values
    y_hold_prob_raw = final_model.predict(hold_seq, verbose=0).ravel()
    y_hold_prob_cal = calibrator.predict(y_hold_prob_raw)

    hold_metrics = evaluate_predictions(y_hold, y_hold_prob_cal, threshold)
    hold_metrics["sensitivity_ci_95"] = wilson_ci(hold_metrics["tp"], hold_metrics["tp"] + hold_metrics["fn"])
    hold_metrics["specificity_ci_95"] = wilson_ci(hold_metrics["tn"], hold_metrics["tn"] + hold_metrics["fp"])
    hold_metrics["ppv_ci_95"] = wilson_ci(hold_metrics["tp"], hold_metrics["tp"] + hold_metrics["fp"])
    hold_metrics["npv_ci_95"] = wilson_ci(hold_metrics["tn"], hold_metrics["tn"] + hold_metrics["fn"])

    print("HOLDOUT metrics:", hold_metrics)

    plot_performance_panels(
        y_true=y_hold,
        y_prob_raw=y_hold_prob_raw,
        y_prob_cal=y_hold_prob_cal,
        threshold=threshold,
        title_prefix="Fundus model (hold-out)",
        out_dir=out_dir / "figures",
        prefix="fundus_holdout",
    )

    # Save artifacts
    final_model.save(out_dir / "models" / "fundus_final_model.keras")
    eye_encoder.save(out_dir / "models" / "fundus_eye_encoder.keras")
    joblib.dump(calibrator, out_dir / "models" / "fundus_isotonic_calibrator.joblib")
    joblib.dump({"threshold": threshold, "config": asdict(cfg)}, out_dir / "models" / "fundus_threshold_and_config.joblib")

    # XAI
    print("\nGenerating XAI examples ...")
    sample_n = min(cfg.xai_samples, len(holdout_df))
    sample_idx = np.random.choice(len(holdout_df), size=sample_n, replace=False)

    xai_rows = []
    for i, idx in enumerate(sample_idx):
        row = holdout_df.iloc[int(idx)]
        od = load_image(row.get("od_path"), cfg.image_size)
        os_img = load_image(row.get("os_path"), cfg.image_size)

        pred = float(final_model.predict({"od_input": od[None, ...], "os_input": os_img[None, ...]}, verbose=0).ravel()[0])

        try:
            od_cam = gradcam_for_eye(eye_encoder, od)
            save_gradcam_panel(od, od_cam, out_dir / "xai" / f"participant_{i+1:02d}_{row['participant_id']}_OD_gradcam.png")
        except Exception as e:
            print(f"OD Grad-CAM failed for {row['participant_id']}: {e}")

        try:
            os_cam = gradcam_for_eye(eye_encoder, os_img)
            save_gradcam_panel(os_img, os_cam, out_dir / "xai" / f"participant_{i+1:02d}_{row['participant_id']}_OS_gradcam.png")
        except Exception as e:
            print(f"OS Grad-CAM failed for {row['participant_id']}: {e}")

        try:
            vessel_mask = extract_vessel_mask(od, vessel_threshold=cfg.vessel_threshold)
            vessel_aware_lime_explain(
                final_model,
                od,
                vessel_mask,
                str(out_dir / "xai" / f"participant_{i+1:02d}_{row['participant_id']}_OD_lime.png"),
                num_samples=cfg.lime_samples,
            )
        except Exception as e:
            print(f"OD LIME failed for {row['participant_id']}: {e}")

        xai_rows.append(
            {
                "participant_id": row["participant_id"],
                "label": int(row["label"]),
                "pred_prob": pred,
                "od_path": row.get("od_path"),
                "os_path": row.get("os_path"),
            }
        )

    if xai_rows:
        pd.DataFrame(xai_rows).to_csv(out_dir / "xai_predictions.csv", index=False)

    results = {
        "config": asdict(cfg),
        "dataset": {
            "n_participants_total": int(len(df_part)),
            "n_train": int(len(train_df)),
            "n_holdout": int(len(holdout_df)),
            "train_prevalence": float(train_df["label"].mean()),
            "holdout_prevalence": float(holdout_df["label"].mean()),
        },
        "oof_metrics": oof_metrics,
        "holdout_metrics": hold_metrics,
        "fold_best_epochs": fold_best_epochs,
        "fold_val_ap": fold_val_ap,
        "threshold": float(threshold),
    }
    write_json(out_dir / "results.json", results)

    print(f"\nDone. Outputs saved to: {out_dir.resolve()}")
    return results


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Fundus-only Parkinson's disease pipeline")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs_fundus")
    parser.add_argument("--metadata_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout_split", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--no_mixed_precision", action="store_true")
    args = parser.parse_args()

    return Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        metadata_path=args.metadata_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        random_seed=args.seed,
        holdout_split=args.holdout_split,
        folds=args.folds,
        mixed_precision=not args.no_mixed_precision,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_pipeline(cfg)
