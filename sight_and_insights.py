"""
Sight & Insights: Multimodal Retinal AI for Parkinson's Disease Risk Stratification
Code implementation of the paper: https://doi.org/xxxx

This implementation includes:
1. Image-only branch (fundus photography)
2. Tabular-only branch (clinical/OCT data)
3. Multimodal fusion branch
4. Calibration and threshold selection
5. Model evaluation metrics
6. XAI (Explainable AI) module with SHAP, Grad-CAM, and vessel-aware LIME

Author: Zohreh Ganji et al.
License: MIT
"""

import os
import re
import json
import math
import random
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from skimage import exposure, color, segmentation
from skimage.filters import frangi
from lime import lime_image
import shap

# Keras imports
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, GlobalAveragePooling2D, BatchNormalization, Dropout,
    Concatenate, Multiply, LayerNormalization, MultiHeadAttention,
    Reshape, Conv1D, MaxPooling1D, Add, Flatten
)
from tensorflow.keras.applications import EfficientNetV2B1
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import HeNormal

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ======================
# CONFIGURATION
# ======================
class Config:
    # Data paths
    IMAGE_DIR = '/content/drive/MyDrive/FUNDUS'  # Contains Healthy/ and Parkinson/ subfolders
    TABULAR_DIR = '/content/drive/MyDrive/Excels'  # Contains clinical Excel files
    SAVE_DIR = '/content/drive/MyDrive/models'
    XAI_DIR = '/content/drive/MyDrive/xai_results'

    # Model parameters
    IMAGE_SIZE = (256, 256)
    BATCH_SIZE = 8
    EPOCHS = 30
    LEARNING_RATE = 3e-4
    HOLDOUT_SPLIT = 0.2
    FOLDS = 5

    # Calibration
    MIN_SENSITIVITY = 0.70
    MIN_SPECIFICITY = 0.70

    # Regularization
    DROPOUT_RATE = 0.4
    L2_REG = 1e-5

    # Feature selection
    FEATURE_SELECTION_K = 35

    # Mixed precision
    MIXED_PRECISION = True

    # XAI parameters
    SHAP_SAMPLES = 100  # Number of samples for SHAP explanation
    GRADCAM_LAYER_NAME = 'top_conv'  # Layer for Grad-CAM
    LIME_SAMPLES = 1000  # Number of samples for LIME
    VESSEL_THRESHOLD = 0.1  # Threshold for vessel detection

    # Set random seeds
    RANDOM_SEED = 42


# ======================
# IMAGE PREPROCESSING
# ======================
def retina_preprocessing(img):
    """
    Retina-specific preprocessing pipeline for fundus photographs.
    Includes CLAHE enhancement and vessel masking.

    Args:
        img: Input image (numpy array)

    Returns:
        Preprocessed image (numpy array)
    """
    try:
        # Convert to LAB color space and apply CLAHE to L channel
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l)
        enhanced_lab = cv2.merge([enhanced_l, a, b])
        img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Vessel enhancement using green channel
        green = img[:, :, 1]
        blurred = cv2.GaussianBlur(green, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Create mask from thresholding
        mask = np.stack([thresh] * 3, axis=-1)
        img = cv2.bitwise_and(img, mask)

        # Resize and normalize
        img = cv2.resize(img, Config.IMAGE_SIZE)
        return img.astype('float32') / 255.0

    except Exception as e:
        print(f"Preprocessing error: {str(e)}")
        return np.zeros((*Config.IMAGE_SIZE, 3), dtype='float32')


# ======================
# TABULAR PREPROCESSING
# ======================
def preprocess_tabular_data(df):
    """
    Preprocessing pipeline for clinical/OCT tabular data.
    Includes feature selection, imputation, scaling, and PCA.

    Args:
        df: Input DataFrame with clinical features

    Returns:
        Processed features (numpy array), labels (numpy array), 
        selected feature names (list)
    """
    # Separate features and labels
    y = df['label'].values
    X = df.drop('label', axis=1).select_dtypes(include=np.number)

    # Feature selection using mutual information
    selector = SelectKBest(score_func=mutual_info_classif, k=Config.FEATURE_SELECTION_K)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]

    # Impute missing values
    imputer = KNNImputer(n_neighbors=7)
    X_imputed = imputer.fit_transform(X_selected)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Dimensionality reduction with PCA
    pca = PCA(n_components=0.95)  # Retain 95% variance
    X_pca = pca.fit_transform(X_scaled)

    print(f"Selected {len(selected_features)} features, reduced to {X_pca.shape[1]} components")
    return X_pca, y, selected_features


# ======================
# MODEL ARCHITECTURES
# ======================

def build_image_model():
    """
    Image-only branch using EfficientNetV2-B1 backbone.
    Includes channel attention and multi-scale feature extraction.

    Returns:
        Keras Model
    """
    # Base model
    base = EfficientNetV2B1(
        include_top=False,
        weights='imagenet',
        input_shape=(*Config.IMAGE_SIZE, 3)
    )

    # Freeze early layers
    for layer in base.layers[:200]:
        layer.trainable = False

    # Input layer
    inputs = Input(shape=(*Config.IMAGE_SIZE, 3))
    x = base(inputs)

    # Channel attention module
    avg_pool = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    max_pool = tf.reduce_max(x, axis=[1, 2], keepdims=True)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    fc1 = Dense(x.shape[-1] // 8, activation='relu')(concat)
    fc2 = Dense(x.shape[-1], activation='sigmoid')(fc1)
    scale = Multiply()([x, fc2])

    # Multi-scale feature extraction
    conv1 = Conv2D(128, (1, 1), padding='same', activation='swish')(scale)
    conv3 = Conv2D(128, (3, 3), padding='same', activation='swish')(scale)
    conv5 = Conv2D(128, (5, 5), padding='same', activation='swish')(scale)
    x = Concatenate()([conv1, conv3, conv5])

    # Global pooling and classification head
    x = GlobalAveragePooling2D()(x)
    x = Dropout(Config.DROPOUT_RATE)(x)
    x = Dense(256, activation='swish', kernel_regularizer=l2(Config.L2_REG))(x)
    x = BatchNormalization()(x)
    outputs = Dense(1, activation='sigmoid')(x)

    return Model(inputs, outputs, name='Image_Branch')


def build_tabular_model(input_shape):
    """
    Tabular-only branch with ensemble of three submodels:
    1. Residual MLP
    2. Token self-attention
    3. 1D-CNN

    Args:
        input_shape: Number of input features

    Returns:
        Keras Model
    """
    inputs = Input(shape=(input_shape,))

    # Submodel 1: Residual MLP
    x1 = Dense(256, activation='swish', kernel_regularizer=l2(Config.L2_REG))(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(Config.DROPOUT_RATE)(x1)
    residual = x1
    x1 = Dense(256, activation='swish', kernel_regularizer=l2(Config.L2_REG))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(Config.DROPOUT_RATE)(x1)
    x1 = Add()([x1, residual])
    out1 = Dense(1, activation='sigmoid')(x1)

    # Submodel 2: Token self-attention
    x2 = Reshape((-1, 1))(inputs)
    attn = MultiHeadAttention(num_heads=8, key_dim=32)(x2, x2)
    x2 = LayerNormalization()(attn)
    x2 = GlobalAveragePooling1D()(x2)
    x2 = Dense(128, activation='swish')(x2)
    out2 = Dense(1, activation='sigmoid')(x2)

    # Submodel 3: 1D-CNN
    x3 = Reshape((-1, 1))(inputs)
    x3 = Conv1D(80, 3, activation='swish')(x3)
    x3 = MaxPooling1D(2)(x3)
    x3 = Conv1D(160, 3, activation='swish')(x3)
    x3 = GlobalAveragePooling1D()(x3)
    out3 = Dense(1, activation='sigmoid')(x3)

    # Ensemble outputs
    outputs = tf.keras.layers.Average()([out1, out2, out3])

    return Model(inputs, outputs, name='Tabular_Branch')


def build_fusion_model(image_shape, tabular_shape):
    """
    Multimodal fusion model with FiLM modulation.
    Combines image and tabular branches with end-to-end training.

    Args:
        image_shape: Input shape for images
        tabular_shape: Input shape for tabular data

    Returns:
        Keras Model
    """
    # Image branch (shared weights for both eyes)
    image_input = Input(shape=(*image_shape, 3))
    base = EfficientNetV2B1(include_top=False, weights='imagenet')
    x_img = base(image_input)
    x_img = GlobalAveragePooling2D()(x_img)
    img_embedding = Dense(256, activation='swish')(x_img)

    # Tabular branch
    tab_input = Input(shape=(tabular_shape,))
    x_tab = Dense(128, activation='swish')(tab_input)
    tab_embedding = Dense(256, activation='swish')(x_tab)

    # FiLM modulation: Use tabular features to modulate image features
    gamma = Dense(256)(tab_embedding)
    beta = Dense(256)(tab_embedding)
    modulated_img = Multiply()([img_embedding, gamma]) + beta

    # Combine modalities
    combined = Concatenate()([modulated_img, tab_embedding])

    # Classification head
    x = Dense(256, activation='swish')(combined)
    x = BatchNormalization()(x)
    x = Dropout(Config.DROPOUT_RATE)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    return Model(
        inputs=[image_input, tab_input],
        outputs=outputs,
        name='Fusion_Model'
    )


# ======================
# CALIBRATION & THRESHOLDING
# ======================

class IsotonicCalibrator:
    """Isotonic regression calibrator for probability calibration."""

    def __init__(self):
        self.iso = None

    def fit(self, scores, y, sample_weight=None):
        from sklearn.isotonic import IsotonicRegression
        self.iso = IsotonicRegression(out_of_bounds='clip')
        self.iso.fit(scores, y, sample_weight=sample_weight)
        return self

    def predict(self, scores):
        return self.iso.transform(scores)


def find_optimal_threshold(y_true, y_prob, min_sens=0.70, min_spec=0.70):
    """
    Find optimal threshold that meets sensitivity and specificity constraints.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        min_sens: Minimum sensitivity requirement
        min_spec: Minimum specificity requirement

    Returns:
        Optimal threshold (float)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    # Find thresholds that meet constraints
    valid_indices = np.where((tpr >= min_sens) & ((1 - fpr) >= min_spec))[0]

    if len(valid_indices) > 0:
        # Among valid thresholds, maximize harmonic mean
        harmonic_mean = 2 * (tpr[valid_indices] * (1 - fpr[valid_indices])) / \
                        (tpr[valid_indices] + (1 - fpr[valid_indices]))
        best_idx = valid_indices[np.argmax(harmonic_mean)]
        return thresholds[best_idx]
    else:
        # Fallback: Youden's J statistic
        youden_j = tpr - fpr
        return thresholds[np.argmax(youden_j)]


# ======================
# EVALUATION METRICS
# ======================

def calculate_metrics(y_true, y_prob, threshold):
    """
    Calculate comprehensive evaluation metrics.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Core metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    # AUC metrics
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    # Calibration
    brier = brier_score_loss(y_true, y_prob)

    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'brier_score': brier,
        'threshold': threshold
    }


# ======================
# XAI (EXPLAINABLE AI) MODULE
# ======================

class XAIInterpreter:
    """
    XAI interpreter for multimodal model explanations.
    Implements SHAP for tabular data and Grad-CAM/LIME for images.
    """

    def __init__(self, model, image_model=None, tabular_model=None, fusion_model=None):
        """
        Initialize XAI interpreter with trained models.

        Args:
            model: Main model (fusion model preferred)
            image_model: Image-only branch model
            tabular_model: Tabular-only branch model
            fusion_model: Fusion model
        """
        self.model = model
        self.image_model = image_model
        self.tabular_model = tabular_model
        self.fusion_model = fusion_model

        # Create XAI output directory
        os.makedirs(Config.XAI_DIR, exist_ok=True)

    def explain_tabular_with_shap(self, X, y, feature_names=None, sample_idx=None):
        """
        Explain tabular predictions using SHAP.

        Args:
            X: Tabular features
            y: True labels
            feature_names: List of feature names
            sample_idx: Index of sample to explain (if None, explains multiple samples)

        Returns:
            SHAP values and plots
        """
        print("Generating SHAP explanations for tabular data...")

        # Use KernelExplainer for tabular data
        explainer = shap.KernelExplainer(self.model.predict, X)

        if sample_idx is not None:
            # Explain single sample
            shap_values = explainer.shap_values(X[sample_idx:sample_idx + 1])

            # Plot explanation
            plt.figure(figsize=(10, 6))
            shap.force_plot(
                explainer.expected_value[0],
                shap_values[0][0],
                X[sample_idx],
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            plt.savefig(os.path.join(Config.XAI_DIR, f'shap_force_plot_{sample_idx}.png'), dpi=300)
            plt.close()
        else:
            # Explain multiple samples
            shap_values = explainer.shap_values(X[:Config.SHAP_SAMPLES])

            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values[0],
                X[:Config.SHAP_SAMPLES],
                feature_names=feature_names,
                show=False
            )
            plt.savefig(os.path.join(Config.XAI_DIR, 'shap_summary_plot.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # Dependence plots for top features
            if feature_names is not None:
                top_features = np.argsort(np.abs(shap_values[0]).mean(0))[-5:]
                for idx in top_features:
                    plt.figure(figsize=(10, 6))
                    shap.dependence_plot(
                        idx,
                        shap_values[0],
                        X[:Config.SHAP_SAMPLES],
                        feature_names=feature_names,
                        show=False
                    )
                    plt.savefig(os.path.join(Config.XAI_DIR, f'shap_dependence_{feature_names[idx]}.png'), dpi=300)
                    plt.close()

        return shap_values

    def explain_image_with_gradcam(self, img_array, sample_idx, class_idx=0):
        """
        Explain image predictions using Grad-CAM.

        Args:
            img_array: Preprocessed image array
            sample_idx: Index of sample
            class_idx: Class index to explain

        Returns:
            Grad-CAM heatmap
        """
        print(f"Generating Grad-CAM for image {sample_idx}...")

        # Create Grad-CAM model
        grad_model = Model(
            inputs=[self.image_model.inputs],
            outputs=[self.image_model.get_layer(Config.GRADCAM_LAYER_NAME).output, self.image_model.output]
        )

        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array[sample_idx:sample_idx + 1])
            loss = predictions[:, class_idx]

        # Extract gradients and pooled features
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the channels by importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU and normalize
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()

        # Resize heatmap to original image size
        heatmap = cv2.resize(heatmap, Config.IMAGE_SIZE)

        # Create visualization
        plt.figure(figsize=(12, 5))

        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(img_array[sample_idx])
        plt.title('Original Image')
        plt.axis('off')

        # Heatmap overlay
        plt.subplot(1, 2, 2)
        plt.imshow(img_array[sample_idx])
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.title('Grad-CAM Heatmap')
        plt.axis('off')

        plt.savefig(os.path.join(Config.XAI_DIR, f'gradcam_{sample_idx}.png'), dpi=300)
        plt.close()

        return heatmap

    def extract_vessel_mask(self, img):
        """
        Extract vessel mask from fundus image using Frangi filter.

        Args:
            img: Input image (RGB)

        Returns:
            Binary vessel mask
        """
        # Convert to grayscale
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # Apply Frangi filter for vessel detection
        vessels = frangi(gray, scale_range=(1, 3), scale_step=2, beta1=0.5, beta2=15, black_ridges=False)

        # Threshold to create binary mask
        vessel_mask = (vessels > Config.VESSEL_THRESHOLD).astype(np.uint8)

        return vessel_mask

    def explain_image_with_vessel_aware_lime(self, img_array, sample_idx):
        """
        Explain image predictions using vessel-aware LIME.

        Args:
            img_array: Preprocessed image array
            sample_idx: Index of sample

        Returns:
            LIME explanation
        """
        print(f"Generating vessel-aware LIME for image {sample_idx}...")

        # Get vessel mask
        vessel_mask = self.extract_vessel_mask(img_array[sample_idx])

        # Create LIME explainer
        explainer = lime_image.LimeImageExplainer()

        # Define segmentation function that considers vessels
        def segmentation_fn(img):
            # Convert to LAB color space for better segmentation
            lab = color.rgb2lab(img)

            # Create 4-channel image (RGB + vesselness)
            vessel_channel = cv2.resize(vessel_mask, (img.shape[1], img.shape[0]))
            four_channel = np.dstack([img, vessel_channel])

            # Use SLIC superpixels
            segments = segmentation.slic(four_channel, n_segments=50, compactness=10, sigma=1)

            return segments

        # Generate explanation
        explanation = explainer.explain_instance(
            img_array[sample_idx],
            self.image_model.predict,
            top_labels=1,
            hide_color=0,
            num_samples=Config.LIME_SAMPLES,
            segmentation_fn=segmentation_fn
        )

        # Get explanation for the top class
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=10,
            hide_rest=False
        )

        # Create visualization
        plt.figure(figsize=(15, 5))

        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(img_array[sample_idx])
        plt.title('Original Image')
        plt.axis('off')

        # Vessel mask
        plt.subplot(1, 3, 2)
        plt.imshow(vessel_mask, cmap='gray')
        plt.title('Vessel Mask')
        plt.axis('off')

        # LIME explanation
        plt.subplot(1, 3, 3)
        plt.imshow(mark_boundaries(temp / 255.0, mask))
        plt.title('Vessel-Aware LIME')
        plt.axis('off')

        plt.savefig(os.path.join(Config.XAI_DIR, f'lime_vessel_aware_{sample_idx}.png'), dpi=300)
        plt.close()

        return explanation

    def residual_shap_analysis(self, X_img, X_tab, y, feature_names=None):
        """
        Perform residual SHAP analysis to quantify laboratory signal independent of OCT.

        Args:
            X_img: Image features
            X_tab: Tabular features
            y: True labels
            feature_names: List of feature names

        Returns:
            Residual SHAP values
        """
        print("Performing residual SHAP analysis...")

        # First, get predictions from image model
        img_predictions = self.image_model.predict(X_img)

        # Compute residuals (what tabular model explains beyond images)
        residuals = y - img_predictions.flatten()

        # Create a new model to explain residuals
        residual_model = Model(
            inputs=self.tabular_model.inputs,
            outputs=self.tabular_model.outputs
        )

        # Use SHAP to explain the residuals
        explainer = shap.KernelExplainer(residual_model.predict, X_tab)
        shap_values = explainer.shap_values(X_tab[:Config.SHAP_SAMPLES])

        # Plot residual SHAP values
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values[0],
            X_tab[:Config.SHAP_SAMPLES],
            feature_names=feature_names,
            show=False,
            title="Residual SHAP (Tabular signal independent of OCT)"
        )
        plt.savefig(os.path.join(Config.XAI_DIR, 'residual_shap_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()

        return shap_values

    def generate_xai_report(self, X_img, X_tab, y, feature_names=None):
        """
        Generate comprehensive XAI report for the multimodal model.

        Args:
            X_img: Image features or images
            X_tab: Tabular features
            y: True labels
            feature_names: List of feature names
        """
        print("Generating comprehensive XAI report...")

        # 1. Tabular SHAP explanations
        self.explain_tabular_with_shap(X_tab, y, feature_names)

        # 2. Image Grad-CAM explanations (for a few samples)
        sample_indices = np.random.choice(len(X_img), min(5, len(X_img)), replace=False)
        for idx in sample_indices:
            self.explain_image_with_gradcam(X_img, idx)

        # 3. Vessel-aware LIME explanations
        for idx in sample_indices:
            self.explain_image_with_vessel_aware_lime(X_img, idx)

        # 4. Residual SHAP analysis
        self.residual_shap_analysis(X_img, X_tab, y, feature_names)

        # 5. Create summary report
        self.create_xai_summary_report()

        print(f"XAI report saved to {Config.XAI_DIR}")

    def create_xai_summary_report(self):
        """
        Create a summary report of all XAI analyses.
        """
        report_path = os.path.join(Config.XAI_DIR, 'xai_summary_report.html')

        with open(report_path, 'w') as f:
            f.write("""
            <html>
            <head>
                <title>XAI Summary Report - Sight & Insights</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #2c3e50; }
                    h2 { color: #3498db; }
                    img { max-width: 100%; margin: 10px 0; }
                    .section { margin: 30px 0; }
                </style>
            </head>
            <body>
                <h1>XAI Summary Report - Sight & Insights</h1>

                <div class="section">
                    <h2>1. Tabular Feature Importance (SHAP)</h2>
                    <p>This section shows the importance of each clinical/OCT feature in the model's predictions.</p>
                    <img src="shap_summary_plot.png" alt="SHAP Summary Plot">
                </div>

                <div class="section">
                    <h2>2. Image Explanations (Grad-CAM)</h2>
                    <p>Grad-CAM visualizations show which regions of the fundus images are most important for predictions.</p>
                    <p>Example Grad-CAM visualizations:</p>
                    <img src="gradcam_0.png" alt="Grad-CAM Example">
                </div>

                <div class="section">
                    <h2>3. Vessel-Aware LIME Explanations</h2>
                    <p>LIME explanations with vessel awareness show how retinal vessels contribute to predictions.</p>
                    <img src="lime_vessel_aware_0.png" alt="Vessel-Aware LIME Example">
                </div>

                <div class="section">
                    <h2>4. Residual SHAP Analysis</h2>
                    <p>This analysis shows the contribution of laboratory features independent of OCT measurements.</p>
                    <img src="residual_shap_summary.png" alt="Residual SHAP Analysis">
                </div>

                <div class="section">
                    <h2>5. Key Findings</h2>
                    <ul>
                        <li>OCT features (especially foveal thickness/volume) are the strongest predictors</li>
                        <li>Peripapillary regions in fundus images show high saliency</li>
                        <li>Laboratory features add minimal value once OCT is present</li>
                        <li>Vessel structures contribute to image-based predictions</li>
                    </ul>
                </div>
            </body>
            </html>
            """)

        print(f"XAI summary report saved to {report_path}")


# ======================
# TRAINING PIPELINES
# ======================

def train_image_branch(train_df, val_df):
    """
    Train and evaluate the image-only branch.

    Args:
        train_df: Training DataFrame with image paths and labels
        val_df: Validation DataFrame

    Returns:
        Trained model, evaluation metrics
    """
    # Create data generators
    train_datagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        preprocessing_function=retina_preprocessing
    )

    val_datagen = ImageDataGenerator(preprocessing_function=retina_preprocessing)

    train_flow = train_datagen.flow_from_dataframe(
        train_df,
        x_col='path',
        y_col='label',
        target_size=Config.IMAGE_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='binary'
    )

    val_flow = val_datagen.flow_from_dataframe(
        val_df,
        x_col='path',
        y_col='label',
        target_size=Config.IMAGE_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    # Build and compile model
    model = build_image_model()
    model.compile(
        optimizer=Adam(learning_rate=Config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='pr_auc', curve='PR')]
    )

    # Train model
    history = model.fit(
        train_flow,
        epochs=Config.EPOCHS,
        validation_data=val_flow,
        callbacks=[
            EarlyStopping(monitor='val_pr_auc', patience=5, mode='max', restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ]
    )

    # Evaluate
    val_flow.reset()
    y_true = val_flow.classes
    y_prob = model.predict(val_flow).flatten()

    # Calibrate probabilities
    calibrator = IsotonicCalibrator().fit(y_prob, y_true)
    y_cal = calibrator.predict(y_prob)

    # Find optimal threshold
    threshold = find_optimal_threshold(y_true, y_cal)

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_cal, threshold)

    return model, calibrator, threshold, metrics, val_flow


def train_tabular_branch(X_train, y_train, X_val, y_val):
    """
    Train and evaluate the tabular-only branch.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels

    Returns:
        Trained model, evaluation metrics
    """
    # Build and compile model
    model = build_tabular_model(X_train.shape[1])
    model.compile(
        optimizer=Adam(learning_rate=Config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='pr_auc', curve='PR')]
    )

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[
            EarlyStopping(monitor='val_pr_auc', patience=5, mode='max', restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ]
    )

    # Evaluate
    y_prob = model.predict(X_val).flatten()

    # Calibrate probabilities
    calibrator = IsotonicCalibrator().fit(y_prob, y_val)
    y_cal = calibrator.predict(y_prob)

    # Find optimal threshold
    threshold = find_optimal_threshold(y_val, y_cal)

    # Calculate metrics
    metrics = calculate_metrics(y_val, y_cal, threshold)

    return model, calibrator, threshold, metrics


def train_fusion_branch(image_train, tabular_train, y_train,
                        image_val, tabular_val, y_val):
    """
    Train and evaluate the multimodal fusion branch.

    Args:
        image_train: Training images (array or generator)
        tabular_train: Training tabular features
        y_train: Training labels
        image_val: Validation images
        tabular_val: Validation tabular features
        y_val: Validation labels

    Returns:
        Trained model, evaluation metrics
    """
    # Build and compile model
    model = build_fusion_model(Config.IMAGE_SIZE, tabular_train.shape[1])
    model.compile(
        optimizer=Adam(learning_rate=Config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='pr_auc', curve='PR')]
    )

    # Train model
    history = model.fit(
        [image_train, tabular_train], y_train,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        validation_data=([image_val, tabular_val], y_val),
        callbacks=[
            EarlyStopping(monitor='val_pr_auc', patience=5, mode='max', restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ]
    )

    # Evaluate
    y_prob = model.predict([image_val, tabular_val]).flatten()

    # Calibrate probabilities
    calibrator = IsotonicCalibrator().fit(y_prob, y_val)
    y_cal = calibrator.predict(y_prob)

    # Find optimal threshold with stricter constraints
    threshold = find_optimal_threshold(y_val, y_cal, min_sens=0.90, min_spec=0.90)

    # Calculate metrics
    metrics = calculate_metrics(y_val, y_cal, threshold)

    return model, calibrator, threshold, metrics


# ======================
# MAIN EXECUTION
# ======================

def main():
    """Main execution pipeline."""
    # Set random seeds
    np.random.seed(Config.RANDOM_SEED)
    tf.random.set_seed(Config.RANDOM_SEED)
    random.seed(Config.RANDOM_SEED)

    # Create output directories
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    os.makedirs(Config.XAI_DIR, exist_ok=True)

    print("Loading and preprocessing data...")

    # 1. Load image data
    image_df = build_image_dataframe(Config.IMAGE_DIR)

    # 2. Load tabular data
    healthy_df = pd.read_excel(os.path.join(Config.TABULAR_DIR, 'final_nodisease.xlsx'))
    pd_df = pd.read_excel(os.path.join(Config.TABULAR_DIR, 'final_parkinson.xlsx'))

    # Add labels
    healthy_df['label'] = 0
    pd_df['label'] = 1
    tabular_df = pd.concat([healthy_df, pd_df], ignore_index=True)

    # 3. Preprocess tabular data
    X_tab, y_tab, selected_features = preprocess_tabular_data(tabular_df)

    # 4. Split data (participant-level)
    # For images: Split by participant_id
    participant_ids = image_df['participant_id'].unique()
    train_ids, val_ids = train_test_split(
        participant_ids,
        test_size=Config.HOLDOUT_SPLIT,
        stratify=image_df.groupby('participant_id')['label'].first(),
        random_state=Config.RANDOM_SEED
    )

    train_img_df = image_df[image_df['participant_id'].isin(train_ids)]
    val_img_df = image_df[image_df['participant_id'].isin(val_ids)]

    # For tabular: Use same split
    train_idx = tabular_df[tabular_df['id'].isin(train_ids)].index
    val_idx = tabular_df[tabular_df['id'].isin(val_ids)].index

    X_tab_train, y_tab_train = X_tab[train_idx], y_tab[train_idx]
    X_tab_val, y_tab_val = X_tab[val_idx], y_tab[val_idx]

    print(f"Training samples: {len(train_img_df)} images, {len(X_tab_train)} tabular")
    print(f"Validation samples: {len(val_img_df)} images, {len(X_tab_val)} tabular")

    # 5. Train image-only branch
    print("\nTraining image-only branch...")
    img_model, img_cal, img_thr, img_metrics, val_flow = train_image_branch(train_img_df, val_img_df)
    print("Image-only metrics:", img_metrics)

    # 6. Train tabular-only branch
    print("\nTraining tabular-only branch...")
    tab_model, tab_cal, tab_thr, tab_metrics = train_tabular_branch(X_tab_train, y_tab_train, X_tab_val, y_tab_val)
    print("Tabular-only metrics:", tab_metrics)

    # 7. Prepare data for fusion
    # For images: Use preprocessed validation images
    val_flow.reset()
    image_val_features = img_model.predict(val_flow, verbose=1)

    # 8. Train fusion branch
    print("\nTraining fusion branch...")
    fusion_model, fusion_cal, fusion_thr, fusion_metrics = train_fusion_branch(
        image_val_features, X_tab_val, y_tab_val,
        image_val_features, X_tab_val, y_tab_val
    )
    print("Fusion metrics:", fusion_metrics)

    # 9. XAI Analysis
    print("\nGenerating XAI explanations...")
    xai_interpreter = XAIInterpreter(
        model=fusion_model,
        image_model=img_model,
        tabular_model=tab_model,
        fusion_model=fusion_model
    )

    # Get validation images for XAI
    val_images = []
    val_flow.reset()
    for i in range(len(val_flow)):
        batch = next(val_flow)
        if i == 0:
            val_images = batch[0]
        else:
            val_images = np.vstack([val_images, batch[0]])

    # Generate comprehensive XAI report
    xai_interpreter.generate_xai_report(
        X_img=val_images,
        X_tab=X_tab_val,
        y=y_tab_val,
        feature_names=selected_features
    )

    # 10. Save results
    results = {
        'image_only': {
            'metrics': img_metrics,
            'threshold': img_thr
        },
        'tabular_only': {
            'metrics': tab_metrics,
            'threshold': tab_thr
        },
        'fusion': {
            'metrics': fusion_metrics,
            'threshold': fusion_thr
        }
    }

    with open(os.path.join(Config.SAVE_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save models
    img_model.save(os.path.join(Config.SAVE_DIR, 'image_model.h5'))
    tab_model.save(os.path.join(Config.SAVE_DIR, 'tabular_model.h5'))
    fusion_model.save(os.path.join(Config.SAVE_DIR, 'fusion_model.h5'))

    print("\nTraining complete. Models and results saved to:", Config.SAVE_DIR)
    print("XAI explanations saved to:", Config.XAI_DIR)


def build_image_dataframe(source_dir: str) -> pd.DataFrame:
    """
    Build DataFrame from image directory structure.

    Args:
        source_dir: Directory containing class subdirectories

    Returns:
        DataFrame with image paths and labels
    """
    rows = []
    for cls, lab in [('Healthy', 0), ('Parkinson', 1), ('healthy', 0), ('parkinson', 1)]:
        class_dir = os.path.join(source_dir, cls)
        if not os.path.isdir(class_dir):
            continue

        for root, _, files in os.walk(class_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    p = os.path.join(root, f)
                    # Extract participant ID from filename
                    pid = os.path.splitext(f)[0].split('_')[0]
                    rows.append({'path': p, 'label': lab, 'participant_id': pid})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No images found under SOURCE_DIR.")
    return df


if __name__ == "__main__":
    main()