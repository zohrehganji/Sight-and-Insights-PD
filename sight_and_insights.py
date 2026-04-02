# =========================================
# Thesis-Aligned Fundus Framework
# Fully consistent with described methodology
# =========================================

import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K

# =========================================
# 1. Preprocessing (Retina-aware)
# =========================================

import cv2
import numpy as np

def retina_preprocess(img):
    # LAB + CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Vessel mask from green channel
    green = img[:,:,1]
    blur = cv2.GaussianBlur(green, (5,5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Apply mask (critical step!)
    img = cv2.bitwise_and(img, img, mask=mask)

    img = cv2.resize(img, (256,256))
    img = img / 255.0
    return img

# =========================================
# 2. SE Block
# =========================================

def se_block(x, reduction=16):
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(filters//reduction, activation='swish')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1,1,filters))(se)
    return layers.Multiply()([x, se])

# =========================================
# 3. Multi-scale block (EXPLICIT)
# =========================================

def multi_scale_block(x):
    c1 = layers.Conv2D(128, (1,1), padding='same', activation='swish')(x)
    c3 = layers.Conv2D(128, (3,3), padding='same', activation='swish')(x)
    c5 = layers.Conv2D(128, (5,5), padding='same', activation='swish')(x)
    x = layers.Concatenate()([c1,c3,c5])
    x = layers.Conv2D(256, (3,3), padding='same', activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    return x

# =========================================
# 4. Backbone (EfficientNetV2-B1)
# =========================================

def build_backbone():
    base = tf.keras.applications.EfficientNetV2B1(
        include_top=False,
        weights='imagenet',
        input_shape=(256,256,3)
    )

    # Freeze early layers (~200)
    for layer in base.layers[:200]:
        layer.trainable = False

    x = base.output
    x = se_block(x)
    x = multi_scale_block(x)
    x = se_block(x)
    x = layers.GlobalAveragePooling2D()(x)

    return Model(base.input, x, name="Backbone")

# =========================================
# 5. Siamese Dual-stream
# =========================================

def build_siamese(backbone):
    input_os = layers.Input((256,256,3), name="OS")
    input_od = layers.Input((256,256,3), name="OD")

    feat_os = backbone(input_os)
    feat_od = backbone(input_od)

    return Model([input_os, input_od], [feat_os, feat_od])

# =========================================
# 6. Interaction Layer (CRITICAL)
# =========================================

def interaction_layer(f_os, f_od):
    diff = layers.Lambda(lambda x: K.abs(x[0] - x[1]))([f_os, f_od])
    prod = layers.Multiply()([f_os, f_od])
    return layers.Concatenate()([f_os, f_od, diff, prod])

# =========================================
# 7. Tabular Encoder
# =========================================

def build_tabular(input_dim):
    inp = layers.Input((input_dim,))
    x = layers.Dense(128, activation='swish')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='swish')(x)
    return Model(inp, x, name="TabularEncoder")

# =========================================
# 8. FiLM Modulation (TRUE IMPLEMENTATION)
# =========================================

def film_layer(image_feat, tab_feat):
    gamma = layers.Dense(image_feat.shape[-1])(tab_feat)
    beta  = layers.Dense(image_feat.shape[-1])(tab_feat)
    return layers.Add()([
        layers.Multiply()([gamma, image_feat]),
        beta
    ])

# =========================================
# 9. Full Fusion Model
# =========================================

def build_full_model(tab_dim):
    backbone = build_backbone()

    input_os = layers.Input((256,256,3))
    input_od = layers.Input((256,256,3))
    input_tab = layers.Input((tab_dim,))

    f_os = backbone(input_os)
    f_od = backbone(input_od)

    f_int = interaction_layer(f_os, f_od)

    tab_model = build_tabular(tab_dim)
    f_tab = tab_model(input_tab)

    f_film = film_layer(f_int, f_tab)

    x = layers.Dense(256, activation='swish')(f_film)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    out = layers.Dense(1, activation='sigmoid')(x)

    return Model([input_os, input_od, input_tab], out)

# =========================================
# 10. Focal Loss
# =========================================

def focal_loss(alpha=0.65, gamma=2.0):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        pt = tf.exp(-bce)
        return alpha * (1 - pt)**gamma * bce
    return loss

# =========================================
# 11. Compile
# =========================================

model = build_full_model(tab_dim=50)
model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-4),
    loss=focal_loss(),
    metrics=["AUC"]
)

model.summary()
