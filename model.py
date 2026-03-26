import tensorflow as tf
from tensorflow.keras import layers, models, backend as K


def dice_coef(y_true, y_pred, smooth=1.0):
    """
    Dice coefficient for semantic segmentation.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    """
    Dice loss (1 - Dice coefficient).
    """
    return 1.0 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    """
    Combined Binary Cross-Entropy + Dice Loss.

    BCE provides stable per-pixel gradients even when the mask is very sparse
    (prevents the model from collapsing to all-zeros).
    Dice loss directly optimises overlap.
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = K.mean(bce)
    return bce + dice_loss(y_true, y_pred)


def conv_block(x, filters, dropout_rate=0.1):
    """
    Two Conv2D layers with BatchNormalization and ReLU activation.
    """
    x = layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    return x


def unet_model(input_size=(256, 256, 3)):
    """
    U-Net with BatchNormalization and 4 encoder levels.
    Uses combined BCE + Dice loss for class-imbalanced segmentation.
    """
    inputs = layers.Input(input_size)

    # ── Encoder ──────────────────────────────────────────────────────────
    c1 = conv_block(inputs, 32, dropout_rate=0.1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 64, dropout_rate=0.1)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 128, dropout_rate=0.2)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 256, dropout_rate=0.2)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # ── Bridge ───────────────────────────────────────────────────────────
    c5 = conv_block(p4, 512, dropout_rate=0.3)

    # ── Decoder ──────────────────────────────────────────────────────────
    u6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = conv_block(u6, 256, dropout_rate=0.2)

    u7 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = conv_block(u7, 128, dropout_rate=0.2)

    u8 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = conv_block(u8, 64, dropout_rate=0.1)

    u9 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = conv_block(u9, 32, dropout_rate=0.1)

    # ── Output ───────────────────────────────────────────────────────────
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=bce_dice_loss,
        metrics=[dice_coef, 'accuracy']
    )

    return model


if __name__ == "__main__":
    model = unet_model()
    model.summary()
