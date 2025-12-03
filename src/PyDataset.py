import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import RandomZoom, RandomTranslation, RandomRotation, RandomFlip
import logging

# Get the logger instance
logger = logging.getLogger(__name__)

class ImageSequence(Sequence):
    def __init__(self, df, target_size=(224, 224), batch_size=32,
                 preprocess_fn=None, shuffle=True, augment=False, **kwargs):
        super().__init__(**kwargs)

        if df is None or len(df) == 0:
            raise ValueError("DataFrame ist leer oder None.")
        self.df = df.reset_index(drop=True)
        self.target_size = tuple(target_size)
        self.batch_size = int(batch_size)
        self.preprocess_fn = preprocess_fn or (lambda x: x)
        self.shuffle = bool(shuffle)
        self.augment = bool(augment)
        
        # Augmentation-Layer speziell für Röntgenbilder initialisieren
        if self.augment:
            self.geometric_augmentation = tf.keras.Sequential([
                RandomFlip("horizontal"),
                RandomRotation(0.1),  # Rotiert um +/- 10% von 360 Grad (ca. +/- 36 Grad)
                RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='nearest'),
                RandomZoom(height_factor=0.1, fill_mode='nearest'),
            ], name="geometric_augmentation")

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _augment_image(self, image):
        # Wenden Sie zuerst die geometrischen Augmentations-Layer an
        image = self.geometric_augmentation(tf.expand_dims(image, 0), training=True)[0]

        # --- Intensitäts-Augmentationen (wichtig für Röntgenbilder) ---
        if np.random.rand() < 0.7:
            image = tf.image.random_brightness(image, max_delta=0.2) # Etwas reduzierter Delta

        if np.random.rand() < 0.7:
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

        return image

    def _load_image(self, path):
        if not os.path.exists(path):
            logger.warning(f"Datei nicht gefunden: {path}")
            return tf.zeros((*self.target_size, 3), dtype=tf.float32)
        try:
            img = tf.io.read_file(path)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, self.target_size)
            img = tf.cast(img, tf.float32)
            return img
        except Exception as e:
            logger.error(f"Fehler beim Laden von '{path}': {e}", exc_info=True)
            return tf.zeros((*self.target_size, 3), dtype=tf.float32)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indices]

        images, labels = [], []
        for _, row in batch_df.iterrows():
            img = self._load_image(row['image_path'])

            if self.augment:
                img = self._augment_image(img)

            img = self.preprocess_fn(img)

            images.append(img)
            labels.append(row['label'])

        X = tf.stack(images)
        y = tf.convert_to_tensor(labels, dtype=tf.float32)

        if len(y.shape) == 1:
            y = tf.expand_dims(y, axis=-1)

        return X, y
