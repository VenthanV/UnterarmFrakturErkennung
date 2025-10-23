import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

class ImageSequence(Sequence):
    def __init__(self, df, target_size=(224,224), batch_size=32,
                 preprocess_fn=None, shuffle=True, augment=False, **kwargs):
        # Wichtig: ruft Basiskonstruktor auf
        super().__init__(**kwargs)

        if df is None or len(df) == 0:
            raise ValueError("DataFrame ist leer oder None.")
        self.df = df.reset_index(drop=True)
        self.target_size = tuple(target_size)
        self.batch_size = int(batch_size)
        self.preprocess_fn = preprocess_fn or (lambda x: x)
        self.shuffle = bool(shuffle)
        self.augment = bool(augment)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _augment_image(self, image):
        if np.random.rand() < 0.5:
            image = tf.image.flip_left_right(image)
        if np.random.rand() < 0.5:
            image = tf.image.random_brightness(image, max_delta=0.2)
        if np.random.rand() < 0.5:
            image = tf.image.random_contrast(image, 0.8, 1.2)
        if np.random.rand() < 0.5:
            k = np.random.randint(0, 4)
            image = tf.image.rot90(image, k)
        if np.random.rand() < 0.5:
            image = tf.image.central_crop(image, central_fraction=0.9)
            image = tf.image.resize(image, self.target_size)
        return image

    def _load_image(self, path):
        if not os.path.exists(path):
            tf.print(f"⚠ Datei nicht gefunden: {path}")
            return tf.zeros((*self.target_size, 3), dtype=tf.float32)
        try:
            img = tf.io.read_file(path)
            img = tf.image.decode_image(img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, self.target_size)
            return img
        except Exception as e:
            tf.print(f"⚠ Fehler beim Laden von '{path}': {e}")
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
        return X, y
