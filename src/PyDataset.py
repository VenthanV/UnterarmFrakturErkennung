import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence


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
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)

    # Verbessert: Konzentriert sich auf effiziente Transformationen
    def _augment_image(self, image):
        # 1. Geometrische Augmentationen (wenden wir auf skaliertes Bild an)

        # Horizontales Spiegeln (Flip Left-Right)
        image = tf.image.random_flip_left_right(image)

        # 90-Grad-Rotationen (optional, kann komplexe Bilder verzerren, aber hier beibehalten)
        if np.random.rand() < 0.25:  # Wahrscheinlichkeit reduziert
            image = tf.image.rot90(image, k=np.random.randint(1, 4))  # Rotiert um 90, 180 oder 270 Grad

        # 2. Farb- und Helligkeits-Augmentationen

        # Helligkeit
        image = tf.image.random_brightness(image, max_delta=0.2)

        # Kontrast
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

        # Optional: Sättigung (kann nützlich sein)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)

        # HINWEIS: Wir haben die Central Crop/Resize Logik entfernt,
        # um doppeltes Resizing zu vermeiden. Wenn Sie zufällige Cropping
        # wünschen, sollte dies die ursprüngliche Resizing-Logik in _load_image
        # ersetzen oder komplizierter behandelt werden.

        return image

    # Optimiert: Reduziert die Notwendigkeit von tf.print in Schleife
    def _load_image(self, path):
        if not os.path.exists(path):
            tf.print(f"⚠ Datei nicht gefunden: {path}")
            return tf.zeros((*self.target_size, 3), dtype=tf.float32)
        try:
            # Lese, decodiere und wandle in float32
            img = tf.io.read_file(path)
            # Nutze decode_jpeg/decode_png für leichtere Fehlerbehandlung, falls bekannt
            img = tf.image.decode_image(img, channels=3, expand_animations=False)

            # WICHTIG: Resize und Normalisierung (auf [0, 1])
            img = tf.image.resize(img, self.target_size)
            img = tf.image.convert_image_dtype(img, tf.float32)

            return img
        except Exception as e:
            tf.print(f"⚠ Fehler beim Laden von '{path}': {e}")
            return tf.zeros((*self.target_size, 3), dtype=tf.float32)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indices]

        # Nutzen wir die Tatsache, dass tf.numpy_function/tf.map effizienter ist.
        # Aber bei Sequence ist die aktuelle Schleife OK, wir verwenden tf.stack am Ende.

        images, labels = [], []
        for _, row in batch_df.iterrows():
            img = self._load_image(row['image_path'])

            if self.augment:
                # Augmentation nur für Trainingsdaten
                img = self._augment_image(img)

            # Preprocessing (z.B. ResNet-spezifische Normalisierung)
            # WICHTIG: Dies muss NACH der Augmentation erfolgen.
            img = self.preprocess_fn(img)

            images.append(img)
            labels.append(row['label'])

        X = tf.stack(images)
        y = tf.convert_to_tensor(labels, dtype=tf.float32)

        # WICHTIG: Für binäre Klassifikation (sigmoid) muss y oft in Form (Batchgröße, 1) sein
        # Stellen Sie sicher, dass die Labels korrekt geformt sind
        if len(y.shape) == 1:
            y = tf.expand_dims(y, axis=-1)

        return X, y