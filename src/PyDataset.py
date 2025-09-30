import numpy as np
import tensorflow as tf
import pandas as pd

class PyDataset(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=32, preprocess_fn=None, shuffle=True, **kwargs):
        super().__init__(**kwargs)  # <-- Wichtig: für Keras-Kompatibilität

        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.preprocess_fn = preprocess_fn or (lambda x: x)
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.classes = self.df["label"].values  # Für evaluate()
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indexes]

        images = []
        labels = []

        for _, row in batch_df.iterrows():
            try:
                image = self.load_and_preprocess(row["image_path"])
                images.append(image)
                labels.append(row["label"])
            except Exception as e:
                print(f"⚠ Fehler beim Laden von '{row['image_path']}': {e}")

        if not images:
            raise ValueError(f"Leerer Batch {idx}. Prüfe Pfade oder DataFrame.")

        return np.stack(images), np.array(labels, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def load_and_preprocess(self, image_path):
        image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        image = tf.keras.utils.img_to_array(image)
        return self.preprocess_fn(image)
