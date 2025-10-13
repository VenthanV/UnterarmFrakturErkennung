import os
import numpy as np
import tensorflow as tf
import pandas as pd

class PyDataset(tf.keras.utils.Sequence):
    def __init__(self, df, target_size=(224,224), batch_size=32,
                 preprocess_fn=None, shuffle=True, augment=False, **kwargs):
        super().__init__(**kwargs)
        if df is None or len(df) == 0:
            raise ValueError("DataFrame ist leer oder None.")
        self.df = df.reset_index(drop=True)
        self.batch_size = int(batch_size)
        self.preprocess_fn = preprocess_fn or (lambda x: x)
        self.target_size = tuple(target_size)
        self.shuffle = bool(shuffle)
        self.augment = bool(augment)
        self.indexes = np.arange(len(self.df))
        self.classes = self.df["label"].values
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _augment_tensor(self, image_tensor):
        # image_tensor: tf.Tensor float32, shape (H, W, C)
        # Alle Augment-Operationen als TF-ops, danach wieder resize
        if tf.random.uniform([]) < 0.5:
            image_tensor = tf.image.flip_left_right(image_tensor)
        if tf.random.uniform([]) < 0.3:
            image_tensor = tf.image.random_brightness(image_tensor, max_delta=0.2)
        if tf.random.uniform([]) < 0.3:
            image_tensor = tf.image.random_contrast(image_tensor, lower=0.8, upper=1.2)
        if tf.random.uniform([]) < 0.3:
            k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
            image_tensor = tf.image.rot90(image_tensor, k)
        if tf.random.uniform([]) < 0.3:
            # Zoom: crop a random box and resize back to target
            h, w = tf.shape(image_tensor)[0], tf.shape(image_tensor)[1]
            crop_frac = tf.random.uniform([], 0.8, 1.0)
            ch = tf.cast(tf.cast(h, tf.float32) * crop_frac, tf.int32)
            cw = tf.cast(tf.cast(w, tf.float32) * crop_frac, tf.int32)
            # ensure crop size at least 1
            ch = tf.maximum(ch, 1)
            cw = tf.maximum(cw, 1)
            image_tensor = tf.image.random_crop(image_tensor, size=[ch, cw, 3])
        # am Ende auf target_size bringen
        image_tensor = tf.image.resize(image_tensor, self.target_size)
        return image_tensor

    def load_and_preprocess(self, image_path):
        if not isinstance(image_path, (str, bytes)):
            raise ValueError(f"Ungültiger Pfad: {image_path!r}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Datei nicht gefunden: {image_path}")

        # load_img liefert PIL image; wir konvertieren direkt zu array und Tensor
        pil = tf.keras.utils.load_img(image_path, target_size=self.target_size)
        arr = tf.keras.utils.img_to_array(pil)  # numpy array float32
        tensor = tf.convert_to_tensor(arr)  # tf.Tensor

        if self.augment:
            tensor = self._augment_tensor(tensor)
        else:
            # Stelle sicher, dass Größe korrekt ist (falls load_img schon resize gemacht hat,
            # ist das redundant, aber sicher)
            tensor = tf.image.resize(tensor, self.target_size)

        # Wende preprocess_fn an. preprocess_fn kann TF-op oder numpy-fn sein.
        # Wir versuchen TF-compat: wenn callable gibt TF Tensor zurück, sonst numpy fallback.
        processed = self.preprocess_fn(tensor)
        # Wenn preprocess_fn zurückgibt numpy (z.B. skimage), handle das:
        if isinstance(processed, np.ndarray):
            out = processed.astype(np.float32)
        else:
            # TF Tensor -> to numpy (Eager mode)
            out = processed.numpy().astype(np.float32)

        # Sicherstellen: shape (H, W, C)
        if out.ndim == 2:
            out = np.expand_dims(out, axis=-1)
        return out

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.__len__():
            raise IndexError("Index außerhalb des gültigen Bereichs.")
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        if len(batch_indexes) == 0:
            raise ValueError(f"Batch {idx} hat keine Indizes (leerer Index-Block).")

        batch_df = self.df.iloc[batch_indexes]

        images = []
        labels = []
        failed = []

        for _, row in batch_df.iterrows():
            img_path = row.get("image_path")
            try:
                img = self.load_and_preprocess(img_path)
                images.append(img)
                labels.append(row["label"])
            except Exception as e:
                # Sammle fehlschläge für bessere Fehlermeldung
                failed.append((img_path, str(e)))
                print(f"⚠ Fehler beim Laden von '{img_path}': {e}")

        if not images:
            # ausführlichere Fehlermeldung mit First few failures
            first_errors = "; ".join([f"{p}: {m}" for p, m in failed[:5]])
            raise ValueError(f"Leerer Batch {idx}. {len(failed)} Dateien konnten nicht geladen werden. Beispiele: {first_errors}")

        # prüfe konsistente shapes
        shapes = [tuple(x.shape) for x in images]
        if len(set(shapes)) != 1:
            images = [np.asarray(tf.image.resize(tf.convert_to_tensor(x), self.target_size)).astype(np.float32) for x in images]

        X = np.stack(images).astype(np.float32)
        y = np.array(labels, dtype=np.float32)
        return X, y
