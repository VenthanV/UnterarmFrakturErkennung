# src/base_cnn_model.py
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.metrics import AUC, Precision, Recall, TopKCategoricalAccuracy
from tensorflow.keras.layers import BatchNormalization
import numpy as np

class BaseCNNModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=1, loss='auto', metrics=None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.loss = loss
        self.metrics = metrics

        # Standardmetriken & Loss automatisch setzen
        if metrics is None or loss == 'auto':
            if num_classes <= 2:
                print("🧩 Binary classification setup erkannt.")
                self.metrics = ['accuracy', AUC(name='auc'), Precision(name='precision'), Recall(name='recall')]
                self.loss = 'binary_crossentropy'
                # intern 1 Ausgabekanal verwenden (für predict-Logik)
                self.num_classes = 1
            else:
                print("🧩 Multi-class setup erkannt.")
                # robust für integer-Labels ohne One-Hot
                self.metrics = ['accuracy', TopKCategoricalAccuracy(k=3, name='top3')]
                self.loss = 'sparse_categorical_crossentropy'

        # Kindklasse muss build_model implementieren (setzt self.model und self.base_model)
        self.build_model()

    def build_model(self):
        """ Muss in der Kindklasse implementiert werden. """
        raise NotImplementedError("Die Kindklasse muss build_model implementieren")

    def _freeze_batchnorm_layers(self):
        """BatchNorm-Schichten generell nicht trainieren (stabileres Fine-Tuning)."""
        for layer in self.base_model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = False

    def train(self, train_gen, val_gen, epochs_feature=5, epochs_finetune=10, fine_tune_layers=40,
              learning_rate_ft=1e-5):
        """
        Zweistufiges Training:
        1️⃣ Feature Extraction (Backbone frozen)
        2️⃣ Fine-Tuning (letzte Layers freigeschaltet; BatchNorm bleibt gefreezed)
        """

        # Falls Wrapper übergeben wurden → echte Datasets extrahieren
        if hasattr(train_gen, "get_dataset"):
            train_gen = train_gen.get_dataset()
        if hasattr(val_gen, "get_dataset"):
            val_gen = val_gen.get_dataset()

        # Optional: steps nur setzen, wenn __len__ vorhanden ist
        steps_per_epoch = len(train_gen) if hasattr(train_gen, "__len__") else None
        validation_steps = len(val_gen) if hasattr(val_gen, "__len__") else None

        print("\n🚀 Stage 1: Feature Extraction (Backbone frozen)...")
        self.base_model.trainable = False
        self._freeze_batchnorm_layers()

        self.model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss=self.loss,
            metrics=self.metrics
        )

        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
            EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
        ]

        self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs_feature,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )

        print("\n🔓 Stage 2: Fine-Tuning der letzten Schichten...")
        if fine_tune_layers > 0:
            # gesamten Backbone erstmal trainierbar setzen
            self.base_model.trainable = True
            # bis auf die letzten N Schichten wieder sperren
            for layer in self.base_model.layers[:-fine_tune_layers]:
                layer.trainable = False

        # BatchNorm-Schichten grundsätzlich eingefroren lassen
        self._freeze_batchnorm_layers()

        self.model.compile(
            optimizer=AdamW(learning_rate=learning_rate_ft),
            loss=self.loss,
            metrics=self.metrics
        )

        self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs_finetune,
            callbacks=callbacks,
            verbose=1
        )

        print("🏁 Training abgeschlossen.")

    def predict(self, X):
        preds = self.model.predict(X)
        if self.num_classes == 1:
            # binär: Wahrscheinlichkeit → 0/1 runden
            return np.round(preds)
        else:
            # multi-class: argmax über Logits/Probs
            return np.argmax(preds, axis=-1)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
