# src/base_cnn_model.py
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger
from tensorflow.keras.metrics import AUC, Precision, Recall, TopKCategoricalAccuracy
from tensorflow.keras.layers import BatchNormalization
import numpy as np
import logging

# Get the logger instance
logger = logging.getLogger(__name__)

class BaseCNNModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=1, loss='auto', metrics=None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.loss = loss
        self.metrics = metrics
        self.model = None
        self.base_model = None

        # Loss & Metriken automatisch setzen anhand num_classes
        if loss == 'auto':
            if self.num_classes == 1:
                logger.info("Binary classification setup erkannt.")
                self.loss = 'binary_crossentropy'
                self.metrics = [
                    'accuracy',
                    AUC(name='auc'),
                    Precision(name='precision'),
                    Recall(name='recall')
                ]
            else:
                logger.info("Multi-class setup erkannt.")
                self.loss = 'sparse_categorical_crossentropy'
                self.metrics = [
                    'accuracy',
                    TopKCategoricalAccuracy(k=3, name='top3')
                ]

        # Modell erst JETZT bauen, nachdem num_classes & loss final sind
        self.build_model()


    def build_model(self):
        """ Muss in der Kindklasse implementiert werden. """
        raise NotImplementedError("Die Kindklasse muss build_model implementieren")

    def _freeze_batchnorm_layers(self):
        """BatchNorm-Schichten generell nicht trainieren (stabileres Fine-Tuning)."""
        if self.base_model is None:
            return
        for layer in self.base_model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = False

    def train(self, train_gen, val_gen,
              epochs_feature=5,
              epochs_finetune=10,
              fine_tune_layers=0,
              optimizer_fe=Adam(learning_rate=1e-3),
              optimizer_ft=AdamW(learning_rate=1e-5),
              callbacks=None):
        """
        Zweistufiges Training mit flexiblen Optimierern und Callbacks.
        1️⃣ Feature Extraction (Backbone frozen)
        2️⃣ Fine-Tuning (letzte Layers freigeschaltet; BatchNorm bleibt gefreezed)
        """

        # Standard-Callbacks definieren, falls keine übergeben wurden
        if callbacks is None:
            callbacks = [
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1),
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
                TensorBoard(log_dir='./logs/tensorboard', histogram_freq=1),
                CSVLogger('./logs/training_log.csv', append=True)
            ]

        # --- Stage 1: Feature Extraction ---
        logger.info("="*50)
        logger.info("🚀 Stage 1: Feature Extraction (Backbone frozen)...")
        self.base_model.trainable = False
        self._freeze_batchnorm_layers()

        self.model.compile(
            optimizer=optimizer_fe,
            loss=self.loss,
            metrics=self.metrics  # <-- KORREKTUR: self.metrics verwenden
        )

        self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs_feature,
            callbacks=callbacks,
            verbose=1
        )

        # --- Stage 2: Fine-Tuning ---
        if epochs_finetune > 0 and fine_tune_layers > 0:
            logger.info("="*50)
            logger.info(f"🔓 Stage 2: Fine-Tuning der letzten {fine_tune_layers} Schichten...")

            self.base_model.trainable = True
            # Sperre alle Schichten bis auf die letzten N
            for layer in self.base_model.layers[:-fine_tune_layers]:
                layer.trainable = False

            # BatchNorm-Schichten grundsätzlich eingefroren lassen
            self._freeze_batchnorm_layers()

            self.model.compile(
                optimizer=optimizer_ft,
                loss=self.loss,
                metrics=self.metrics  # <-- KORREKTUR: self.metrics verwenden
            )

            self.model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=epochs_finetune,
                callbacks=callbacks,
                verbose=1
            )

        logger.info("🏁 Training abgeschlossen.")

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
