# src/base_cnn_model.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np
from tensorflow.keras.metrics import AUC, Precision, Recall

class BaseCNNModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=1, loss='binary_crossentropy', metrics=None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.loss = loss
        if metrics is None:
            if num_classes <= 2:
                print("Binary classification setup erkannt.")
                self.metrics = ['accuracy', AUC(name='auc'), Precision(name='precision'), Recall(name='recall')]
                self.loss = 'binary_crossentropy'
                self.num_classes = 1  # wichtig für predict()
            else:
                print("Multi-class setup erkannt.")
                self.metrics = ['accuracy']
                self.loss = 'categorical_crossentropy'

        self.build_model()

    def build_model(self):
        """
        Diese Methode MUSS in der Kindklasse implementiert werden.
        Sie sollte self.model und self.base_model setzen.
        """
        raise NotImplementedError("Die Kindklasse muss build_model implementieren")

    def train(self, train_gen, val_gen, epochs_feature=5, epochs_finetune=10, fine_tune_layers=40,
              learning_rate_ft=1e-5):
        """
        Zweistufiges Training:
        1. Feature Extraction (Backbone frozen)
        2. Fine-Tuning (letzte Schichten trainierbar)
        """
        print("\n🚀 Stage 1: Feature Extraction (Backbone frozen)...")
        self.base_model.trainable = False
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
            steps_per_epoch=len(train_gen),
            validation_steps=len(val_gen),
            callbacks=callbacks,
            verbose=1
        )

        print("\n🔓 Stage 2: Fine-Tuning der letzten Schichten...")
        # Letzte Schichten trainierbar machen
        if fine_tune_layers > 0:
            for layer in self.base_model.layers[-fine_tune_layers:]:
                layer.trainable = True

        self.model.compile(
            optimizer=AdamW(learning_rate=learning_rate_ft),
            loss=self.loss,
            metrics=self.metrics
        )
        self.model.fit(train_gen, validation_data=val_gen, epochs=epochs_finetune, callbacks=callbacks,multiprocessing=True, verbose=1)
        print("🏁 Training abgeschlossen.")

    def predict(self, X):
        preds = self.model.predict(X)
        if self.num_classes == 1:
            return np.round(preds)
        else:
            return np.argmax(preds, axis=-1)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
