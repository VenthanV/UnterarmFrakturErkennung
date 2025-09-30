# src/resnet_model.py
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam, AdamW
import numpy as np
import tensorflow as tf

class ResNet101V2Model:
    def __init__(self, input_shape=(224,224,3), num_classes=10, fine_tune_after=5, fine_tune_lr=1e-4, unfreeze_last_n=40):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.fine_tune_after = fine_tune_after
        self.fine_tune_lr = fine_tune_lr
        self.unfreeze_last_n = unfreeze_last_n
        self.build_model()

    def build_model(self):
        print("Baue ResNet101V2 Modell...")
        # Basis-Modell
        base = ResNet50V2(include_top=False, input_shape=self.input_shape, weights='imagenet')
        base.trainable = False  # zunächst alles einfrieren

        x = GlobalAveragePooling2D()(base.output)

        # Dense + BatchNorm + Swish + Dropout
        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x = Activation('swish')(x)
        x = Dropout(0.5)(x)

        output = Dense(self.num_classes, activation='softmax')(x)
        self.model = Model(inputs=base.input, outputs=output)
        self.model.compile(optimizer=Adam(3e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.base_model = base  # speichern für Fine-Tuning
        print("Modell gebaut.")

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=70, batch_size=32):
        # Zuerst Training mit eingefrorenem Backbone
        initial_epochs = self.fine_tune_after
        print("Training mit eingefrorenem Backbone für {} Epochen...".format(initial_epochs))
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=initial_epochs,
            batch_size=batch_size,
            verbose=1
        )

        # Fine-Tuning: Letzte N Layer des Backbones entfroren
        self.base_model.trainable = True
        for layer in self.base_model.layers[:-self.unfreeze_last_n]:
            layer.trainable = False

        self.model.compile(optimizer=Adam(self.fine_tune_lr),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        # Restliche Epochen trainieren
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs - initial_epochs,
            batch_size=batch_size
        )

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)