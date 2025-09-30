# src/resnet_model.py
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam, AdamW
import numpy as np
import tensorflow as tf

class ResNet101V2Model:
    def __init__(self, input_shape=(224,224,3)):
        self.input_shape = input_shape
        self.build_model()

    def build_model(self):
        print("Baue ResNet101V2 Modell...")
        # Basis-Modell
        base = ResNet101V2(include_top=False, input_shape=self.input_shape, weights='imagenet')
        base.trainable = False  # zunächst alles einfrieren

        x = GlobalAveragePooling2D()(base.output)

        # Dense + BatchNorm + Swish + Dropout
        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x = Activation('swish')(x)
        x = Dropout(0.5)(x)

        output = Dense(1, activation='sigmoid')(x)
        self.model = Model(inputs=base.input, outputs=output)
        self.model.compile(
            optimizer=Adam(3e-4),
            loss='binary_crossentropy',
            metrics=[
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc")
            ]
        )
        self.base_model = base  # für späteres finetuning
        print("Modell gebaut.")






    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)