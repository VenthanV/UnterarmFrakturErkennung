# src/convnext_model.py
from tensorflow.keras.applications import ConvNeXtSmall
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model

import tensorflow as tf
import numpy as np
import pandas as pd

from Base_CNN_Model import BaseCNNModel
from DataGenerator import DataGenerator

class ConvNeXtModel(BaseCNNModel):
    def build_model(self):
        base = ConvNeXtSmall(include_top=False, weights='imagenet', input_shape=self.input_shape, pooling='avg')
        base.trainable = False

        x = Dense(256, activation='relu')(base.output)
        x = Dropout(0.3)(x)
        output = Dense(1, activation='sigmoid')(x)

        self.model = Model(inputs=base.input, outputs=output)
        self.base_model = base

if __name__ == "__main__":

    # Seed setzen
    tf.random.set_seed(42)
    np.random.seed(42)

    # Dataset laden
    ukgm = pd.read_csv("../data/dataset.csv")
    ukgm['image_path'] = '../data/' + ukgm['image_path'].astype(str)

    # Daten splitten
    train_df, val_df, test_df = DataGenerator.split_data(ukgm)

    # Generatoren erstellen
    train_gen, val_gen = DataGenerator.get_generators(train_df, val_df, model_type="convnext", batch_size=32)

    # Modell initialisieren
    num_classes = ukgm['label'].nunique()
    model = ConvNeXtModel(input_shape=(224, 224, 3), num_classes=num_classes)

    # Optional: Model summary anzeigen
    #model.model.summary()

    # Training starten
    model.train(train_gen, val_gen, epochs_feature=5, epochs_finetune=10)
