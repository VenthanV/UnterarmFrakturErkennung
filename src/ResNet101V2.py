# src/resnet_model.py
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

import numpy as np
import pandas as pd
from DataGenerator import DataGenerator

from Base_CNN_Model import BaseCNNModel  # Pfad/Namensgebung wie bei dir
# Falls ihr das Preprocessing im DataGenerator erledigt, hier NICHT erneut anwenden.

class ResNet101V2Model(BaseCNNModel):
    def build_model(self, dropout_rate=0.3, dense_units=128, weight_decay=1e-4):
        base = ResNet50V2(
            include_top=False,
            input_shape=self.input_shape,
            weights='imagenet'
        )

        # Start: Feature-Extraktion
        base.trainable = False

        x = GlobalAveragePooling2D(name="gap")(base.output)
        x = Dense(dense_units, kernel_regularizer=l2(weight_decay), name="fc")(x)
        x = BatchNormalization(name="fc_bn")(x)
        x = Activation('swish', name="fc_swish")(x)
        x = Dropout(dropout_rate, name="fc_do")(x)

        # Output je nach Setup aus Base:
        if self.num_classes == 1:
            # binary (Base hat loss='binary_crossentropy' gesetzt)
            output = Dense(1, activation='sigmoid', dtype="float32", name="pred")(x)
        else:
            # multi-class (Base nutzt sparse_categorical_crossentropy)
            output = Dense(self.num_classes, activation='softmax', dtype="float32", name="pred")(x)

        self.model = Model(inputs=base.input, outputs=output, name="ResNet101V2_custom")
        self.base_model = base


if __name__ == "__main__":
    # Seed setzen
    tf.random.set_seed(42)
    np.random.seed(42)

    # Dataset laden
    ukgm = pd.read_csv("../data/dataset.csv")
    ukgm['image_path'] = '../data/' + ukgm['image_path'].astype(str)
    ukgm['label'] = ukgm['label'].astype(int)

    # Daten splitten
    train_df, val_df, test_df = DataGenerator.split_data(ukgm)
    model = ResNet101V2Model(input_shape=(224, 224, 3), num_classes=ukgm['label'].nunique())

    # Generatoren als tf.data.Dataset
    train_ds, val_ds = DataGenerator.get_generators(train_df, val_df, model_type="resnet", batch_size=32)

    # Training
    model.train(train_ds, val_ds, epochs_feature=5, epochs_finetune=25, fine_tune_layers=40)


