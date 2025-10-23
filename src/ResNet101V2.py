# src/resnet_model.py
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from Base_CNN_Model import BaseCNNModel
from tensorflow.keras.applications.resnet_v2 import preprocess_input

import tensorflow as tf
import numpy as np
import pandas as pd
from DataGenerator import DataGenerator
from PyDataset import ImageSequence

class ResNet101V2Model(BaseCNNModel):
    def build_model(self):
        base = ResNet101V2(include_top=False, input_shape=self.input_shape, weights='imagenet')
        base.trainable = False
        x = GlobalAveragePooling2D()(base.output)
        x = Dense(128, kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Activation('swish')(x)
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
    ukgm['label'] = ukgm['label'].astype(int)

    # Daten splitten
    train_df, val_df, test_df = DataGenerator.split_data(ukgm)
    model = ResNet101V2Model(input_shape=(224, 224, 3), num_classes=ukgm['label'].nunique())

    # Generatoren als tf.data.Dataset
    train_ds, val_ds = DataGenerator.get_generators(train_df, val_df, model_type="resnet", batch_size=32)

    # Training
    model.train(train_ds, val_ds, epochs_feature=5, epochs_finetune=25, fine_tune_layers=40)


