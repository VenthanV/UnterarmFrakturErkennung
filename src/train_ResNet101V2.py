from ResNet101V2 import ResNet101V2Model
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from PyDataset import PyDataset
from tensorflow.keras.applications import resnet_v2
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications import EfficientNetV2M




def get_generators(train_df, val_df, batch_size=32, model_type="resnet_v2"):
    train_gen = PyDataset(train_df,target_size=(224,224), batch_size=batch_size, preprocess_fn=resnet_v2.preprocess_input, shuffle=True)
    val_gen = PyDataset(val_df,target_size=(224,224), batch_size=batch_size, preprocess_fn=resnet_v2.preprocess_input, shuffle=False)
    return train_gen, val_gen

if __name__ == "__main__":
    # Setze zufälligen Seed für Reproduzierbarkeit
    tf.random.set_seed(42)
    np.random.seed(42)

    ukgm = pd.read_csv("../data/dataset.csv")

    ukgm['image_path'] = '../data/' + ukgm['image_path'].astype(str)
    train_df, temp_df = (train_test_split
                         (ukgm,
                          test_size=0.3,
                          stratify = ukgm["label"],
                          random_state=42)
                         )
    val_df, test_df = (train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=42
    ))

    # --- 3. Kontrolle: Größen und Verteilung ---
    print("Train size:", train_df.shape)
    print(train_df['label'].value_counts(), '\n')

    print("Validation size:", val_df.shape)
    print(val_df['label'].value_counts(), '\n')

    print("Test size:", test_df.shape)
    print(test_df['label'].value_counts(), '\n')



    train_gen, val_gen = get_generators(train_df, val_df)






    # Modell initialisieren
    model = ResNet101V2Model(input_shape=(224,224,3))


    model.model.compile(optimizer=tf.keras.optimizers.AdamW(3e-4),
                        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
                        metrics = [
                            tf.keras.metrics.Recall(name="recall"),
                            tf.keras.metrics.Precision(name="precision"),
                            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                            tf.keras.metrics.AUC(name="auc")
                        ])


    # Training (mit Validation Split)
    # Statt X_train, y_train, ...:

    model.model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        batch_size=32,
        verbose=1
    )

