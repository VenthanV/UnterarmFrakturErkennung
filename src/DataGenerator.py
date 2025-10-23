# src/data_generator.py
from PyDataset import ImageSequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import resnet_v2, convnext

class DataGenerator:

    @staticmethod
    def get_generators(train_df, val_df, model_type, batch_size=32):
        if not model_type:
            raise ValueError("model_type must be specified!")

        # Wähle passende Preprocessing-Funktion
        model_type = model_type.lower()
        if model_type.startswith("resnet"):
            preprocess_fn = resnet_v2.preprocess_input
        elif model_type.startswith("convnext"):
            preprocess_fn = convnext.preprocess_input
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        train_gen = ImageSequence(
            train_df,
            target_size=(224, 224),
            batch_size=batch_size,
            preprocess_fn=preprocess_fn,
            augment=True,
            shuffle=True
        )

        val_gen = ImageSequence(
            val_df,
            target_size=(224, 224),
            batch_size=batch_size,
            preprocess_fn=preprocess_fn,
            shuffle=False
        )

        return train_gen, val_gen

    @staticmethod
    def split_data(ukgm, test_size=0.3, val_size=0.5, random_state=42):
        train_df, temp_df = train_test_split(
            ukgm,
            test_size=test_size,
            stratify=ukgm["label"],
            random_state=random_state
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=val_size,
            stratify=temp_df["label"],
            random_state=random_state
        )

        return train_df, val_df, test_df
