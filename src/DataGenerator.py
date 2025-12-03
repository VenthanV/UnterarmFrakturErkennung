# src/data_generator.py
from typing import Tuple, Callable
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import resnet_v2, convnext

from PyDataset import ImageSequence

class DataGenerator:
    """
    A utility class for creating data generators and splitting datasets for training.
    """

    @staticmethod
    def _get_preprocess_fn(model_type: str) -> Callable:
        """
        Returns the appropriate preprocessing function for a given model type.
        """
        model_type = model_type.lower()
        if model_type.startswith("resnet"):
            return resnet_v2.preprocess_input
        elif model_type.startswith("convnext"):
            return convnext.preprocess_input
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    @staticmethod
    def get_generators(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        model_type: str,
        target_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32
    ) -> Tuple[ImageSequence, ImageSequence]:
        """
        Creates training and validation data generators.

        Args:
            train_df: DataFrame for the training set.
            val_df: DataFrame for the validation set.
            model_type: The type of model ('resnet', 'convnext', etc.).
            target_size: The target image dimensions (height, width).
            batch_size: The number of samples per batch.

        Returns:
            A tuple containing the training and validation ImageSequence generators.
        """
        preprocess_fn = DataGenerator._get_preprocess_fn(model_type)

        train_gen = ImageSequence(
            train_df,
            target_size=target_size,
            batch_size=batch_size,
            preprocess_fn=preprocess_fn,
            augment=True,
            shuffle=True
        )

        val_gen = ImageSequence(
            val_df,
            target_size=target_size,
            batch_size=batch_size,
            preprocess_fn=preprocess_fn,
            augment=False,
            shuffle=False
        )

        return train_gen, val_gen

    @staticmethod
    def split_data(
        df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.25,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits a DataFrame into training, validation, and test sets.

        The split is performed stratified by the 'label' column.

        Args:
            df: The complete DataFrame to split.
            test_size: The proportion of the dataset to include in the test split.
            val_size: The proportion of the *training* set to use for validation.
                      e.g., test_size=0.2, val_size=0.25 results in:
                      - 80% for train+val -> 25% of this is 20% of total for val
                      - 60% Train, 20% Validation, 20% Test
            random_state: The seed for the random number generator.

        Returns:
            A tuple containing the training, validation, and test DataFrames.
        """
        # Split into train+validation set and test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df["label"],
            random_state=random_state
        )

        # Split the train+validation set into a training set and a validation set
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size,
            stratify=train_val_df["label"],
            random_state=random_state
        )

        return train_df, val_df, test_df

    @staticmethod
    def get_test_generator(
        test_df: pd.DataFrame,
        model_type: str,
        target_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32
    ) -> ImageSequence:
        """
        Creates a data generator for the test set.

        Args:
            test_df: DataFrame for the test set.
            model_type: The type of model ('resnet', 'convnext', etc.).
            target_size: The target image dimensions (height, width).
            batch_size: The number of samples per batch.

        Returns:
            An ImageSequence generator for the test set.
        """
        preprocess_fn = DataGenerator._get_preprocess_fn(model_type)

        test_gen = ImageSequence(
            test_df,
            target_size=target_size,
            batch_size=batch_size,
            preprocess_fn=preprocess_fn,
            augment=False,
            shuffle=False
        )
        return test_gen
