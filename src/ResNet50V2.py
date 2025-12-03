# src/resnet50v2_model.py
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import numpy as np
import pandas as pd
import logging
import os

from Base_CNN_Model import BaseCNNModel
from DataGenerator import DataGenerator

def setup_logging():
    """Configures the root logger."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, "execution.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

class ResNet50V2Model(BaseCNNModel):
    def __init__(self, dense_units=128, dropout_rate=0.3, weight_decay=1e-4, *args, **kwargs):
        # Store architecture-specific hyperparameters
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        
        # Call the parent constructor, which will in turn call build_model
        super().__init__(*args, **kwargs)

    def build_model(self):
        """Builds the ResNet50V2-based custom model."""
        base = ResNet50V2(
            include_top=False,
            input_shape=self.input_shape,
            weights='imagenet'
        )
        base.trainable = False  # Start with a frozen backbone

        # Custom head
        x = GlobalAveragePooling2D(name="gap")(base.output)
        x = Dense(self.dense_units, kernel_regularizer=l2(self.weight_decay), name="fc")(x)
        x = BatchNormalization(name="fc_bn")(x)
        x = Activation('swish', name="fc_swish")(x)
        x = Dropout(self.dropout_rate, name="fc_do")(x)

        # Output layer
        if self.num_classes == 1:
            output = Dense(1, activation='sigmoid', dtype="float32", name="pred")(x)
        else:
            output = Dense(self.num_classes, activation='softmax', dtype="float32", name="pred")(x)

        # Assign models to instance attributes
        self.model = Model(inputs=base.input, outputs=output, name="ResNet50V2_custom")
        self.base_model = base


if __name__ == "__main__":
    # --- Configuration ---
    INPUT_SHAPE = (224, 224, 3)
    NUM_CLASSES = 1  # Binary classification
    BATCH_SIZE = 32
    
    # Model Hyperparameters
    DENSE_UNITS = 256
    DROPOUT_RATE = 0.4
    
    # Training Hyperparameters
    EPOCHS_FEATURE = 5
    EPOCHS_FINETUNE = 25
    FINE_TUNE_LAYERS = 40

    # --- Setup ---
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Setting random seeds for reproducibility.")
    tf.random.set_seed(42)
    np.random.seed(42)

    # --- Data Loading ---
    logger.info("Loading dataset...")
    try:
        ukgm = pd.read_csv("../data/dataset.csv")
        ukgm['image_path'] = '../data/' + ukgm['image_path'].astype(str)
        ukgm['label'] = ukgm['label'].astype(int)
        logger.info(f"Dataset loaded successfully with {len(ukgm)} records.")
    except FileNotFoundError:
        logger.error("Error: dataset.csv not found. Please check the path.", exc_info=True)
        exit()

    # --- Data Splitting & Generators ---
    logger.info("Splitting data and creating generators...")
    train_df, val_df, test_df = DataGenerator.split_data(ukgm)
    
    train_ds, val_ds = DataGenerator.get_generators(
        train_df, val_df, model_type="resnet", batch_size=BATCH_SIZE
    )
    
    test_ds = DataGenerator.get_test_generator(
        test_df, model_type="resnet", batch_size=BATCH_SIZE
    )

    # --- Model Initialization ---
    logger.info("Initializing ResNet50V2 model...")
    model = ResNet50V2Model(
        input_shape=INPUT_SHAPE,
        num_classes=NUM_CLASSES,
        loss='auto',
        dense_units=DENSE_UNITS,
        dropout_rate=DROPOUT_RATE
    )
    
    # Log model summary to a string to avoid messy print output
    string_list = []
    model.model.summary(print_fn=lambda x: string_list.append(x))
    logger.info("Model Summary:\n" + "\n".join(string_list))

    # --- Training ---
    logger.info("Starting model training...")
    model.train(
        train_ds,
        val_ds,
        epochs_feature=EPOCHS_FEATURE,
        epochs_finetune=EPOCHS_FINETUNE,
        fine_tune_layers=FINE_TUNE_LAYERS
    )

    # --- Evaluation ---
    logger.info("Evaluating final model on the test set...")
    results = model.model.evaluate(test_ds)
    results_dict = dict(zip(model.model.metrics_names, results))
    logger.info(f"Test Results: {results_dict}")

    # --- Saving Model ---
    logger.info("Saving the final model...")
    model.model.save("resnet50v2_custom_model.keras")
    logger.info("Model saved to resnet50v2_custom_model.keras")
