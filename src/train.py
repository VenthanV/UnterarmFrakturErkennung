
from ResNet101V2 import ResNet101V2Model
import tensorflow as tf
import numpy as np
import os

if __name__ == "__main__":
    # Setze zufälligen Seed für Reproduzierbarkeit
    tf.random.set_seed(42)
    np.random.seed(42)
    print("Test")




    X_train = None
    y_train = None
    X_test = None
    y_test = None


    # Modell initialisieren
    model = ResNet101V2Model(input_shape=(224,224,3), num_classes=10)
    model.build_model()
    # Training (mit Validation Split)
    model.train(
    X_train, y_train,
    X_val=X_test, y_val=y_test,
    epochs=10, batch_size=32
    )

    # Modell speichern
    os.makedirs('models', exist_ok=True)
    model.model.save('models/resnet101v2_fracturedetection.keras')
    print("Modell gespeichert unter: models/resnet101v2_fracturedetection.keras")

    # Evaluation
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test-Loss: {loss:.4f}, Test-Accuracy: {acc:.4f}")