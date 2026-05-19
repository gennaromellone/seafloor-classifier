import numpy as np
import tensorflow.keras as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

IMG_W = 224
IMG_H = 224
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2
DATASET_DIR = "dataset"
OUTPUT_MODEL = "models/TM_new.h5"


def build_model():
    base = tf.applications.MobileNet(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_H, IMG_W, 3)
    )
    base.trainable = False

    model = tf.models.Sequential([
        base,
        tf.layers.GlobalAveragePooling2D(),
        tf.layers.Dense(128, activation="relu"),
        tf.layers.Dropout(0.3),
        tf.layers.Dense(2, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=VAL_SPLIT,
    )
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VAL_SPLIT,
    )

    train = train_datagen.flow_from_directory(
        DATASET_DIR,
        subset="training",
        target_size=(IMG_H, IMG_W),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val = val_datagen.flow_from_directory(
        DATASET_DIR,
        subset="validation",
        target_size=(IMG_H, IMG_W),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    print("Classi:", train.class_indices)
    print(f"Train: {train.n} immagini | Val: {val.n} immagini")

    model = build_model()
    model.summary()

    history = model.fit(
        train,
        epochs=EPOCHS,
        validation_data=val,
        callbacks=[
            tf.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_accuracy"),
            tf.callbacks.ModelCheckpoint(OUTPUT_MODEL, save_best_only=True, monitor="val_accuracy"),
        ]
    )

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()
    plt.savefig("models/training_history.png")
    print(f"Modello salvato in {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
