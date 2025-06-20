import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense,
    BatchNormalization,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import itertools
from tensorflow.keras import backend as K
import gc

# --- Paths ---
train_data_path = "/content/face_emo/dataset/fer2013/train"
val_data_path = "/content/face_emo/dataset/fer2013/validation"
models_dir = "/content/facial_emotion_detection/models"
best_model_path = os.path.join(models_dir, "best_emotion_model.h5")


# --- Load train and test (split from training folder) ---
def load_and_split_data(train_path, img_size=(48, 48), test_size=0.2, batch_size=64):
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=test_size)

    train_gen = datagen.flow_from_directory(
        train_path,
        target_size=img_size,
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=batch_size,
        subset="training",
        shuffle=True,
    )

    test_gen = datagen.flow_from_directory(
        train_path,
        target_size=img_size,
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=batch_size,
        subset="validation",
        shuffle=True,
    )

    return train_gen, test_gen


# --- Load true validation (unseen) data ---
def load_final_validation_data(val_path, img_size=(48, 48), batch_size=64):
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_gen = datagen.flow_from_directory(
        val_path,
        target_size=img_size,
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False,
    )
    return val_gen


# --- Parameters to Grid Search ---
param_grid = {
    "filters": [32, 64],
    "dense_units": [128, 256],
    "dropout_rate": [0.3, 0.5],
}


# --- Build CNN Model ---
def build_model(
    filters=64,
    dense_units=256,
    dropout_rate=0.5,
    input_shape=(48, 48, 1),
    num_classes=7,
):
    model = Sequential()
    model.add(Conv2D(filters, (3, 3), activation="relu", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(filters * 2, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(dense_units, activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.0001),
        metrics=["accuracy"],
    )
    return model


# --- Grid Search Training ---
def train_with_grid_search():
    os.makedirs(models_dir, exist_ok=True)

    train_gen, test_gen = load_and_split_data(train_data_path)
    final_val_gen = load_final_validation_data(val_data_path)

    best_val_acc = 0
    best_model = None
    best_params = {}

    for params in itertools.product(*param_grid.values()):
        filters, dense_units, dropout_rate = params

        # Skip the first 4 combinations (already done)
        # if (filters, dense_units, dropout_rate) in [
        #    (32, 128, 0.3),
        #    (32, 128, 0.5),
        #    (32, 256, 0.3),
        #    (32, 256, 0.5),
        # ]:
        #    print(
        #        f"Skipping already trained model: filters={filters}, dense_units={dense_units}, dropout_rate={dropout_rate}"
        #    )
        #    continue

        print(
            f"Training model with filters={filters}, dense_units={dense_units}, dropout_rate={dropout_rate}"
        )

        model = build_model(filters, dense_units, dropout_rate)

        temp_model_path = os.path.join(
            models_dir, f"temp_model_{filters}_{dense_units}_{int(dropout_rate*10)}.h5"
        )
        checkpoint = ModelCheckpoint(
            temp_model_path, monitor="val_loss", save_best_only=True, verbose=0
        )

        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
            ),
            checkpoint,
        ]

        model.fit(
            train_gen,
            validation_data=test_gen,
            epochs=20,
            callbacks=callbacks,
            verbose=1,
        )

        val_loss, val_acc = model.evaluate(final_val_gen, verbose=0)
        print(f"Final validation accuracy: {val_acc:.4f}")

        # Keep best model only
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            best_params = {
                "filters": filters,
                "dense_units": dense_units,
                "dropout_rate": dropout_rate,
            }

            # Save best model
            best_model.save(best_model_path)

        # After model training & evaluation
        K.clear_session()
        gc.collect()

        # Delete temp model file
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

    if best_model:
        print(
            f"\nBest model saved as '{best_model_path}' with accuracy: {best_val_acc:.4f}"
        )
        print("Best parameters:", best_params)
    else:
        print("No model trained successfully.")


if __name__ == "__main__":
    train_with_grid_search()
