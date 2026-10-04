import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import os

# -----------------------------
# 1. Load dataset
# -----------------------------

df = pd.read_csv("HAM10000_cleaned_metadata.csv")

image_folder = "HAM10000_images"

df["image_path"] = df["image_id"].apply(
    lambda x: f"{image_folder}/{x}.jpg"
)

# Keep only images that exist
df = df[df["image_path"].apply(os.path.exists)].copy()

df["is_malignant"] = df["is_malignant"].astype(str)

print("Images available:", len(df))

# -----------------------------
# 2. Image generators
# -----------------------------
print("\nUnique lesions:")
print(df["lesion_id"].nunique())

print("\nTotal images:")
print(len(df))

print("\nImages per lesion:")
print(df.groupby("lesion_id").size().describe())

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_dataframe(
    dataframe=df,
    x_col="image_path",
    y_col="is_malignant",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="training",
    shuffle=True
)

validation_data = datagen.flow_from_dataframe(
    dataframe=df,
    x_col="image_path",
    y_col="is_malignant",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# -----------------------------
# 3. MobileNetV2
# -----------------------------

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False

# -----------------------------
# 4. Build model
# -----------------------------

model = models.Sequential([
    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.3),

    layers.Dense(128, activation="relu"),

    layers.Dropout(0.3),

    layers.Dense(1, activation="sigmoid")
])

# -----------------------------
# 5. Compile
# -----------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# 6. Train
# -----------------------------

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=10
)

# -----------------------------
# 7. Save model
# -----------------------------

model.save("model.keras")

print("\nModel saved as model.keras")

# -----------------------------
# 8. Evaluate the model
# -----------------------------

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Get predictions
validation_data.reset()

predictions = model.predict(validation_data)

# Convert probabilities to 0 or 1
predicted_classes = (predictions > 0.5).astype(int).flatten()

# Actual labels
true_classes = validation_data.classes

print("\nClassification Report:")
print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=["Benign", "Malignant"]
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        true_classes,
        predicted_classes
    )
)