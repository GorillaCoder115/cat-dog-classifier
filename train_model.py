import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -------------------------
# Ensure static folder exists
# -------------------------
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

# -------------------------
# 1. Data Preprocessing
# -------------------------
train_dir = 'data/train'
img_size = (150, 150)
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# -------------------------
# 2. Build CNN Model
# -------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# -------------------------
# 3. Train Model
# -------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# Save model
model.save('dog_cat_model.h5')

# -------------------------
# 4. Visualizations
# -------------------------

# Accuracy plot
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy Over Epochs")
plt.legend()
plt.savefig(os.path.join(STATIC_DIR, "accuracy_plot.png"))
plt.close()

# Confusion Matrix
val_gen.reset()
predictions = (model.predict(val_gen) > 0.5).astype("int32")
cm = confusion_matrix(val_gen.classes, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Cat','Dog'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(STATIC_DIR, "confusion_matrix.png"))
plt.close()

# Class Distribution
labels = []
for folder in os.listdir(train_dir):
    folder_path = os.path.join(train_dir, folder)
    labels.extend([folder] * len(os.listdir(folder_path)))
pd.Series(labels).value_counts().plot(kind='bar', color=['orange','blue'])
plt.title("Class Distribution")
plt.savefig(os.path.join(STATIC_DIR, "class_distribution.png"))
plt.close()

print("Model and visualizations saved successfully!")
