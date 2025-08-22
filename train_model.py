import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay




STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)



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



history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)


model.save('dog_cat_model.h5')




plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy Over Epochs")
plt.legend()
plt.savefig(os.path.join(STATIC_DIR, "accuracy_plot.png"))
plt.close()


val_gen.reset()
predictions = (model.predict(val_gen) > 0.5).astype("int32")
cm = confusion_matrix(val_gen.classes, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Cat','Dog'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(STATIC_DIR, "confusion_matrix.png"))
plt.close()


labels = []
for folder in os.listdir(train_dir):
    folder_path = os.path.join(train_dir, folder)
    labels.extend([folder] * len(os.listdir(folder_path)))
pd.Series(labels).value_counts().plot(kind='bar', color=['orange','blue'])
plt.title("Class Distribution")
plt.savefig(os.path.join(STATIC_DIR, "class_distribution.png"))
plt.close()

print("Model and visualizations saved successfully!")



# SOURCES FOR THIS CODE: 

#machinelearningmastery. (2025b). How to classify photos of dogs and cats (with 97% accuracy) - machinelearningmastery.com. https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/ 

 

#Team, K. (2025). Keras documentation: Image Classification From Scratch. https://keras.io/examples/vision/image_classification_from_scratch/ 

#GeeksforGeeks. (2025, July 23). Cat & dog classification using Convolutional Neural Network in python. https://www.geeksforgeeks.org/deep-learning/cat-dog-classification-using-convolutional-neural-network-in-python/ 











