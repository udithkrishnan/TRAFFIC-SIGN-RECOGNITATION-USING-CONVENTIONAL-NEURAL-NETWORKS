import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load GTSRB dataset
def load_data(data_dir):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        if label.isdigit():
            for image_file in os.listdir(os.path.join(data_dir, label)):
                image_path = os.path.join(data_dir, label, image_file)
                image = cv2.imread(image_path)
                image = cv2.resize(image, (32, 32))
                images.append(image)
                labels.append(int(label))
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Preprocess data
def preprocess_data(X, y):
    X = X.astype('float32') / 255.0
    y = to_categorical(y, num_classes=len(np.unique(y)))
    return X, y

# Build CNN model
def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Main program
data_dir = r'C:\Users\uditk\OneDrive\Desktop\project\archive\Train'   # Update with the path to the GTSRB dataset
X, y = load_data(data_dir)
X, y = preprocess_data(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_shape = X_train.shape[1:]
num_classes = y_train.shape[1]

model = build_model(input_shape, num_classes)
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy:.4f}')

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()

# Make predictions
def predict_image(model, image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (32, 32))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return np.argmax(prediction)

# Example usage
image_path = r'C:\Users\uditk\OneDrive\Desktop\project\archive\Test'  # Update with the path to a test image
predicted_label = predict_image(model, image_path)
print(f'Predicted label: {predicted_label}')
