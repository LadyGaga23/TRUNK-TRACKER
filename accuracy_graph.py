import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define paths and image size
data_dir = r'C:\Users\hp\Desktop\4 Sem\human-wildlife-conflict-master\Elephant'
image_size = (224, 224)

# Function to load raw images for visualization
def load_raw_images(data_dir, count=10):
    images = []
    for label in ['single_elephant', 'multiple_separate_elephants', 'multiple_obstructing_elephants', 'human_and_elephant']:
        path = os.path.join(data_dir, 'Object', label)
        for file in os.listdir(path)[:count // 2]:
            img = cv2.imread(os.path.join(path, file))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
    for label in ['human', 'goat']:
        path = os.path.join(data_dir, 'Object', label)
        for file in os.listdir(path)[:count // 2]:
            img = cv2.imread(os.path.join(path, file))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
    return images[:count]

# Plot image grid
def plot_images(images, title, rows=2, cols=5):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
    fig.suptitle(title, fontsize=16)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.imshow(images[i])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Load raw images and show
raw_images = load_raw_images(data_dir)
plot_images(raw_images, "Images Before Preprocessing")

# Load and preprocess data
def load_preprocessed_data(data_dir):
    data, labels = [], []
    for label in ['single_elephant', 'multiple_separate_elephants', 'multiple_obstructing_elephants', 'human_and_elephant']:
        path = os.path.join(data_dir, 'Object', label)
        for file in os.listdir(path):
            img = cv2.imread(os.path.join(path, file))
            if img is not None:
                img = cv2.resize(img, image_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img_to_array(img) / 255.0
                data.append(img)
                labels.append(1)
    for label in ['human', 'goat']:
        path = os.path.join(data_dir, 'Object', label)
        for file in os.listdir(path):
            img = cv2.imread(os.path.join(path, file))
            if img is not None:
                img = cv2.resize(img, image_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img_to_array(img) / 255.0
                data.append(img)
                labels.append(0)
    return np.array(data), np.array(labels)

# Show preprocessed images
data, labels = load_preprocessed_data(data_dir)
plot_images(data[:10], "Images After Preprocessing")

# Split data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32)

# Plot accuracy & loss
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['binary_accuracy'], label="Train Acc")
    plt.plot(history.history['val_binary_accuracy'], label="Val Acc")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label="Train Loss")
    plt.plot(history.history['val_loss'], label="Val Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Show graphs
plot_training_history(history)
