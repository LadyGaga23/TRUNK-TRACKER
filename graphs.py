import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array

# Define dataset path and image size
data_dir = r'C:\Users\hp\Desktop\4 Sem\human-wildlife-conflict-master\Elephant'
image_size = (224, 224)

# Function to load raw images before preprocessing (for plotting)
def load_raw_images(data_dir, count=10):
    images = []
    for object_label in ['single_elephant', 'multiple_separate_elephants', 'multiple_obstructing_elephants', 'human_and_elephant']:
        object_dir = os.path.join(data_dir, 'Object', object_label)
        for img_file in os.listdir(object_dir)[:count // 2]:
            img_path = os.path.join(object_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
    for object_label in ['human', 'goat']:
        object_dir = os.path.join(data_dir, 'Object', object_label)
        for img_file in os.listdir(object_dir)[:count // 2]:
            img_path = os.path.join(object_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
    return images[:count]

# Function to display images
def plot_images(images, title, rows=2, cols=5):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
    fig.suptitle(title, fontsize=16)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.imshow(images[i])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Load and show raw images
raw_images = load_raw_images(data_dir)
plot_images(raw_images, "Images Before Preprocessing")

# Function to load and preprocess images
def load_preprocessed_data(data_dir, count=10):
    preprocessed_images = []
    labels = []

    for object_label in ['single_elephant', 'multiple_separate_elephants', 'multiple_obstructing_elephants', 'human_and_elephant']:
        object_dir = os.path.join(data_dir, 'Object', object_label)
        for img_file in os.listdir(object_dir)[:count // 2]:
            img_path = os.path.join(object_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, image_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img_to_array(img) / 255.0  # Normalize
                preprocessed_images.append(img)
                labels.append(1)

    for object_label in ['human', 'goat']:
        object_dir = os.path.join(data_dir, 'Object', object_label)
        for img_file in os.listdir(object_dir)[:count // 2]:
            img_path = os.path.join(object_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, image_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img_to_array(img) / 255.0  # Normalize
                preprocessed_images.append(img)
                labels.append(0)

    return np.array(preprocessed_images), np.array(labels)

# Load and show preprocessed images
pre_images, labels = load_preprocessed_data(data_dir)
plot_images(pre_images, "Images After Preprocessing")

print(f"Total Preprocessed Images: {len(pre_images)}")
