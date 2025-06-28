import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset Path
data_dir = 'Elephant'
image_size = (224, 224)

# --- 1. Load images and generate random temperatures ---
def load_data_with_temperature(data_dir):
    data = []
    temperature = []

    for object_label in ['single_elephant', 'multiple_separate_elephants', 
                         'multiple_obstructing_elephants', 'human_and_elephant']:
        object_dir = os.path.join(data_dir, 'Object', object_label)
        for img_file in os.listdir(object_dir):
            img_path = os.path.join(object_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, image_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                img = img_to_array(img)
                data.append(img)
                temperature.append(np.random.uniform(37.5, 39.0))  # Elephant temperatures
            #img = cv2.resize(img, image_size)
            #img = img_to_array(img)
            #data.append(img)
            #temperature.append(np.random.uniform(38.0, 40.5))  # Elephant-like temperatures

    for object_label in ['human', 'goat']:
        object_dir = os.path.join(data_dir, 'Object', object_label)
        for img_file in os.listdir(object_dir):
            img_path = os.path.join(object_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, image_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                img = img_to_array(img)
                data.append(img)
                temperature.append(np.random.uniform(34.0, 36.0))  # Non-elephant temperatures
            #img = cv2.resize(img, image_size)
            #img = img_to_array(img)
            #data.append(img)
            #temperature.append(np.random.uniform(36.0, 37.5))  # Non-elephant temperatures

    data = np.array(data, dtype="float32") / 255.0
    temperature = np.array(temperature, dtype="float32")

    return data, temperature

data, temperature = load_data_with_temperature(data_dir)
X_train, X_test, y_train_temp, y_test_temp = train_test_split(data, temperature, 
                                                              test_size=0.2, random_state=42)

train_datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2,
                                   zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator()

# --- 2. Model for Temperature Prediction Only ---
input_img = Input(shape=(224, 224, 3))
x = Conv2D(32, (3, 3), activation='relu')(input_img)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
temperature_output = Dense(1, name='temperature')(x)

model = Model(inputs=input_img, outputs=temperature_output)

model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])

checkpoint = ModelCheckpoint('best_temp_model_2.h5', save_best_only=True, 
                             monitor='val_loss', mode='min', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, 
                               restore_best_weights=True, verbose=1)

model.fit(train_datagen.flow(X_train, y_train_temp, batch_size=32),
          epochs=10,
          validation_data=test_datagen.flow(X_test, y_test_temp, batch_size=32),
          callbacks=[checkpoint, early_stopping])

model.save('temperature_prediction_model_2.h5')
import matplotlib.pyplot as plt

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # MAE
    plt.subplot(1, 2, 1)
    plt.plot(history.history['mae'], label="Train MAE")
    plt.plot(history.history['val_mae'], label="Val MAE")
    plt.title("Mean Absolute Error Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label="Train Loss")
    plt.plot(history.history['val_loss'], label="Val Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_history(history)
