# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np

# # Load the trained model
# model = load_model('elephant_detection_model.h5')


# def predict_image(img_path):
#     # Load the image and preprocess it
#     img = image.load_img(img_path, target_size=(224, 224))  # Resize to model input size
#     img = image.img_to_array(img)  # Convert image to array
#     img = np.expand_dims(img, axis=0) / 255.0  # Normalize the image

#     # Predict the class (elephant or non-elephant)
#     prediction = model.predict(img)

#     # If prediction > 0.5, it’s an elephant
#     return 'Elephant' if prediction[0] > 0.5 else 'Non-Elephant'

# # Test with a new image
# result = predict_image(r'Elephant\Object\goat\1970-01-01_02-21-19.442069.png')
# print(result)




from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('elephant_detection_temperature_model.h5')



def predict_image(img_path):
    # Load the image and preprocess it
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to model input size
    img = image.img_to_array(img)  # Convert image to array
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize the image

    # Predict the class (elephant or non-elephant) and the temperature
    elephant_prediction, temperature_prediction = model.predict(img)

    # Interpret elephant presence (if prediction > 0.5, it’s an elephant)
    elephant_label = 'Elephant' if elephant_prediction[0] > 0.5 else 'Non-Elephant'

    # Return both predictions
    return elephant_label, temperature_prediction[0][0]

# Test with a new image
result = predict_image(r'Elephant\Object\multiple_separate_elephants\2019-09-10_01-21-27.079944.png')
print(f"Prediction Result: {result[0]}, Predicted Temperature: {result[1]}°C")
