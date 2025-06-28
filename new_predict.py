import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model('temperature_prediction_model_2.h5', compile=False)

# Load and preprocess the image
def preprocess_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype("float32") / 255.0
    return image

# Predict temperature and elephant presence
def predict_temperature_and_presence(image_path):
    try:
        image = preprocess_image(image_path)
        predicted_temp = model.predict(image)[0][0]
        #Validate temperature range
        if not 34.0 <= predicted_temp <= 40.0:
            print(f"Warning: Predicted temperature {predicted_temp:.2f}°C is out of expected range (34.0–40.0°C)")
            elephant_present = False
        else:
            elephant_present = predicted_temp >= 37.0  # Adjusted threshold
        print(f"Predicted Temperature: {predicted_temp:.2f}°C")
        print("Elephant Detected ✅" if elephant_present else "No Elephant ❌")
        return predicted_temp, elephant_present
    except ValueError as e:
        print(f"Error: {str(e)}")
        return None, None

    # Elephant presence logic based on temperature
   # elephant_present = predicted_temp >= 38.0  # adjust threshold if needed
    
    #print(f"Predicted Temperature: {predicted_temp:.2f}°C")
    #print("Elephant Detected ✅" if elephant_present else "No Elephant ❌")
    #return predicted_temp, elephant_present

# Example usage
#img_path = r'F:\human-wildlife-conflict-master\human-wildlife-conflict-master\Elephant\Object\human\00010_16082019_1440_1445_CAM2_2.5b.png'  # Change path accordingly
#predict_temperature_and_presence(img_path)
