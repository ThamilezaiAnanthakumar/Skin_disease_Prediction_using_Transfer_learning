import numpy as np
import cv2
from PIL import Image
import subprocess
import io
import tensorflow as tf

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="skin_disease_model_2.tflite")
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels for the skin diseases
class_labels = ['cellulitis', 'impetigo', 'athlete-foot', 'nail-fungus', 'ringworm',
                'cutaneous-larva-migrans', 'chickenpox', 'shingles']

def preprocess_image(image):
    # Resize image to match the model's input size (224x224)
    img = cv2.resize(np.array(image), (224, 224))
    img = img / 255.0  # Normalize the image
    return np.expand_dims(img, axis=0)

def predict_disease(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], processed_image)
    
    # Invoke the interpreter (run inference)
    interpreter.invoke()
    
    # Get the output tensor (predictions)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Get the predicted class
    predicted_class_index = np.argmax(output_data)
    predicted_class = class_labels[predicted_class_index]
    probability = output_data[0][predicted_class_index]
    
    return predicted_class, probability

def capture_image():
    # Use libcamera-still to capture an image and save it to a temporary file
    image_path = '/tmp/captured_image.jpg'
    subprocess.run(["libcamera-still", "-o", image_path])
    
    # Open the captured image using PIL
    image = Image.open(image_path)
    return image

def main():
    print("Starting the skin disease prediction application...")
    while True:
        # Capture an image from the Raspberry Pi Camera using libcamera
        print("Capturing image...")
        image = capture_image()
        
        # Show the captured image
        image.show()  # This will open the image using the default image viewer

        # Make a prediction
        predicted_class, probability = predict_disease(image)
        
        # Display the result
        print(f"Predicted Disease: {predicted_class}")
        print(f"Confidence: {probability:.2%}")
        
        # Optionally, prompt the user to continue or exit
        cont = input("Do you want to capture another image? (y/n): ")
        if cont.lower() != 'y':
            break

if __name__ == "__main__":
    main()
