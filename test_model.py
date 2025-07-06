import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

# Load TFLite model
interpreter = tflite.Interpreter(model_path='recyclable_classifier.tflite')
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess test image
img = Image.open('test_image.jpg').resize((224, 224))
img_array = np.array(img, dtype=np.float32) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Run inference
interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()
predictions = interpreter.get_tensor(output_details[0]['index'])

# Map predictions to classes
classes = ['ORGANIC', 'RECYCLABLE']
predicted_class = classes[np.argmax(predictions[0])]
print(f"Predicted class: {predicted_class}, Confidence: {np.max(predictions[0]):.2f}")


