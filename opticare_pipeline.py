from predict_eye_disease import predict_image
from load_description import load_description

image_path = "sample_eye.jpg"

disease = predict_image(image_path)
description = load_description(disease)

print("Predicted Disease:", disease)
print("\nExplanation:\n", description)
