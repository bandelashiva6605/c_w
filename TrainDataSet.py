import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# Define the path to the directory containing the "crop" and "weed" subdirectories
dataset_path = "dataset"

# Initialize empty lists to store image data (X) and labels (Y)
X = []
Y = []

# Load the class labels from the "classes.txt" file
class_file_path = os.path.join(dataset_path, "classes.txt")
with open(class_file_path, "r") as class_file:
    class_labels = class_file.read().splitlines()

# Use label encoding to map class names to numeric labels
label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(class_labels)

# Iterate through the subdirectories ("crop" and "weed")
for class_label, class_name in enumerate(["crop", "weed"]):
    class_dir = os.path.join(dataset_path, class_name)
    
    # Iterate through image files in each class subdirectory
    for filename in os.listdir(class_dir):
        if filename.endswith(".jpg"):  # Adjust the file extension as needed
            try:
                # Load and preprocess the image (e.g., resizing, normalization)
                image = Image.open(os.path.join(class_dir, filename))
                image = image.resize((64, 64))  # Resize to a common size
                image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]

                # Append the image data to X
                X.append(image)

                # Assign labels from numeric_labels to Y based on the class label
                Y.append(class_label)

            except Exception as e:
                print(f"Error loading image {filename}: {e}")

# Convert the lists of images and labels to NumPy arrays
X = np.array(X)
Y = np.array(Y)

# Save the NumPy arrays to files
np.save('X.npy', X)
np.save('Y.npy', Y)

print(f"Loaded {len(X)} images and labels.")
print("X.npy and Y.npy files created.")