# import matplotlib.pyplot as plt
# from flask import Flask, render_template, request, redirect, url_for
# import numpy as np
# import os
# import cv2
# from tensorflow.keras.models import load_model
# from tensorflow.keras import backend as K
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib

# # Set Matplotlib to use the 'Agg' backend for non-interactive plotting
# matplotlib.use('Agg')

# # Initialize Flask application
# app = Flask(__name__)

# # Ensure the static folder exists
# if not os.path.exists('static'):
#     os.makedirs('static')

# # Define the Jaccard coefficient function for the model


# def jaccard_coef(y_true, y_pred):
#     y_true_flatten = K.flatten(y_true)
#     y_pred_flatten = K.flatten(y_pred)
#     intersection = K.sum(y_true_flatten * y_pred_flatten)
#     final_coef_value = (intersection + 1.0) / (K.sum(y_true_flatten) +
#                                                K.sum(y_pred_flatten) - intersection + 1.0)
#     return final_coef_value


# # Load the pre-trained model
# satellite_model = load_model(
#     './models/nice1.h5', custom_objects={'jaccard_coef': jaccard_coef})

# # Define the class colors as RGB numpy arrays (consistent with your original code)
# class_water = np.array([226, 169, 41])      # water
# class_land = np.array([132, 41, 246])       # land
# class_road = np.array([110, 193, 228])      # road
# class_building = np.array([60, 16, 152])    # building
# class_vegetation = np.array([254, 221, 58])  # vegetation
# class_unlabeled = np.array([155, 155, 155])  # unlabeled

# # Function to convert RGB image to label indices


# def rgb_to_label(label):
#     """
#     Convert an RGB image to a label segment with class indices (0 to 5).

#     Args:
#         label (np.ndarray): RGB image of shape (height, width, 3)

#     Returns:
#         np.ndarray: Label segment of shape (height, width) with values 0 to 5
#     """
#     label_segment = np.zeros(label.shape[:2], dtype=np.uint8)
#     label_segment[np.all(label == class_water, axis=-1)] = 0
#     label_segment[np.all(label == class_land, axis=-1)] = 1
#     label_segment[np.all(label == class_road, axis=-1)] = 2
#     label_segment[np.all(label == class_building, axis=-1)] = 3
#     label_segment[np.all(label == class_vegetation, axis=-1)] = 4
#     label_segment[np.all(label == class_unlabeled, axis=-1)] = 5
#     return label_segment

# # Function to convert label indices back to RGB colors


# def label_to_rgb(label_segment):
#     """
#     Convert a label segment to an RGB image using predefined class colors.

#     Args:
#         label_segment (np.ndarray): Array of shape (height, width) with values 0 to 5

#     Returns:
#         np.ndarray: RGB image of shape (height, width, 3)
#     """
#     height, width = label_segment.shape
#     rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
#     rgb_image[label_segment == 0] = class_water
#     rgb_image[label_segment == 1] = class_land
#     rgb_image[label_segment == 2] = class_road
#     rgb_image[label_segment == 3] = class_building
#     rgb_image[label_segment == 4] = class_vegetation
#     rgb_image[label_segment == 5] = class_unlabeled
#     return rgb_image

# # Define the image processing function


# def process_input_image(image_source):
#     """
#     Process the input image to generate a color-coded RGB prediction.

#     Args:
#         image_source (np.ndarray): Normalized image array of shape (256, 256, 3)

#     Returns:
#         np.ndarray: RGB image array of shape (256, 256, 3)
#     """
#     # Expand dimensions for model input
#     image = np.expand_dims(image_source, 0)
#     # Make prediction
#     prediction = satellite_model.predict(image)
#     # Get the class probabilities and convert to indices
#     predicted_indices = np.argmax(prediction, axis=3)[0, :, :]
#     # Convert indices to RGB colors
#     predicted_rgb = label_to_rgb(predicted_indices)
#     return predicted_rgb

# # Route for the main page


# @app.route('/')
# def index():
#     """Render the index page for uploading images."""
#     return render_template('index.html')

# # Route for handling image upload and prediction


# @app.route('/predict', methods=['POST'])
# def predict():
#     """Handle image upload, process it, and display the result with axes."""
#     if request.method == 'POST':
#         # Get the uploaded file
#         file = request.files['file']
#         if file:
#             # Define the path to save the uploaded image
#             uploaded_path = 'static/uploaded.png'
#             file.save(uploaded_path)

#             # Read the image in BGR format
#             image = cv2.imread(uploaded_path, cv2.IMREAD_COLOR)
#             # Resize to 256x256 (model input size)
#             image_resized = cv2.resize(image, (256, 256))

#             # Normalize the image for prediction
#             minmaxscaler = MinMaxScaler()
#             image_normalized = minmaxscaler.fit_transform(
#                 image_resized.reshape(-1, image_resized.shape[-1])).reshape(image_resized.shape)

#             # Generate the color-coded RGB prediction
#             predicted_rgb = process_input_image(image_normalized)

#             # Convert the resized image to RGB for plotting
#             image_resized_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

#             # Plot and save the original image with axes
#             fig, ax = plt.subplots()
#             ax.imshow(image_resized_rgb)
#             ax.set_title("Original Image")
#             ax.set_xticks(np.arange(0, 256, 50))
#             ax.set_yticks(np.arange(0, 256, 50))
#             plt.savefig('static/original_with_axes.png')
#             plt.close()

#             # Plot and save the predicted image with axes
#             fig, ax = plt.subplots()
#             ax.imshow(predicted_rgb)
#             ax.set_title("Predicted Image")
#             ax.set_xticks(np.arange(0, 256, 50))
#             ax.set_yticks(np.arange(0, 256, 50))
#             plt.savefig('static/predicted_with_axes.png')
#             plt.close()

#             # Render the result page with both images
#             return render_template('result.html')

#     # Redirect to index if something goes wrong
#     return redirect(url_for('index'))


# # Run the application
# if __name__ == '__main__':
#     app.run(debug=True)

'''2nd version of the code with patches'''
# import matplotlib.pyplot as plt
# from flask import Flask, render_template, request, redirect, url_for
# import numpy as np
# import os
# import cv2
# from tensorflow.keras.models import load_model
# from tensorflow.keras import backend as K
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib

# # Set Matplotlib to use the 'Agg' backend for non-interactive plotting
# matplotlib.use('Agg')

# # Initialize Flask application
# app = Flask(__name__)

# # Ensure the static folder exists
# if not os.path.exists('static'):
#     os.makedirs('static')

# # Define the Jaccard coefficient function for the model


# def jaccard_coef(y_true, y_pred):
#     y_true_flatten = K.flatten(y_true)
#     y_pred_flatten = K.flatten(y_pred)
#     intersection = K.sum(y_true_flatten * y_pred_flatten)
#     final_coef_value = (intersection + 1.0) / (K.sum(y_true_flatten) +
#                                                K.sum(y_pred_flatten) - intersection + 1.0)
#     return final_coef_value


# # Load the pre-trained model
# satellite_model = load_model(
#     './models/nice1.h5', custom_objects={'jaccard_coef': jaccard_coef})

# # Define the class colors as RGB numpy arrays (consistent with your original code)
# class_water = np.array([226, 169, 41])      # water
# class_land = np.array([132, 41, 246])       # land
# class_road = np.array([110, 193, 228])      # road
# class_building = np.array([60, 16, 152])    # building
# class_vegetation = np.array([254, 221, 58])  # vegetation
# class_unlabeled = np.array([155, 155, 155])  # unlabeled

# # Function to convert RGB image to label indices


# def rgb_to_label(label):
#     """
#     Convert an RGB image to a label segment with class indices (0 to 5).

#     Args:
#         label (np.ndarray): RGB image of shape (height, width, 3)

#     Returns:
#         np.ndarray: Label segment of shape (height, width) with values 0 to 5
#     """
#     label_segment = np.zeros(label.shape[:2], dtype=np.uint8)
#     label_segment[np.all(label == class_water, axis=-1)] = 0
#     label_segment[np.all(label == class_land, axis=-1)] = 1
#     label_segment[np.all(label == class_road, axis=-1)] = 2
#     label_segment[np.all(label == class_building, axis=-1)] = 3
#     label_segment[np.all(label == class_vegetation, axis=-1)] = 4
#     label_segment[np.all(label == class_unlabeled, axis=-1)] = 5
#     return label_segment

# # Function to convert label indices back to RGB colors


# def label_to_rgb(label_segment):
#     """
#     Convert a label segment to an RGB image using predefined class colors.

#     Args:
#         label_segment (np.ndarray): Array of shape (height, width) with values 0 to 5

#     Returns:
#         np.ndarray: RGB image of shape (height, width, 3)
#     """
#     height, width = label_segment.shape
#     rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
#     rgb_image[label_segment == 0] = class_water
#     rgb_image[label_segment == 1] = class_land
#     rgb_image[label_segment == 2] = class_road
#     rgb_image[label_segment == 3] = class_building
#     rgb_image[label_segment == 4] = class_vegetation
#     rgb_image[label_segment == 5] = class_unlabeled
#     return rgb_image

# # Function to prepare image into 256x256 patches


# def prepare_image_for_model(image, patch_size=256):
#     h, w = image.shape[:2]
#     # Calculate the number of patches
#     patches_h = (h + patch_size - 1) // patch_size
#     patches_w = (w + patch_size - 1) // patch_size

#     # Pad the image to fit into complete patches
#     pad_h = patches_h * patch_size - h
#     pad_w = patches_w * patch_size - w
#     padded_image = cv2.copyMakeBorder(
#         image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])

#     # Extract patches
#     patches = []
#     for i in range(patches_h):
#         for j in range(patches_w):
#             patch = padded_image[i * patch_size:(i + 1) *
#                                  patch_size, j * patch_size:(j + 1) * patch_size]
#             patches.append(patch)

#     return patches, padded_image.shape[:2], (patches_h, patches_w)

# # Function to stitch patches back into original image shape


# def stitch_patches(patches, original_shape, patch_shape, patches_h, patches_w):
#     h, w = original_shape
#     result = np.zeros((h, w, 3), dtype=np.uint8)
#     patch_size = patch_shape[0]

#     idx = 0
#     for i in range(patches_h):
#         for j in range(patches_w):
#             if i * patch_size < h and j * patch_size < w:
#                 result[i * patch_size:(i + 1) * patch_size, j *
#                        patch_size:(j + 1) * patch_size] = patches[idx]
#             idx += 1

#     # Crop back to original dimensions
#     return result[:h, :w]

# # Define the image processing function


# def process_input_image(image_patches):
#     predictions = []
#     for patch in image_patches:
#         # Normalize the patch
#         minmaxscaler = MinMaxScaler()
#         patch_normalized = minmaxscaler.fit_transform(
#             patch.reshape(-1, patch.shape[-1])).reshape(patch.shape)
#         # Expand dimensions for model input
#         patch_input = np.expand_dims(patch_normalized, 0)
#         # Make prediction
#         prediction = satellite_model.predict(patch_input)
#         # Get the class indices
#         predicted_indices = np.argmax(prediction, axis=3)[0, :, :]
#         # Convert indices to RGB colors
#         predicted_rgb = label_to_rgb(predicted_indices)
#         predictions.append(predicted_rgb)
#     return predictions

# # Route for the main page


# @app.route('/')
# def index():
#     """Render the index page for uploading images."""
#     return render_template('index.html')

# # Route for handling image upload and prediction


# @app.route('/predict', methods=['POST'])
# def predict():
#     """Handle image upload, process it, and display the result with axes."""
#     if request.method == 'POST':
#         # Get the uploaded file
#         file = request.files['file']
#         if file:
#             # Define the path to save the uploaded image
#             uploaded_path = 'static/uploaded.png'
#             file.save(uploaded_path)

#             # Read the image in BGR format
#             image = cv2.imread(uploaded_path, cv2.IMREAD_COLOR)
#             # Prepare image into patches without altering original ratio
#             patches, original_shape, (patches_h,
#                                       patches_w) = prepare_image_for_model(image)

#             # Process each patch
#             predictions = process_input_image(patches)

#             # Stitch the predictions back together
#             predicted_rgb = stitch_patches(
#                 predictions, original_shape, (256, 256), patches_h, patches_w)

#             # Convert the original image to RGB for plotting
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             # Plot and save the original image with axes
#             fig, ax = plt.subplots()
#             ax.imshow(image_rgb)
#             ax.set_title("Original Image")
#             ax.set_xticks(np.arange(0, image.shape[1], 50))
#             ax.set_yticks(np.arange(0, image.shape[0], 50))
#             ax.set_aspect('equal')
#             plt.savefig('static/original_with_axes.png')
#             plt.close()

#             # Plot and save the predicted image with axes
#             fig, ax = plt.subplots()
#             ax.imshow(predicted_rgb)
#             ax.set_title("Predicted Image")
#             ax.set_xticks(np.arange(0, image.shape[1], 50))
#             ax.set_yticks(np.arange(0, image.shape[0], 50))
#             ax.set_aspect('equal')
#             plt.savefig('static/predicted_with_axes.png')
#             plt.close()

#             # Render the result page with both images
#             return render_template('result.html')

#     # Redirect to index if something goes wrong
#     return redirect(url_for('index'))


# # Run the application
# if __name__ == '__main__':
#     app.run(debug=True)
'''3rd version'''
# from flask import Flask, render_template, request, redirect, url_for
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import cv2
# from tensorflow.keras.models import load_model
# from tensorflow.keras import backend as K
# import matplotlib
# from sklearn.preprocessing import MinMaxScaler

# # Set Matplotlib to use the 'Agg' backend for non-interactive plotting
# matplotlib.use('Agg')

# # Initialize Flask application
# app = Flask(__name__)

# # Ensure the static folder exists
# if not os.path.exists('static'):
#     os.makedirs('static')

# # Define the Jaccard coefficient function for the model


# def jaccard_coef(y_true, y_pred):
#     y_true_flatten = K.flatten(y_true)
#     y_pred_flatten = K.flatten(y_pred)
#     intersection = K.sum(y_true_flatten * y_pred_flatten)
#     final_coef_value = (intersection + 1.0) / (K.sum(y_true_flatten) +
#                                                K.sum(y_pred_flatten) - intersection + 1.0)
#     return final_coef_value


# # Load the pre-trained model
# satellite_model = load_model(
#     './models/nice1.h5', custom_objects={'jaccard_coef': jaccard_coef})

# # Define the class colors as RGB numpy arrays (consistent with your original code)
# class_water = np.array([226, 169, 41])      # water
# class_land = np.array([132, 41, 246])       # land
# class_road = np.array([110, 193, 228])      # road
# class_building = np.array([60, 16, 152])    # building
# class_vegetation = np.array([254, 221, 58])  # vegetation
# class_unlabeled = np.array([155, 155, 155])  # unlabeled

# # Function to convert RGB image to label indices


# def rgb_to_label(label):
#     """
#     Convert an RGB image to a label segment with class indices (0 to 5).

#     Args:
#         label (np.ndarray): RGB image of shape (height, width, 3)

#     Returns:
#         np.ndarray: Label segment of shape (height, width) with values 0 to 5
#     """
#     label_segment = np.zeros(label.shape[:2], dtype=np.uint8)
#     label_segment[np.all(label == class_water, axis=-1)] = 0
#     label_segment[np.all(label == class_land, axis=-1)] = 1
#     label_segment[np.all(label == class_road, axis=-1)] = 2
#     label_segment[np.all(label == class_building, axis=-1)] = 3
#     label_segment[np.all(label == class_vegetation, axis=-1)] = 4
#     label_segment[np.all(label == class_unlabeled, axis=-1)] = 5
#     return label_segment

# # Function to convert label indices back to RGB colors


# def label_to_rgb(label_segment):
#     """
#     Convert a label segment to an RGB image using predefined class colors.

#     Args:
#         label_segment (np.ndarray): Array of shape (height, width) with values 0 to 5

#     Returns:
#         np.ndarray: RGB image of shape (height, width, 3)
#     """
#     height, width = label_segment.shape
#     rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
#     rgb_image[label_segment == 0] = class_water
#     rgb_image[label_segment == 1] = class_land
#     rgb_image[label_segment == 2] = class_road
#     rgb_image[label_segment == 3] = class_building
#     rgb_image[label_segment == 4] = class_vegetation
#     rgb_image[label_segment == 5] = class_unlabeled
#     return rgb_image

# # Function to prepare image into 256x256 patches without resizing the original


# def prepare_image_for_model(image, patch_size=256):
#     h, w = image.shape[:2]
#     # Handle images smaller than patch_size
#     if h <= patch_size and w <= patch_size:
#         return [image], (h, w), (1, 1)

#     # Calculate the number of patches
#     patches_h = (h + patch_size - 1) // patch_size
#     patches_w = (w + patch_size - 1) // patch_size

#     # Pad the image to fit into complete patches
#     pad_h = patches_h * patch_size - h
#     pad_w = patches_w * patch_size - w
#     padded_image = cv2.copyMakeBorder(
#         image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])

#     # Extract patches
#     patches = []
#     for i in range(patches_h):
#         for j in range(patches_w):
#             patch = padded_image[i * patch_size:(i + 1) *
#                                  patch_size, j * patch_size:(j + 1) * patch_size]
#             patches.append(patch)

#     return patches, (h, w), (patches_h, patches_w)

# # Function to stitch patches back into original image shape


# def stitch_patches(patches, original_shape, patch_shape, patches_h, patches_w):
#     h, w = original_shape
#     result = np.zeros((h, w, 3), dtype=np.uint8)
#     patch_size = patch_shape[0]

#     idx = 0
#     for i in range(patches_h):
#         for j in range(patches_w):
#             if idx >= len(patches):
#                 break
#             patch_h_start = i * patch_size
#             patch_h_end = min((i + 1) * patch_size, h)
#             patch_w_start = j * patch_size
#             patch_w_end = min((j + 1) * patch_size, w)

#             if patch_h_end > patch_h_start and patch_w_end > patch_w_start:
#                 # Crop the patch to fit the original dimensions
#                 patch = patches[idx][:patch_h_end -
#                                      patch_h_start, :patch_w_end - patch_w_start]
#                 result[patch_h_start:patch_h_end,
#                        patch_w_start:patch_w_end] = patch
#             idx += 1

#     return result

# # Define the image processing function


# def process_input_image(image_patches):
#     predictions = []
#     for patch in image_patches:
#         # Normalize the patch
#         minmaxscaler = MinMaxScaler()
#         patch_normalized = minmaxscaler.fit_transform(
#             patch.reshape(-1, patch.shape[-1])).reshape(patch.shape)
#         # Expand dimensions for model input
#         patch_input = np.expand_dims(patch_normalized, 0)
#         # Make prediction
#         prediction = satellite_model.predict(patch_input)
#         # Get the class indices
#         predicted_indices = np.argmax(prediction, axis=3)[0, :, :]
#         # Convert indices to RGB colors
#         predicted_rgb = label_to_rgb(predicted_indices)
#         predictions.append(predicted_rgb)
#     return predictions

# # Route for the main page


# @app.route('/')
# def index():
#     """Render the index page for uploading images."""
#     return render_template('index.html')

# # Route for handling image upload and prediction


# @app.route('/predict', methods=['POST'])
# def predict():
#     """Handle image upload, process it, and display the result with axes."""
#     if request.method == 'POST':
#         # Get the uploaded file
#         file = request.files['file']
#         if file:
#             # Define the path to save the uploaded image
#             uploaded_path = 'static/uploaded.png'
#             file.save(uploaded_path)

#             # Read the image in BGR format without resizing
#             image = cv2.imread(uploaded_path, cv2.IMREAD_COLOR)
#             # Prepare image into patches without altering original ratio
#             patches, original_shape, (patches_h,
#                                       patches_w) = prepare_image_for_model(image)

#             # Process each patch
#             predictions = process_input_image(patches)

#             # Stitch the predictions back together
#             predicted_rgb = stitch_patches(
#                 predictions, original_shape, (256, 256), patches_h, patches_w)

#             # Convert the original image to RGB for plotting (no resizing)
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             # Plot and save the original image with axes
#             fig, ax = plt.subplots()
#             ax.imshow(image_rgb)
#             ax.set_title("Original Image")
#             ax.set_xticks(np.arange(0, image.shape[1], 50))
#             ax.set_yticks(np.arange(0, image.shape[0], 50))
#             ax.set_aspect('equal')
#             plt.savefig('static/original_with_axes.png')
#             plt.close()

#             # Plot and save the predicted image with axes
#             fig, ax = plt.subplots()
#             ax.imshow(predicted_rgb)
#             ax.set_title("Predicted Image")
#             ax.set_xticks(np.arange(0, image.shape[1], 50))
#             ax.set_yticks(np.arange(0, image.shape[0], 50))
#             ax.set_aspect('equal')
#             plt.savefig('static/predicted_with_axes.png')
#             plt.close()

#             # Render the result page with both images
#             return render_template('result.html')

#     # Redirect to index if something goes wrong
#     return redirect(url_for('index'))


# # Run the application
# if __name__ == '__main__':
#     app.run(debug=True)
'''4th version'''
from flask import Flask, render_template, request, redirect, url_for
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import matplotlib
from sklearn.preprocessing import MinMaxScaler

# Set Matplotlib to use the 'Agg' backend for non-interactive plotting
matplotlib.use('Agg')

# Initialize Flask application
app = Flask(__name__)

# Ensure the static folder exists
if not os.path.exists('static'):
    os.makedirs('static')

# Define the Jaccard coefficient function for the model


def jaccard_coef(y_true, y_pred):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_true_flatten * y_pred_flatten)
    final_coef_value = (intersection + 1.0) / (K.sum(y_true_flatten) +
                                               K.sum(y_pred_flatten) - intersection + 1.0)
    return final_coef_value


# Load the pre-trained model
satellite_model = load_model(
    './models/nice1.h5', custom_objects={'jaccard_coef': jaccard_coef})

# Define the class colors as RGB numpy arrays (consistent with your original code)
class_water = np.array([226, 169, 41])      # water
class_land = np.array([132, 41, 246])       # land
class_road = np.array([110, 193, 228])      # road
class_building = np.array([60, 16, 152])    # building
class_vegetation = np.array([254, 221, 58])  # vegetation
class_unlabeled = np.array([155, 155, 155])  # unlabeled

# Function to convert RGB image to label indices


def rgb_to_label(label):
    """
    Convert an RGB image to a label segment with class indices (0 to 5).

    Args:
        label (np.ndarray): RGB image of shape (height, width, 3)

    Returns:
        np.ndarray: Label segment of shape (height, width) with values 0 to 5
    """
    label_segment = np.zeros(label.shape[:2], dtype=np.uint8)
    label_segment[np.all(label == class_water, axis=-1)] = 0
    label_segment[np.all(label == class_land, axis=-1)] = 1
    label_segment[np.all(label == class_road, axis=-1)] = 2
    label_segment[np.all(label == class_building, axis=-1)] = 3
    label_segment[np.all(label == class_vegetation, axis=-1)] = 4
    label_segment[np.all(label == class_unlabeled, axis=-1)] = 5
    return label_segment

# Function to convert label indices back to RGB colors


def label_to_rgb(label_segment):
    """
    Convert a label segment to an RGB image using predefined class colors.

    Args:
        label_segment (np.ndarray): Array of shape (height, width) with values 0 to 5

    Returns:
        np.ndarray: RGB image of shape (height, width, 3)
    """
    height, width = label_segment.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_image[label_segment == 0] = class_water
    rgb_image[label_segment == 1] = class_land
    rgb_image[label_segment == 2] = class_road
    rgb_image[label_segment == 3] = class_building
    rgb_image[label_segment == 4] = class_vegetation
    rgb_image[label_segment == 5] = class_unlabeled
    return rgb_image

# Function to prepare image into 256x256 patches without resizing the original


def prepare_image_for_model(image, patch_size=256):
    h, w = image.shape[:2]
    # Handle images smaller than patch_size
    if h <= patch_size and w <= patch_size:
        return [image], (h, w), (1, 1)

    # Calculate the number of patches
    patches_h = (h + patch_size - 1) // patch_size
    patches_w = (w + patch_size - 1) // patch_size

    # Pad the image to fit into complete patches
    pad_h = patches_h * patch_size - h
    pad_w = patches_w * patch_size - w
    padded_image = cv2.copyMakeBorder(
        image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Extract patches
    patches = []
    for i in range(patches_h):
        for j in range(patches_w):
            patch = padded_image[i * patch_size:(i + 1) *
                                 patch_size, j * patch_size:(j + 1) * patch_size]
            patches.append(patch)

    return patches, (h, w), (patches_h, patches_w)

# Function to stitch patches back into original image shape


def stitch_patches(patches, original_shape, patch_shape, patches_h, patches_w):
    h, w = original_shape
    result = np.zeros((h, w, 3), dtype=np.uint8)
    patch_size = patch_shape[0]

    idx = 0
    for i in range(patches_h):
        for j in range(patches_w):
            if idx >= len(patches):
                break
            patch_h_start = i * patch_size
            patch_h_end = min((i + 1) * patch_size, h)
            patch_w_start = j * patch_size
            patch_w_end = min((j + 1) * patch_size, w)

            if patch_h_end > patch_h_start and patch_w_end > patch_w_start:
                # Crop the patch to fit the original dimensions
                patch = patches[idx][:patch_h_end -
                                     patch_h_start, :patch_w_end - patch_w_start]
                result[patch_h_start:patch_h_end,
                       patch_w_start:patch_w_end] = patch
            idx += 1

    return result

# Define the image processing function


def process_input_image(image_patches):
    predictions = []
    for patch in image_patches:
        # Normalize the patch
        minmaxscaler = MinMaxScaler()
        patch_normalized = minmaxscaler.fit_transform(
            patch.reshape(-1, patch.shape[-1])).reshape(patch.shape)
        # Expand dimensions for model input
        patch_input = np.expand_dims(patch_normalized, 0)
        # Make prediction
        prediction = satellite_model.predict(patch_input)
        # Get the class indices
        predicted_indices = np.argmax(prediction, axis=3)[0, :, :]
        # Convert indices to RGB colors
        predicted_rgb = label_to_rgb(predicted_indices)
        predictions.append(predicted_rgb)
    return predictions

# Route for the main page


@app.route('/')
def index():
    """Render the index page for uploading images."""
    return render_template('index.html')

# Route for handling image upload and prediction


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload, process it, and display the result without axes."""
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        if file:
            # Define the path to save the uploaded image
            uploaded_path = 'static/uploaded.png'
            file.save(uploaded_path)

            # Read the image in BGR format without resizing
            image = cv2.imread(uploaded_path, cv2.IMREAD_COLOR)
            # Prepare image into patches without altering original ratio
            patches, original_shape, (patches_h,
                                      patches_w) = prepare_image_for_model(image)

            # Process each patch
            predictions = process_input_image(patches)

            # Stitch the predictions back together
            predicted_rgb = stitch_patches(
                predictions, original_shape, (256, 256), patches_h, patches_w)

            # Convert the original image to RGB for plotting (no resizing)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Plot and save the original image without axes
            fig, ax = plt.subplots()
            ax.imshow(image_rgb)
            ax.set_title("Original Image")
            # Remove axes
            ax.axis('off')
            plt.savefig('static/original_with_axes.png')
            plt.close()

            # Plot and save the predicted image without axes
            fig, ax = plt.subplots()
            ax.imshow(predicted_rgb)
            ax.set_title("Predicted Image")
            # Remove axes
            ax.axis('off')
            plt.savefig('static/predicted_with_axes.png')
            plt.close()

            # Render the result page with both images
            return render_template('result.html')

    # Redirect to index if something goes wrong
    return redirect(url_for('index'))


# Run the application
if __name__ == '__main__':
    app.run(debug=True)
