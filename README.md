# Face Verification System

## Overview

The Face Verification System is designed to enhance financial inclusion by verifying the identity of individuals using facial recognition technology. This system leverages a Django application with an API endpoint to verify users by comparing facial features extracted from images. The solution incorporates **MTCNN** for face detection, **FaceNet** embeddings for feature extraction, and a **Support Vector Machine (SVM)** for classification.

## Architecture

### 1. Django Application Interface
- A simple Django app was developed with a user-friendly interface.
- The app takes the image path as input to verify the identity of the person in the image.
- The image path is sent to an endpoint (`verify_image`) which processes the image and returns the name of the person detected.

### 2. Face Detection and Embedding
- **MTCNN (Multi-task Cascaded Convolutional Networks)** is used for face detection.
  - MTCNN detects faces in images, returning bounding box coordinates for each detected face.
- **FaceNet Embeddings**:
  - Once faces are detected, **FaceNet** is used to extract facial embeddings (feature vectors) from each face.
  - FaceNet is a deep learning model that generates a fixed-length vector representation for each face, capturing the unique features of the individual.

### 3. Model Training
- **Support Vector Machine (SVM)** was used as the classification model to predict the identity of the person in the image.
- Each image is labeled with the name of the person it contains.
- **Label Encoding** is used to transform the names into integer labels suitable for model training.
  - After prediction, the integer labels are decoded back to names to provide the correct identity.
- **Accuracy** is chosen as the evaluation metric to assess the performance of the SVM model. It measures the percentage of correct predictions out of the total predictions.

### 4. Model Saving
- After training, the SVM model, label encoder, and FaceNet model are saved for future use.
- The trained models are serialized and stored so they can be loaded in the endpoint for real-time image verification.

### 5. Django Endpoint
- The `verify_image` endpoint is developed to process incoming image paths.
  - The endpoint:
    - Loads the trained models.
    - Processes the image using MTCNN to detect faces.
    - Extracts face embeddings using FaceNet.
    - Classifies the embedding using the trained SVM model.
    - Returns the predicted name of the person in the image.

## Model Evaluation
- The SVM model was evaluated using **accuracy**, which provides an indication of how well the model performs in predicting the correct identity.
- The training dataset consisted of labeled images of individuals, and the model was tested on new images to assess its generalization capability.

## Guide to Running the Pipeline

### 1. Create a virtual environment
```bash
python3 -m venv venv
```

### 2. Activate the virtual environment
- On Windows:
```bash
venv\Scripts\activate
```
- On macOS/Linux:
```bash
source venv/bin/activate
```

### 3. Install the required Python packages
```bash
pip install -r requirements.txt
```

### 4. Run the Django development server
```bash
python manage.py runserver
```

### 5. Access the web application
Once the server is running, you can access the web application via your browser. By default, the application will be accessible at:
```
http://127.0.0.1:8000
```

### 6. Enter the image path
On the homepage, you will see an input box asking for the image path.

### 7. Verify the image
Enter the path to the image you wish to verify (e.g., `uploads/sample_image.jpg`).

### 8. Click the "Verify" button
Click the "Verify" button to send the image path to the backend, which will process the image and return the predicted identity.

## Conclusion

This Face Verification System provides an efficient and scalable solution for verifying identities using facial recognition. It leverages state-of-the-art deep learning models and is deployed as a web service to perform real-time verification.