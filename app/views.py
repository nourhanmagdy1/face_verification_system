from django.http import JsonResponse
import cv2 as cv
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
import pickle
from django.shortcuts import render



def verify_image_interface(request):
    """
    Render an interface to upload an image for face verification.
    """
    return render(request, 'verify_image.html')

def get_embedding(face_img):
    """
    Compute the embedding for a given face image using the FaceNet model.

    Args:
        face_img (numpy.ndarray): The face image in a NumPy array format.

    Returns:
        numpy.ndarray: A 1D array representing the embedding vector of the face.
    """
    embedder = FaceNet()
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)
    embeddings__= embedder.embeddings(face_img)
    return embeddings__[0]


def verify_image(request):
    """
    Verify the identity of a face in an image using a trained SVM classifier.

    This function loads a face image specified in the request, detects the face,
    preprocesses it, computes the embedding using FaceNet, and then uses a pre-trained
    SVM model to predict the identity of the face. The predicted name is returned
    as a JSON response.

    Args:
        request (HttpRequest): The HTTP request containing the image path in the 'Image_Path' parameter.

    Returns:
        JsonResponse: A JSON object containing the predicted name of the face.
                      Example: {'Face_Name': 'Predicted_Name'}
    """
    try:
        # Get image path
        image_path = request.GET.get('Image_Path', '')
        print(image_path)
        # Initialize face detector model
        detector = MTCNN()
        
        # Load trained classifier and name encoder
        with open('app/models/svm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('app/models/label_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)

        # Read and preprocess the image
        img = cv.imread(image_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x, y, w, h = detector.detect_faces(img)[0]['box']
        img = img[y: y+h, x: x+w]
        img = cv.resize(img, (160, 160))
        test_im = get_embedding(img)

        # Make the prediction
        prediction = model.predict([test_im])
        predicted_label = encoder.inverse_transform(prediction)
        return JsonResponse({'Face_Name': predicted_label[0]})

    except Exception as e:
        # Handle any exceptions (If the image is not readable)
        return JsonResponse({'error': str(e)}, status=500)
