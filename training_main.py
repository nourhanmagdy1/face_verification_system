# Import packages
import cv2 as cv
import os
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
from tqdm import tqdm


class FACELOADING:
    def __init__(self, directory):
        """
        Initialize the FACELOADING class with the given directory.

        Parameters:
        - directory (str): Path to the directory containing subdirectories of face images.

        Attributes:
        - target_size (tuple): Desired size to resize face images (default is 160x160).
        - X (list): List to store face image arrays.
        - Y (list): List to store corresponding labels.
        """
        self.directory = directory
        self.target_size = (160,160)
        self.detector = MTCNN()
        self.embedder = FaceNet()
        self.X = []
        self.Y = []


    def extract_face(self, filename):
        """
        Extract a face from an image file using a face detector.

        Parameters:
        - filename (str): Path to the image file.

        Returns:
        - face_arr (ndarray): Resized face array extracted from the image.
        """
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x,y,w,h = self.detector.detect_faces(img)[0]['box']
        x,y = abs(x), abs(y)
        face = img[y:y+h, x:x+w]
        face_arr = cv.resize(face, self.target_size)
        return face_arr


    def load_faces(self, dir):
        """
        Load and extract faces from all images in a specified directory.

        Parameters:
        - dir (str): Path to the directory containing face images.

        Returns:
        - FACES (list): List of face image arrays extracted from the directory.
        """
        FACES = []
        for im_name in os.listdir(dir):
            try:
                path = dir + im_name
                single_face = self.extract_face(path)
                FACES.append(single_face)
            except Exception as e:
                pass
        return FACES


    def load_classes(self):
        """
        Load all face images and their corresponding labels from the directory.

        The directory should contain subdirectories, each named according to a class label.

        Returns:
        - X (ndarray): Array of face images.
        - Y (ndarray): Array of corresponding labels.
        """
        for sub_dir in tqdm(os.listdir(self.directory)):
            path = self.directory +'/'+ sub_dir+'/'
            FACES = self.load_faces(path)
            labels = [sub_dir for _ in range(len(FACES))]
            self.X.extend(FACES)
            self.Y.extend(labels)
        return np.asarray(self.X), np.asarray(self.Y)
   

    def get_embedding(self, face_img):
        """
        Generate an embedding for a given face image using a pre-trained embedder.

        Parameters:
        - face_img (ndarray): Face image array.

        Returns:
        - embedding (ndarray): Embedding vector for the face image.
        """
        face_img = face_img.astype('float32')
        face_img = np.expand_dims(face_img, axis=0)
        embeddings__= self.embedder.embeddings(face_img)
        return embeddings__[0]

# Initialize class
dataset_path = "./app/dataset/lfw-deepfunneled/lfw-deepfunneled"
face_loader = FACELOADING(dataset_path)
X, Y = face_loader.load_classes()

EMBEDDED_X = []
for img in X:
    EMBEDDED_X.append(face_loader.get_embedding(img))

EMBEDDED_X = np.asarray(EMBEDDED_X)
np.savez_compressed('./app/models/faces_embeddings_done_4classes.npz', EMBEDDED_X, Y)

# Initialize label encoder to encode faces names
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)

# Save label encoder to be used in API
with open('./app/models/label_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)


# Split data into 90% train and 10% test
X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, Y, test_size=0.1, shuffle=True, random_state=17)

model = SVC(kernel='linear', probability=True)

# Train model to classify images
model.fit(X_train, Y_train)

# Save the model to be used in API
with open('./app/models/svm_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Predict test samples and get accuracy score
ypreds_test = model.predict(X_test)
print(accuracy_score(Y_test, ypreds_test))




