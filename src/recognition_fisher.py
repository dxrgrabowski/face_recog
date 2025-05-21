import cv2
import numpy as np
import os

MODEL_PATH = "models/fisher_model.yml"

def train_fisher_recognizer(X_train, y_train, save_path=MODEL_PATH):
    """
    Trains a FisherFace recognizer and saves the model.

    Args:
        X_train (np.array): Flattened training images (grayscale).
        y_train (np.array): Corresponding labels for training images.
        save_path (str): Path to save the trained model.

    Returns:
        cv2.face_FaceRecognizer: The trained FisherFace recognizer, or None if training fails.
    """
    if X_train is None or y_train is None or X_train.size == 0 or y_train.size == 0:
        print("Error: Training data or labels are empty for FisherFaces.")
        return None
    if len(np.unique(y_train)) < 2:
        print("Error: FisherFaces requires at least two different classes (persons) for training.")
        print(f"Received {len(np.unique(y_train))} unique labels: {np.unique(y_train)}")
        return None

    print(f"Training FisherFace Recognizer with {X_train.shape[0]} samples...")
    fisher_recognizer = cv2.face.FisherFaceRecognizer_create()
    fisher_recognizer.train(X_train, y_train)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fisher_recognizer.save(save_path)
    print(f"FisherFace model saved to {save_path}")
    return fisher_recognizer

def predict_fisher(model, X_test):
    """
    Predicts labels for test images using a trained FisherFace model.

    Args:
        model (cv2.face_FaceRecognizer): The trained FisherFace recognizer.
        X_test (np.array): Flattened test images (grayscale).

    Returns:
        list: A list of predicted labels.
    """
    if model is None:
        print("Error: FisherFace model is not loaded for prediction.")
        return []
    if X_test is None or X_test.size == 0:
        print("Warning: Test data is empty for FisherFace prediction.")
        return []

    y_pred = []
    print(f"Predicting with FisherFaces on {X_test.shape[0]} samples...")
    for img_array in X_test:
        label, confidence = model.predict(img_array)
        y_pred.append(label)
    return y_pred

def load_fisher_model(load_path=MODEL_PATH):
    """
    Loads a trained FisherFace model from a file.

    Args:
        load_path (str): Path to the saved model file.

    Returns:
        cv2.face_FaceRecognizer: The loaded model, or None if loading fails.
    """
    if not os.path.exists(load_path):
        print(f"Error: FisherFace model file not found at {load_path}")
        return None
    fisher_recognizer = cv2.face.FisherFaceRecognizer_create()
    fisher_recognizer.read(load_path)
    print(f"FisherFace model loaded from {load_path}")
    return fisher_recognizer

if __name__ == '__main__':
    from src.data_preprocessing import load_data_from_processed_dir, TRAIN_DIR, TEST_DIR

    print("Testing FisherFace Recognizer module...")

    if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
        print(f"Train ({TRAIN_DIR}) or Test ({TEST_DIR}) directory not found.")
        print("Please run data_preprocessing.py first.")
        print("Creating dummy data for FisherFaces test...")
        dummy_X_train = np.array([np.random.randint(0, 256, 160*160, dtype=np.uint8) for _ in range(20)])
        dummy_y_train = np.array([0]*10 + [1]*10) 
        dummy_X_test = np.array([np.random.randint(0, 256, 160*160, dtype=np.uint8) for _ in range(4)])
        dummy_y_test = np.array([0,1,0,1])
        X_train, y_train = dummy_X_train, dummy_y_train
        X_test, y_test = dummy_X_test, dummy_y_test
        print(f"Using dummy data: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples.")
    else:
        print(f"Loading data from {TRAIN_DIR} and {TEST_DIR}...")
        X_train, y_train, _, _ = load_data_from_processed_dir(TRAIN_DIR)
        X_test, y_test, _, _ = load_data_from_processed_dir(TEST_DIR)

        if X_train.size == 0 or X_test.size == 0:
            print("Failed to load data. Exiting FisherFaces test.")
            exit()
        if len(np.unique(y_train)) < 2:
            print(f"FisherFaces requires at least 2 people for training. Found {len(np.unique(y_train))}. Exiting.")
            exit()

    fisher_model = train_fisher_recognizer(X_train, y_train)

    if fisher_model:
        loaded_fisher_model = load_fisher_model()
        if loaded_fisher_model:
            predictions = predict_fisher(loaded_fisher_model, X_test)
            if predictions:
                print(f"FisherFaces Predictions: {predictions}")
                if y_test.size > 0:
                    print(f"FisherFaces True Labels: {y_test.tolist()}")
                    accuracy = np.mean(np.array(predictions) == y_test) * 100
                    print(f"FisherFaces Dummy Accuracy: {accuracy:.2f}%")
        else:
            print("Failed to load FisherFace model for testing.")
    else:
        print("Failed to train FisherFace model.") 