import cv2
import numpy as np
import os

MODEL_PATH = "models/eigen_model.yml"

def train_eigen_recognizer(X_train, y_train, save_path=MODEL_PATH):
    """
    Trains an EigenFace recognizer and saves the model.

    Args:
        X_train (np.array): Flattened training images (grayscale).
        y_train (np.array): Corresponding labels for training images.
        save_path (str): Path to save the trained model.

    Returns:
        cv2.face_FaceRecognizer: The trained EigenFace recognizer, or None if training fails.
    """
    if X_train is None or y_train is None or X_train.size == 0 or y_train.size == 0:
        print("Error: Training data or labels are empty for EigenFaces.")
        return None
    if len(np.unique(y_train)) < 2:
        print("Error: EigenFaces requires at least two different classes (persons) for training.")
        print(f"Received {len(np.unique(y_train))} unique labels: {np.unique(y_train)}")
        return None
        
    print(f"Training EigenFace Recognizer with {X_train.shape[0]} samples...")
    eigen_recognizer = cv2.face.EigenFaceRecognizer_create()
    eigen_recognizer.train(X_train, y_train)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    eigen_recognizer.save(save_path)
    print(f"EigenFace model saved to {save_path}")
    return eigen_recognizer

def predict_eigen(model, X_test):
    """
    Predicts labels for test images using a trained EigenFace model.

    Args:
        model (cv2.face_FaceRecognizer): The trained EigenFace recognizer.
        X_test (np.array): Flattened test images (grayscale).

    Returns:
        list: A list of predicted labels.
    """
    if model is None:
        print("Error: EigenFace model is not loaded for prediction.")
        return []
    if X_test is None or X_test.size == 0:
        print("Warning: Test data is empty for EigenFace prediction.")
        return []

    y_pred = []
    print(f"Predicting with EigenFaces on {X_test.shape[0]} samples...")
    for img_array in X_test:
        label, confidence = model.predict(img_array)
        y_pred.append(label)
    return y_pred

def load_eigen_model(load_path=MODEL_PATH):
    """
    Loads a trained EigenFace model from a file.

    Args:
        load_path (str): Path to the saved model file.

    Returns:
        cv2.face_FaceRecognizer: The loaded model, or None if loading fails.
    """
    if not os.path.exists(load_path):
        print(f"Error: EigenFace model file not found at {load_path}")
        return None
    eigen_recognizer = cv2.face.EigenFaceRecognizer_create()
    eigen_recognizer.read(load_path)
    print(f"EigenFace model loaded from {load_path}")
    return eigen_recognizer

if __name__ == '__main__':
    # This is a placeholder for testing the module directly.
    # It requires preprocessed data.
    from src.data_preprocessing import load_data_from_processed_dir, TRAIN_DIR, TEST_DIR

    print("Testing EigenFace Recognizer module...")

    if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
        print(f"Train ({TRAIN_DIR}) or Test ({TEST_DIR}) directory not found.")
        print("Please run data_preprocessing.py first to create these directories.")
        # As a fallback for quick testing if dirs are missing, create dummy data
        # This is NOT a substitute for actual preprocessing
        print("Creating dummy data for EigenFaces test...")
        # Ensure at least 2 classes (persons) and enough samples per class
        dummy_X_train = np.array([np.random.randint(0, 256, 160*160, dtype=np.uint8) for _ in range(20)])
        dummy_y_train = np.array([0]*10 + [1]*10) # 2 persons, 10 images each
        
        dummy_X_test = np.array([np.random.randint(0, 256, 160*160, dtype=np.uint8) for _ in range(4)])
        dummy_y_test = np.array([0,1,0,1]) # Test images for both persons
        train_person_names_dummy = ['personA', 'personB']
        train_label_map_dummy = {'personA':0, 'personB':1}

        X_train, y_train = dummy_X_train, dummy_y_train
        X_test, y_test = dummy_X_test, dummy_y_test
        print(f"Using dummy data: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples.")

    else:
        print(f"Loading data from {TRAIN_DIR} and {TEST_DIR}...")
        X_train, y_train, train_person_names, train_label_map = load_data_from_processed_dir(TRAIN_DIR)
        X_test, y_test, _, _ = load_data_from_processed_dir(TEST_DIR)

        if X_train.size == 0 or X_test.size == 0:
            print("Failed to load data. Exiting EigenFaces test.")
            exit()
        if len(np.unique(y_train)) < 2:
            print(f"EigenFaces requires at least 2 people for training. Found {len(np.unique(y_train))}. Exiting.")
            exit()

    # Train the model
    eigen_model = train_eigen_recognizer(X_train, y_train)

    if eigen_model:
        # Load the model (as if from a different session)
        loaded_eigen_model = load_eigen_model()
        if loaded_eigen_model:
            # Make predictions
            predictions = predict_eigen(loaded_eigen_model, X_test)
            if predictions:
                print(f"EigenFaces Predictions: {predictions}")
                if y_test.size > 0:
                    print(f"EigenFaces True Labels: {y_test.tolist()}")
                    accuracy = np.mean(np.array(predictions) == y_test) * 100
                    print(f"EigenFaces Dummy Accuracy: {accuracy:.2f}%")
        else:
            print("Failed to load EigenFace model for testing.")
    else:
        print("Failed to train EigenFace model.") 