import numpy as np
import os
import cv2
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.pipeline import make_pipeline
import joblib
import tensorflow as tf # Using tensorflow for Keras FaceNet model

# Define paths
FACENET_ARCH_PATH = os.path.join("models", "facenet_model.json")
FACENET_WEIGHTS_PATH = os.path.join("models", "facenet_model_weights.h5")
FACENET_MODEL_PATH = os.path.join("models", "facenet_keras.h5") # This is the primary .h5 file we were using
FACENET_ALT_MODEL_PATH = os.path.join("models", "facenet_alternate_model.h5") # For the other model.h5
CLASSIFIER_MODEL_PATH = os.path.join("models", "facenet_classifier.pkl")

# Expected embedding size from FaceNet
EMBEDDING_SIZE = 128 # For InceptionResnetV1 based FaceNet, sometimes 512 for others
IMG_SIZE = (160, 160) # FaceNet typically expects 160x160 or 96x96

def load_facenet_keras_model(): # Removed model_path default argument to control attempts internally
    """Loads the pre-trained FaceNet Keras model using multiple strategies."""
    original_tf_log_level = tf.get_logger().level
    tf.get_logger().setLevel('ERROR') # Suppress informational TF logs
    model = None
    print("Attempting to load FaceNet model...")

    # Attempt 1: Load from JSON architecture and H5 weights
    if os.path.exists(FACENET_ARCH_PATH) and os.path.exists(FACENET_WEIGHTS_PATH):
        print(f"Strategy 1: Trying to load model from JSON ({FACENET_ARCH_PATH}) and weights ({FACENET_WEIGHTS_PATH})...")
        try:
            with open(FACENET_ARCH_PATH, 'r') as json_file:
                loaded_model_json = json_file.read()
            model = tf.keras.models.model_from_json(loaded_model_json)
            model.load_weights(FACENET_WEIGHTS_PATH)
            print("FaceNet model loaded successfully from JSON architecture and H5 weights.")
        except Exception as e:
            print(f"Failed to load model from JSON/weights. Exception type: {type(e)}") # Enhanced error print
            model = None # Ensure model is None if this attempt fails
    else:
        print(f"Strategy 1: JSON architecture ({FACENET_ARCH_PATH}) or weights ({FACENET_WEIGHTS_PATH}) not found.")

    # Attempt 2: Load from primary H5 file (facenet_keras.h5)
    if model is None and os.path.exists(FACENET_MODEL_PATH):
        print(f"Strategy 2: Trying to load model from primary H5 file ({FACENET_MODEL_PATH})...")
        try:
            model = tf.keras.models.load_model(FACENET_MODEL_PATH, compile=False)
            print(f"FaceNet model loaded successfully from {FACENET_MODEL_PATH}.")
        except Exception as e:
            print(f"Failed to load model from {FACENET_MODEL_PATH}: {e}")
            model = None
    elif model is None:
        print(f"Strategy 2: Primary H5 model file ({FACENET_MODEL_PATH}) not found.")

    # Attempt 3: Load from alternate H5 file (facenet_alternate_model.h5)
    if model is None and os.path.exists(FACENET_ALT_MODEL_PATH):
        print(f"Strategy 3: Trying to load model from alternate H5 file ({FACENET_ALT_MODEL_PATH})...")
        try:
            model = tf.keras.models.load_model(FACENET_ALT_MODEL_PATH, compile=False)
            print(f"FaceNet model loaded successfully from {FACENET_ALT_MODEL_PATH}.")
        except Exception as e:
            print(f"Failed to load model from {FACENET_ALT_MODEL_PATH}: {e}")
            model = None
    elif model is None:
        print(f"Strategy 3: Alternate H5 model file ({FACENET_ALT_MODEL_PATH}) not found.")

    if model is not None:
        if model.output_shape[-1] != EMBEDDING_SIZE:
            tf.get_logger().setLevel(original_tf_log_level) # Reset logger level
            print(f"Warning: Loaded FaceNet model output dimension ({model.output_shape[-1]}) does not match expected EMBEDDING_SIZE ({EMBEDDING_SIZE}).")
            print("This may cause issues if the downstream classifier expects a different embedding size.")
        return model
    else:
        tf.get_logger().setLevel(original_tf_log_level) # Reset logger level
        print("Error: All attempts to load the FaceNet Keras model failed.")
        return None

def get_embedding(facenet_model, face_pixels):
    """
    Calculates the embedding for a single face image.
    face_pixels: A single face image (numpy array, HxWxC, float32, preprocessed).
    """
    # Ensure the input is the correct shape (1, height, width, channels)
    if face_pixels.ndim == 3:
        face_pixels = np.expand_dims(face_pixels, axis=0)
    
    # Standardize pixel values (0-1 range, as typically expected by FaceNet)
    # This assumes input images are 0-255 uint8. If they are already float, adjust accordingly.
    if face_pixels.dtype == np.uint8:
        face_pixels = face_pixels.astype('float32')
        face_pixels = (face_pixels - 127.5) / 128.0 # Normalize to [-1, 1] which some FaceNet versions expect
        # Alternatively, simple / 255.0 if model expects [0,1]
        # It's crucial to match the preprocessing of the original FaceNet training.
        # Most Keras FaceNet implementations (like nyoki-mtl's) use a Lambda layer for this internally or expect input in [0,1] or [-1,1].
        # The facenet_keras.h5 from nyoki-mtl expects input pixels to be standardized (mean and std dev normalization).
        # For simplicity here, we will normalize to [0,1] if it's uint8, assuming the model handles further normalization or expects this range.
        # A common approach: (img - img.mean()) / img.std()
        # Or, if trained on images scaled to [0,1]: img / 255.0
        # Or, if trained on images scaled to [-1,1]: (img - 127.5) / 127.5

        # Let's assume the facenet_keras.h5 model from nyoki-mtl has a Lambda layer for standardization.
        # If not, manual standardization is needed here.
        # For now, we scale to [0,1] as a common pre-step.
        face_pixels = face_pixels / 255.0

    embedding = facenet_model.predict(face_pixels)
    return embedding[0] # Returns a 1D array of size EMBEDDING_SIZE

def generate_embeddings(facenet_model, X_data):
    """
    Generates embeddings for a list of face images.
    X_data: list or np.array of face images (HxWxC).
    """
    embeddings = []
    if X_data is None or len(X_data) == 0:
        return np.array([])
        
    print(f"Generating FaceNet embeddings for {len(X_data)} images...")
    for i, face_img in enumerate(X_data):
        if face_img.shape != (*IMG_SIZE, 3):
             # print(f"Warning: Image at index {i} has shape {face_img.shape}, resizing to {(*IMG_SIZE, 3)} for FaceNet.")
             face_img = cv2.resize(face_img, IMG_SIZE)
        # Ensure it's RGB if loaded by OpenCV (BGR)
        # Data loader should provide RGB, but double check or convert here if necessary.
        # Assuming `load_data_for_facenet` in `data_preprocessing.py` already handles BGR to RGB via cv2.imread then potentially cv2.cvtColor.
        # If not, and images are BGR: face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        embeddings.append(get_embedding(facenet_model, face_img))
        if (i+1) % 100 == 0:
            print(f"  Processed {i+1}/{len(X_data)} images for embeddings.")
    return np.asarray(embeddings)

def train_facenet_classifier(X_train_embeddings, y_train, save_path=CLASSIFIER_MODEL_PATH):
    """
    Trains an SVM classifier on FaceNet embeddings.
    """
    if X_train_embeddings is None or y_train is None or X_train_embeddings.size == 0 or y_train.size == 0:
        print("Error: Embeddings or labels are empty for FaceNet classifier training.")
        return None
    if len(np.unique(y_train)) < 1: # Technically SVM can train on 1 class, but it's not useful for recognition
        print("Error: FaceNet classifier training requires at least one class. Multiple classes are needed for recognition.")
        if len(np.unique(y_train)) < 2:
             print(f"Warning: Training FaceNet classifier with only {len(np.unique(y_train))} class(es). Predictions will be limited.")
        # return None # Allowing training on single class for now, though evaluation will be poor.

    print(f"Training SVM classifier on {X_train_embeddings.shape[0]} FaceNet embeddings...")
    
    # Normalize embeddings (L2 normalization is common for FaceNet)
    in_encoder = Normalizer(norm='l2')
    X_train_norm = in_encoder.transform(X_train_embeddings)

    # Encode labels
    out_encoder = LabelEncoder()
    out_encoder.fit(y_train)
    y_train_encoded = out_encoder.transform(y_train)

    # Train SVM
    # Added probability=True for predict_proba if needed by evaluation metrics later
    model = SVC(kernel='linear', probability=True, C=1.0, random_state=42) 
    model.fit(X_train_norm, y_train_encoded)

    # Save the label encoder and the SVM model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump({'svm_model': model, 'label_encoder': out_encoder, 'normalizer': in_encoder}, save_path)
    print(f"FaceNet classifier (SVM, LabelEncoder, Normalizer) saved to {save_path}")
    return model, out_encoder, in_encoder # Return all components

def load_facenet_classifier(load_path=CLASSIFIER_MODEL_PATH):
    """
    Loads the trained SVM classifier, label encoder, and normalizer.
    """
    if not os.path.exists(load_path):
        print(f"Error: FaceNet classifier model file not found at {load_path}")
        return None, None, None
    try:
        data = joblib.load(load_path)
        print(f"FaceNet classifier loaded from {load_path}")
        return data['svm_model'], data['label_encoder'], data['normalizer']
    except Exception as e:
        print(f"Error loading FaceNet classifier: {e}")
        return None, None, None

def predict_facenet(facenet_model, classifier_components, X_test_images):
    """
    Predicts labels for test images using FaceNet embeddings and the SVM classifier.
    classifier_components: tuple (svm_model, label_encoder, normalizer)
    X_test_images: np.array of raw test images (not embeddings yet).
    """
    svm_model, label_encoder, normalizer = classifier_components
    if facenet_model is None or svm_model is None or label_encoder is None or normalizer is None:
        print("Error: One or more FaceNet components (FaceNet model, SVM, LabelEncoder, Normalizer) are not loaded.")
        return []
    if X_test_images is None or len(X_test_images) == 0:
        print("Warning: Test images are empty for FaceNet prediction.")
        return []

    print(f"Generating embeddings for {len(X_test_images)} test images for FaceNet prediction...")
    X_test_embeddings = generate_embeddings(facenet_model, X_test_images)
    if X_test_embeddings.size == 0:
        print("Error: Failed to generate embeddings for test images.")
        return []

    X_test_norm = normalizer.transform(X_test_embeddings)
    
    print("Predicting with SVM on FaceNet embeddings...")
    y_pred_encoded = svm_model.predict(X_test_norm)
    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)
    
    # Return original numeric labels if that's what other models use for comparison
    # This depends on how y_true is structured for evaluation.
    # For now, returning the encoded labels as integer predictions.
    return y_pred_encoded # These are the integer labels corresponding to original y values


if __name__ == '__main__':
    from src.data_preprocessing import load_data_for_facenet, TRAIN_DIR, TEST_DIR, IMG_SIZE

    print("Testing FaceNet Recognition module...")

    # 1. Load FaceNet Keras model
    keras_facenet = load_facenet_keras_model()
    if keras_facenet is None:
        print("Failed to load Keras FaceNet model. Cannot proceed with FaceNet testing.")
        exit()

    # 2. Load preprocessed data (RGB images)
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
        print(f"Train ({TRAIN_DIR}) or Test ({TEST_DIR}) directory not found.")
        print("Please run data_preprocessing.py first.")
        # Create dummy data for FaceNet if dirs are missing (RGB images)
        print("Creating dummy RGB data for FaceNet test...")
        # (batch, height, width, channels)
        dummy_X_train_rgb = np.random.randint(0, 256, (20, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
        dummy_y_train_labels = np.array([0]*10 + [1]*10) 
        dummy_X_test_rgb = np.random.randint(0, 256, (4, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
        dummy_y_test_labels = np.array([0,1,0,1])

        X_train_imgs, y_train = dummy_X_train_rgb, dummy_y_train_labels
        X_test_imgs, y_test = dummy_X_test_rgb, dummy_y_test_labels
        train_person_names_dummy = ['personA', 'personB'] # Not directly used here but good for context
        train_label_map_dummy = {'personA':0, 'personB':1}
        print(f"Using dummy RGB data: {X_train_imgs.shape[0]} train, {X_test_imgs.shape[0]} test.")
    else:
        print(f"Loading RGB data from {TRAIN_DIR} and {TEST_DIR} for FaceNet...")
        X_train_imgs, y_train, train_person_names, train_label_map = load_data_for_facenet(TRAIN_DIR, img_size=IMG_SIZE)
        X_test_imgs, y_test, _, _ = load_data_for_facenet(TEST_DIR, img_size=IMG_SIZE)

        if X_train_imgs.size == 0 or X_test_imgs.size == 0:
            print("Failed to load RGB data for FaceNet. Exiting test.")
            exit()
        if len(np.unique(y_train)) < 1:
            print(f"FaceNet classifier needs at least 1 person for training. Found {len(np.unique(y_train))}. Exiting.")
            exit()

    # 3. Generate embeddings for training data
    # Note: X_train_imgs are raw images, not paths
    X_train_embeddings = generate_embeddings(keras_facenet, X_train_imgs)
    if X_train_embeddings.size == 0:
        print("Failed to generate training embeddings. Exiting FaceNet test.")
        exit()

    # 4. Train a classifier on the embeddings
    # train_facenet_classifier returns (svm_model, label_encoder, normalizer)
    classifier_tuple = train_facenet_classifier(X_train_embeddings, y_train)
    if classifier_tuple is None or classifier_tuple[0] is None:
        print("Failed to train FaceNet classifier. Exiting FaceNet test.")
        exit()

    # 5. Load the classifier (as if in a new session)
    loaded_classifier_tuple = load_facenet_classifier()
    if loaded_classifier_tuple is None or loaded_classifier_tuple[0] is None:
        print("Failed to load FaceNet classifier. Exiting FaceNet test.")
        exit()

    # 6. Make predictions on test images (not embeddings)
    # The predict_facenet function will internally generate embeddings for X_test_imgs
    predictions_encoded = predict_facenet(keras_facenet, loaded_classifier_tuple, X_test_imgs)

    if predictions_encoded:
        # The predictions are already encoded labels. y_test contains the original numeric labels.
        print(f"FaceNet Predictions (encoded): {predictions_encoded}")
        if y_test.size > 0:
            print(f"FaceNet True Labels: {y_test.tolist()}") # y_test should be comparable directly
            accuracy = np.mean(np.array(predictions_encoded) == y_test) * 100
            print(f"FaceNet SVM Accuracy (on encoded labels vs original numeric labels): {accuracy:.2f}%")
            
            # If you need to see the actual names:
            # label_encoder_from_tuple = loaded_classifier_tuple[1]
            # predicted_names = label_encoder_from_tuple.inverse_transform(predictions_encoded)
            # true_names = [train_person_names[i] for i in y_test] # Assuming y_test aligns with train_person_names mapping
            # print(f"Predicted Person Names: {predicted_names}")
            # print(f"True Person Names: {true_names}")
    else:
        print("FaceNet prediction failed or returned no results.") 