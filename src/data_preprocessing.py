import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import kagglehub
from src.face_detection import detect_faces, get_largest_face

# Path to the FaceNet Keras model h5 file
FACENET_MODEL_PATH = os.path.join("models", "facenet_keras.h5") # Ensured consistent

# Fallback path for Haar Cascade XML
DEFAULT_CASCADE_PATH = os.path.join("models", "haarcascade_frontalface_default.xml")
CASCADE_PATH = os.environ.get("HAARCASCADE_PATH", DEFAULT_CASCADE_PATH)

# Define paths - RAW_DATA_PATH is now more of a conceptual/original target if not using Kaggle cache
DATA_DIR = "data"
# RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "Face Recognition dataset", "dataset") # This is more of a conceptual path now
# The actual raw data path will be determined by download_dataset() from kagglehub's cache or download location.
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed")
TRAIN_DIR = os.path.join(PROCESSED_DATA_PATH, "train")
TEST_DIR = os.path.join(PROCESSED_DATA_PATH, "test")
IMG_SIZE = (160, 160)

def download_dataset():
    """
    Downloads the image dataset and the FaceNet Keras model using kagglehub.
    Returns the path to the actual raw image data directory.
    """
    actual_image_data_path = None
    print("Downloading face image dataset (vasukipatel/face-recognition-dataset)...")
    try:
        # kagglehub.dataset_download returns the path to the downloaded dataset directory
        # For "vasukipatel/face-recognition-dataset", images are in a subdirectory.
        image_dataset_root_path = kagglehub.dataset_download("vasukipatel/face-recognition-dataset")
        print(f"Image dataset root downloaded to: {image_dataset_root_path}")
        
        # Construct the path to the actual images based on known dataset structure
        # The dataset "vasukipatel/face-recognition-dataset" has images under "Faces"
        actual_image_data_path = os.path.join(image_dataset_root_path, "Faces") # Corrected subdirectory
        
        if not os.path.isdir(actual_image_data_path):
            print(f"ERROR: Expected image directory not found at {actual_image_data_path} after download.")
            print("Please check the dataset structure from Kaggle: vasukipatel/face-recognition-dataset")
            # Attempt to list contents if path is wrong, for debugging.
            try:
                if os.path.exists(image_dataset_root_path):
                    print(f"Contents of downloaded root path '{image_dataset_root_path}': {os.listdir(image_dataset_root_path)}")
            except Exception as e_list:
                print(f"Could not list contents of {image_dataset_root_path}: {e_list}")
            actual_image_data_path = None # Indicate failure
        else:
            print(f"Image data is expected to be at: {actual_image_data_path}")

    except Exception as e: # Changed to generic Exception
        print(f"Caught exception of type: {type(e)}") # Added type print
        # Retain more specific user guidance if possible, though type check is harder now
        if isinstance(e, ModuleNotFoundError): # Example, may not be KaggleApiHTTPError directly
             print("Please ensure 'kagglehub' is installed ('pip install kagglehub') and you are authenticated ('kaggle login').")
        elif "401" in str(e) or "authentication failed" in str(e).lower():
            print("Authentication failed. Please ensure you are logged into Kaggle CLI ('kaggle login') and kaggle.json is correctly placed.")
        elif "404" in str(e) or "not found" in str(e).lower():
            print("Dataset not found. Please check the dataset handle: 'vasukipatel/face-recognition-dataset'")
        else:
            print("An unknown API or general error occurred during image dataset download.")
        actual_image_data_path = None # Indicate failure

    print("\nDownloading FaceNet Keras model (utkarshsaxenadn/facenet-keras)...")
    try:
        # Download to Kaggle's default cache directory first
        print("Attempting to download FaceNet model dataset to Kaggle cache...")
        cached_download_path = kagglehub.dataset_download("utkarshsaxenadn/facenet-keras")
        print(f"FaceNet model dataset downloaded to cache path: {cached_download_path}")

        # Define source and target paths for model files
        # FACENET_MODEL_PATH is already "models/facenet_keras.h5"
        files_to_copy = {
            "model.json": os.path.join("models", "facenet_model.json"),
            "weights.h5": os.path.join("models", "facenet_model_weights.h5"),
            "facenet_keras.h5": FACENET_MODEL_PATH,
            "model.h5": os.path.join("models", "facenet_alternate_model.h5") # For the other model.h5
        }

        os.makedirs("models", exist_ok=True) # Ensure models directory exists

        found_and_copied_any_main_component = False

        if os.path.isdir(cached_download_path):
            print(f"Files in cached FaceNet download path '{cached_download_path}': {os.listdir(cached_download_path)}")
            print(f"Searching for model files in cache directory: {cached_download_path}")
            for source_filename, target_path in files_to_copy.items():
                potential_source_path = os.path.join(cached_download_path, source_filename)
                if os.path.exists(potential_source_path):
                    print(f"Found '{source_filename}' in cache. Copying to '{target_path}'...")
                    try:
                        shutil.copy(potential_source_path, target_path)
                        if os.path.exists(target_path):
                            print(f"  Successfully copied '{source_filename}' to '{target_path}'.")
                            if source_filename == "model.json" or source_filename == "weights.h5":
                                found_and_copied_any_main_component = True
                            elif source_filename == "facenet_keras.h5" and not found_and_copied_any_main_component:
                                # If json/weights aren't found, facenet_keras.h5 is the primary fallback
                                found_and_copied_any_main_component = True
                        else:
                            print(f"  ERROR: Failed to copy '{source_filename}' to '{target_path}'.")
                    except Exception as e_copy:
                        print(f"  ERROR copying '{source_filename}': {e_copy}")
                else:
                    print(f"  '{source_filename}' not found in '{cached_download_path}'.")
        elif os.path.isfile(cached_download_path):
            # This case handles if dataset_download returns a single file path (less likely for multi-file datasets)
            print(f"Cache path '{cached_download_path}' is a file. Checking if it's one of the expected model files.")
            filename_in_cache = os.path.basename(cached_download_path)
            if filename_in_cache in files_to_copy:
                target_path = files_to_copy[filename_in_cache]
                print(f"Copying '{filename_in_cache}' from '{cached_download_path}' to '{target_path}'...")
                try:
                    shutil.copy(cached_download_path, target_path)
                    if os.path.exists(target_path):
                        print(f"  Successfully copied '{filename_in_cache}' to '{target_path}'.")
                        if filename_in_cache == "model.json" or filename_in_cache == "weights.h5":
                            found_and_copied_any_main_component = True
                        elif filename_in_cache == "facenet_keras.h5" and not found_and_copied_any_main_component:
                            found_and_copied_any_main_component = True
                    else:
                        print(f"  ERROR: Failed to copy '{filename_in_cache}' to '{target_path}'.")
                except Exception as e_copy:
                     print(f"  ERROR copying '{filename_in_cache}': {e_copy}")
            else:
                print(f"Downloaded file '{filename_in_cache}' is not one of the expected model files.")
        else:
            print(f"ERROR: Kaggle download path '{cached_download_path}' is not a valid file or directory.")

        if not found_and_copied_any_main_component:
            print(f"ERROR: Core FaceNet model components (model.json/weights.h5 or facenet_keras.h5) were not found or copied from '{cached_download_path}'.")

    except Exception as e: # Changed to generic Exception
        print(f"Error downloading FaceNet Keras model: {e}")
        print(f"Caught exception of type: {type(e)}") # Added type print
        # Retain more specific user guidance
        if isinstance(e, ModuleNotFoundError):
            print("Please ensure 'kagglehub' is installed ('pip install kagglehub') and you are authenticated ('kaggle login').")
        elif "401" in str(e) or "authentication failed" in str(e).lower():
            print("Authentication failed. Please ensure you are logged into Kaggle CLI ('kaggle login') and kaggle.json is correctly placed.")
        elif "404" in str(e) or "not found" in str(e).lower():
            print("FaceNet model not found. Please check the model handle: 'utkarshsaxenadn/facenet-keras'")
        else:
            print("An unknown API or general error occurred during FaceNet model download.")
        print("  You might need to manually download 'facenet_keras.h5' from Kaggle ('utkarshsaxenadn/facenet-keras')")
        print("  and place it in the 'models/' directory.")

    # Return the path to the image dataset. If it failed, this will be None.
    if actual_image_data_path:
        print(f"Returning image data path: {actual_image_data_path}")
    else:
        print("Image data path could not be determined due to download/extraction issues.")
        # Fallback to defined RAW_DATA_PATH if all else fails, though it's unlikely to exist if download failed.
        # This makes the function always return a string, but main.py should check for its validity.
        # print(f"Falling back to default RAW_DATA_PATH: {RAW_DATA_PATH} - this path might not be populated.") # Commented out RAW_DATA_PATH fallback
        return None # Return None if path could not be determined. Main.py handles this.

    return actual_image_data_path

def preprocess_and_save_faces(raw_data_dir, processed_data_dir, img_size=(160, 160)):
    """
    Detects faces, crops, resizes, and saves them.
    Ensures the cascade classifier XML is available.
    """
    if not os.path.exists(CASCADE_PATH):
        print(f"Haar cascade file '{CASCADE_PATH}' not found. Attempting to locate or download...")
        # Try common OpenCV path
        cv_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        if os.path.exists(cv_cascade_path):
            shutil.copy(cv_cascade_path, CASCADE_PATH)
            print(f"Copied cascade file from {cv_cascade_path}")
        else:
            # Attempt download
            try:
                import urllib.request
                url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
                print(f"Downloading {CASCADE_PATH} from {url}...")
                urllib.request.urlretrieve(url, CASCADE_PATH)
                print(f"{CASCADE_PATH} downloaded successfully.")
            except Exception as e:
                print(f"Error downloading {CASCADE_PATH}: {e}. Please ensure it's available.")
                return False # Cannot proceed without cascade
    
    if not os.path.exists(raw_data_dir):
        print(f"Error: Raw data directory not found at {raw_data_dir}")
        return False

    if os.path.exists(processed_data_dir):
        print(f"Processed data directory {processed_data_dir} already exists. Clearing it.")
        shutil.rmtree(processed_data_dir)
    os.makedirs(processed_data_dir, exist_ok=True)

    # Get all jpg files from the raw data directory
    image_files = []
    for root, _, files in os.walk(raw_data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))

    # Process each image and organize by person
    person_processed_count = {}
    total_processed = 0

    for img_path in image_files:
        # Extract person name from filename (format: "Person Name_Number.jpg")
        person_name = os.path.basename(img_path).split('_')[0]
        
        # Create person's directory if it doesn't exist
        person_processed_path = os.path.join(processed_data_dir, person_name)
        os.makedirs(person_processed_path, exist_ok=True)

        # Read and process image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        faces = detect_faces(img_path, CASCADE_PATH)
        if faces is None:  # Critical error with cascade
            print(f"Critical error: Face detection failed for {img_path} due to cascade issues. Skipping further processing.")
            return False
        
        if len(faces) > 0:
            largest_face = get_largest_face(faces)
            if largest_face is not None:
                x, y, w, h = largest_face
                face_crop = image[y:y+h, x:x+w]
                
                if face_crop.size == 0:
                    print(f"Warning: Cropped face from {img_path} is empty. Original face coords: {largest_face}")
                    continue

                face_resized = cv2.resize(face_crop, img_size)
                
                # Save processed face
                save_path = os.path.join(person_processed_path, os.path.basename(img_path))
                cv2.imwrite(save_path, face_resized)
                
                # Update counters
                person_processed_count[person_name] = person_processed_count.get(person_name, 0) + 1
                total_processed += 1

    # Print processing summary
    print("\nProcessing Summary:")
    for person, count in person_processed_count.items():
        print(f"Processed {count} images for {person}")
    print(f"Total processed images: {total_processed}")

    if total_processed == 0:
        print("Error: No images were successfully processed.")
        return False

    print("Face preprocessing complete.")
    return True


def split_data(processed_data_dir, train_dir, test_dir, test_size=0.2):
    """
    Splits processed faces into train and test sets, ensuring each person is in both.
    """
    if not os.path.exists(processed_data_dir):
        print(f"Error: Processed data directory {processed_data_dir} not found.")
        return

    if os.path.exists(train_dir): shutil.rmtree(train_dir)
    if os.path.exists(test_dir): shutil.rmtree(test_dir)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_dir_basename = os.path.basename(train_dir)
    test_dir_basename = os.path.basename(test_dir)

    for person_name in os.listdir(processed_data_dir):
        # Explicitly skip 'train' and 'test' directories if they are listed
        if person_name == train_dir_basename or person_name == test_dir_basename:
            continue

        person_path = os.path.join(processed_data_dir, person_name)
        if not os.path.isdir(person_path): # Skip if it's not a directory
            continue

        person_train_path = os.path.join(train_dir, person_name)
        person_test_path = os.path.join(test_dir, person_name)
        os.makedirs(person_train_path, exist_ok=True)
        os.makedirs(person_test_path, exist_ok=True)

        images = [img for img in os.listdir(person_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(images) < 2 : # Need at least 2 images to split
            print(f"Warning: Person {person_name} has fewer than 2 images. Copying all to train set.")
            for img_name in images:
                shutil.copy(os.path.join(person_path, img_name), os.path.join(person_train_path, img_name))
            continue

        # Stratified split per person if possible
        try:
            # Create dummy labels for stratification, not used otherwise for this split
            labels = [0] * len(images) 
            train_imgs, test_imgs = train_test_split(images, test_size=test_size, random_state=42, stratify=labels if len(np.unique(labels)) > 1 else None)
        except ValueError: # Happens if not enough samples for stratification with a single class
             train_imgs, test_imgs = train_test_split(images, test_size=test_size, random_state=42)


        if not test_imgs and len(images) > 0: # Ensure test set gets at least one image if possible
            if len(train_imgs) > 1: # If we have enough in train, move one to test
                test_imgs = [train_imgs.pop()]
            else: # Otherwise, copy the single image to both if that's all we have
                 print(f"Warning: Could not create a distinct test set for {person_name} with test_size {test_size}. Copying one image to test.")
                 if images:
                    test_imgs = [images[0]] # Use the first available image for test

        for img_name in train_imgs:
            shutil.copy(os.path.join(person_path, img_name), os.path.join(person_train_path, img_name))
        for img_name in test_imgs:
            shutil.copy(os.path.join(person_path, img_name), os.path.join(person_test_path, img_name))
        
        print(f"Split data for {person_name}: {len(train_imgs)} train, {len(test_imgs)} test")

    print("Data splitting complete.")


def load_data_from_processed_dir(directory):
    """Loads images and labels from a processed directory (train or test)."""
    X, y, person_names = [], [], []
    label_map = {} # Map person names to integer labels
    current_label = 0

    for person_name in os.listdir(directory):
        person_dir = os.path.join(directory, person_name)
        if os.path.isdir(person_dir):
            if person_name not in label_map:
                label_map[person_name] = current_label
                person_names.append(person_name)
                current_label += 1
            
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # Eigen/Fisher work on grayscale
                if image is not None:
                    X.append(np.asarray(image, dtype=np.uint8).flatten()) # Flatten for Eigen/Fisher
                    y.append(label_map[person_name])
                else:
                    print(f"Warning: Could not load image {img_path} during data loading.")
    
    if not X: # If no data was loaded
        print(f"Warning: No data loaded from {directory}. Check directory structure and image files.")
        return np.array([]), np.array([]), [], {}

    return np.asarray(X), np.asarray(y), person_names, label_map

def load_data_for_facenet(directory, img_size=(160,160)):
    """Loads images for FaceNet (RGB, not flattened) and labels from a processed directory."""
    X, y, person_names = [], [], []
    label_map = {}
    current_label = 0

    for person_name in os.listdir(directory):
        person_dir = os.path.join(directory, person_name)
        if os.path.isdir(person_dir):
            if person_name not in label_map:
                label_map[person_name] = current_label
                person_names.append(person_name)
                current_label += 1
            
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                image = cv2.imread(img_path) # Load in color for FaceNet
                if image is not None:
                    # Ensure image is correct size, though preprocessing should handle this
                    if image.shape[:2] != img_size:
                        image = cv2.resize(image, img_size) 
                    X.append(image) # Keep as 3D array (height, width, channels)
                    y.append(label_map[person_name])
                else:
                    print(f"Warning: Could not load image {img_path} for FaceNet.")
    
    if not X:
        print(f"Warning: No FaceNet data loaded from {directory}.")
        return np.array([]), np.array([]), [], {}
        
    return np.asarray(X), np.asarray(y), person_names, label_map


if __name__ == '__main__':
    # 1. Download data
    raw_dataset_path = download_dataset()
    if not raw_dataset_path or not os.path.exists(raw_dataset_path):
        print(f"Failed to get raw dataset path or path does not exist: {raw_dataset_path}")
        exit()

    # 2. Preprocess faces
    # Ensure the PROCESSED_DATA_PATH points to a directory for "person_name/image.jpg" structure
    # The preprocess_and_save_faces already creates person-specific subfolders in PROCESSED_DATA_PATH/
    # So, the output of this step is PROCESSED_DATA_PATH itself.
    print(f"Starting preprocessing. Raw data from: {raw_dataset_path}, Processed data to: {PROCESSED_DATA_PATH}")
    if not preprocess_and_save_faces(raw_dataset_path, PROCESSED_DATA_PATH, img_size=IMG_SIZE):
        print("Failed to preprocess data. Exiting.")
        exit()

    # 3. Split data
    # The split_data function takes the output of preprocess_and_save_faces (PROCESSED_DATA_PATH)
    # and creates TRAIN_DIR and TEST_DIR within it.
    print(f"Splitting data from {PROCESSED_DATA_PATH} into {TRAIN_DIR} and {TEST_DIR}")
    split_data(PROCESSED_DATA_PATH, TRAIN_DIR, TEST_DIR)

    print("Data preprocessing and splitting finished.")
    print(f"Train data is in: {TRAIN_DIR}")
    print(f"Test data is in: {TEST_DIR}")

    # Example of loading data (for Eigen/Fisher)
    print("\nLoading training data for Eigen/Fisher...")
    X_train, y_train, train_person_names, train_label_map = load_data_from_processed_dir(TRAIN_DIR)
    if X_train.size > 0:
        print(f"Loaded {X_train.shape[0]} training samples. {len(train_person_names)} unique persons.")
        print(f"Person to label map: {train_label_map}")
    else:
        print("No training data loaded. Check TRAIN_DIR and logs.")

    print("\nLoading test data for Eigen/Fisher...")
    X_test, y_test, test_person_names, test_label_map = load_data_from_processed_dir(TEST_DIR)
    if X_test.size > 0:
        print(f"Loaded {X_test.shape[0]} test samples. {len(test_person_names)} unique persons.")
    else:
        print("No test data loaded. Check TEST_DIR and logs.")

    # Example of loading data for FaceNet
    print("\nLoading training data for FaceNet...")
    X_train_facenet, y_train_facenet, _, _ = load_data_for_facenet(TRAIN_DIR, IMG_SIZE)
    if X_train_facenet.size > 0:
        print(f"Loaded {X_train_facenet.shape[0]} training samples for FaceNet.")
    else:
        print("No FaceNet training data loaded.") 