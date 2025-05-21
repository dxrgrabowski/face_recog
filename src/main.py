import os
import numpy as np

# Project-specific imports from src
from src.data_preprocessing import (
    download_dataset, preprocess_and_save_faces, split_data,
    load_data_from_processed_dir, load_data_for_facenet,
    PROCESSED_DATA_PATH, TRAIN_DIR, TEST_DIR, IMG_SIZE, CASCADE_PATH
)
from src.recognition_eigen import (
    train_eigen_recognizer, load_eigen_model, predict_eigen,
    MODEL_PATH as EIGEN_MODEL_PATH
)
from src.recognition_fisher import (
    train_fisher_recognizer, load_fisher_model, predict_fisher,
    MODEL_PATH as FISHER_MODEL_PATH
)
from src.recognition_facenet import (
    load_facenet_keras_model, generate_embeddings, train_facenet_classifier,
    load_facenet_classifier, predict_facenet, FACENET_MODEL_PATH, CLASSIFIER_MODEL_PATH as FACENET_CLASSIFIER_PATH
)
from src.evaluation import evaluate_model, plot_overall_metrics_comparison, ensure_results_dir


def check_prerequisites_post_download():
    """Checks for essential files AFTER initial download attempts."""
    all_prerequisites_met = True # Assume true initially
    critical_prerequisites_met = True # Specific for FaceNet model
    print("--- Checking Prerequisites (Post Initial Download Attempts) ---")

    # 1. Haar Cascade
    if not os.path.exists(CASCADE_PATH):
        print(f"WARNING: Haar cascade '{CASCADE_PATH}' is NOT available yet.")
        print("  This is not critical at this stage, as preprocess_and_save_faces() will attempt to acquire it.")
        # Not setting all_prerequisites_met = False here, as preprocess_and_save_faces handles it.
    else:
        print(f"Haar cascade '{CASCADE_PATH}' is available.")

    # 2. FaceNet Keras Model (This is critical)
    if not os.path.exists(FACENET_MODEL_PATH):
        print(f"CRITICAL: FaceNet Keras model '{FACENET_MODEL_PATH}' not found after download attempt.")
        print("  Please check for errors in the 'Data Download' step logs.")
        print("  If download failed, you might need to manually download it from Kaggle ('srijanc1997/facenet-kerash5')")
        print("  and place 'facenet_keras.h5' into the 'models/' directory.")
        all_prerequisites_met = False
        critical_prerequisites_met = False
    else:
        print(f"FaceNet Keras model '{FACENET_MODEL_PATH}' is available.")
    
    if not critical_prerequisites_met:
        print("One or more CRITICAL prerequisites (like the FaceNet model) are missing. The pipeline cannot continue.")
    elif not all_prerequisites_met: # This case might not be hit if only critical_prerequisites_met can make it false
        print("Some non-critical prerequisites were noted. Proceeding, but downstream steps might fail if not resolved.")
    else:
        print("All checked prerequisites appear to be met, or will be handled by subsequent steps.")
    
    return critical_prerequisites_met # Return status based on CRITICAL prerequisites only for early exit

def main():
    print("===== Face Recognition Pipeline Started =====")

    ensure_results_dir()

    # --- Step 1: Data Download (including FaceNet model) ---
    print("\n--- Step 1.1: Data Download (Image Dataset & FaceNet Model) ---")
    actual_raw_data_path = download_dataset() # This downloads images and FaceNet model
    
    if not actual_raw_data_path or not os.path.exists(actual_raw_data_path):
        print(f"Failed to download or locate image dataset. Expected at related to: {actual_raw_data_path}. Exiting.")
        return
    print("Image dataset download/location attempt finished.")
    # FaceNet model download attempt also occurred within download_dataset()

    # --- Prerequisite Check (Post Download Attempts) ---
    # Now check if critical files (like FaceNet model) are actually present after download attempt.
    if not check_prerequisites_post_download():
        print("Critical prerequisites check failed after download attempts. Exiting pipeline.")
        return

    # --- Step 1.2: Data Preprocessing (Haar Cascade handled here) ---
    print("\n--- Step 1.2: Image Data Preprocessing (Cropping, Resizing, Splitting) ---")
    # preprocess_and_save_faces also handles Haar Cascade XML file availability.
    if not preprocess_and_save_faces(raw_data_dir=actual_raw_data_path, processed_data_dir=PROCESSED_DATA_PATH, img_size=IMG_SIZE):
        print("Data preprocessing (cropping/resizing faces) failed. Exiting.")
        return
    split_data(PROCESSED_DATA_PATH, TRAIN_DIR, TEST_DIR)
    print("Data preprocessing and splitting complete.")

    print("\n--- Step 2: Loading Processed Data ---")
    X_train_cv, y_train_cv, train_person_names, train_label_map = load_data_from_processed_dir(TRAIN_DIR)
    X_test_cv, y_test_cv, _, test_label_map = load_data_from_processed_dir(TEST_DIR)
    
    X_train_fn, y_train_fn, _, _ = load_data_for_facenet(TRAIN_DIR, img_size=IMG_SIZE)
    X_test_fn, y_test_fn, _, _ = load_data_for_facenet(TEST_DIR, img_size=IMG_SIZE)

    if X_train_cv.size == 0 or X_test_cv.size == 0:
        print("Failed to load data for OpenCV models. Check preprocessing logs and data directories. Exiting.")
        return
    if X_train_fn.size == 0 or X_test_fn.size == 0:
        print("Failed to load data for FaceNet model. Check preprocessing logs and data directories. Exiting.")
        return
    
    master_label_map = train_label_map
    sorted_labels = sorted(master_label_map.items(), key=lambda item: item[1])
    class_names_for_plot = [name for name, label in sorted_labels]

    print("\n--- Step 3.1: EigenFaces ---")
    if len(np.unique(y_train_cv)) < 2:
        print("Skipping EigenFaces: requires at least 2 classes for training.")
    else:
        eigen_model = train_eigen_recognizer(X_train_cv, y_train_cv)
        if eigen_model:
            loaded_eigen_model = load_eigen_model(EIGEN_MODEL_PATH)
            if loaded_eigen_model:
                y_pred_eigen = predict_eigen(loaded_eigen_model, X_test_cv)
                if y_pred_eigen:
                    evaluate_model(y_test_cv, np.array(y_pred_eigen), "EigenFaces", class_names=class_names_for_plot, label_map=master_label_map)
            else:
                print("Failed to load saved EigenFaces model for evaluation.")
        else:
            print("Failed to train EigenFaces model.")

    print("\n--- Step 3.2: FisherFaces ---")
    if len(np.unique(y_train_cv)) < 2:
        print("Skipping FisherFaces: requires at least 2 classes for training.")
    else:
        fisher_model = train_fisher_recognizer(X_train_cv, y_train_cv)
        if fisher_model:
            loaded_fisher_model = load_fisher_model(FISHER_MODEL_PATH)
            if loaded_fisher_model:
                y_pred_fisher = predict_fisher(loaded_fisher_model, X_test_cv)
                if y_pred_fisher:
                    evaluate_model(y_test_cv, np.array(y_pred_fisher), "FisherFaces", class_names=class_names_for_plot, label_map=master_label_map)
            else:
                print("Failed to load saved FisherFaces model for evaluation.")
        else:
            print("Failed to train FisherFaces model.")

    print("\n--- Step 3.3: FaceNet ---")
    print("\n--- Listing contents of models/ directory before FaceNet load ---")
    models_dir = "models"
    if os.path.exists(models_dir) and os.path.isdir(models_dir):
        print(f"Contents of '{models_dir}': {sorted(os.listdir(models_dir))}")
        for item_name in sorted(os.listdir(models_dir)):
            item_path = os.path.join(models_dir, item_name)
            try:
                print(f"  - {item_name} (Size: {os.path.getsize(item_path)} bytes)")
            except OSError:
                print(f"  - {item_name} (Error getting size)")
    else:
        print(f"Directory '{models_dir}' does not exist or is not a directory.")
    print("----------------------------------------------------------------")
    
    facenet_keras_model = load_facenet_keras_model()
    if not facenet_keras_model:
        print("FaceNet Keras model could not be loaded. Skipping FaceNet training and evaluation.")
    elif X_train_fn.size == 0 or y_train_fn.size == 0:
        print("Skipping FaceNet: No training data loaded.")
    elif len(np.unique(y_train_fn)) < 1:
        print("Skipping FaceNet: Requires at least 1 class for SVM training (though >1 is meaningful).")
    else:
        print("Generating FaceNet training embeddings...")
        train_embeddings = generate_embeddings(facenet_keras_model, X_train_fn)
        
        if train_embeddings.size > 0:
            facenet_classifier_tuple = train_facenet_classifier(train_embeddings, y_train_fn, FACENET_CLASSIFIER_PATH)
            
            if facenet_classifier_tuple and facenet_classifier_tuple[0]:
                loaded_classifier_tuple = load_facenet_classifier(FACENET_CLASSIFIER_PATH)
                if loaded_classifier_tuple and loaded_classifier_tuple[0]:
                    y_pred_facenet_encoded = predict_facenet(facenet_keras_model, loaded_classifier_tuple, X_test_fn)
                    
                    if y_pred_facenet_encoded:
                        evaluate_model(y_test_fn, np.array(y_pred_facenet_encoded), "FaceNet_SVM", class_names=class_names_for_plot, label_map=master_label_map)
                else:
                    print("Failed to load saved FaceNet classifier for evaluation.")
            else:
                print("Failed to train FaceNet classifier.")
        else:
            print("Failed to generate FaceNet training embeddings.")

    print("\n--- Step 4: Generating Overall Metrics Comparison Plot ---")
    plot_overall_metrics_comparison()

    print("\n===== Face Recognition Pipeline Finished =====")

if __name__ == '__main__':
    main() 