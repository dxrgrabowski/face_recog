import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

RESULTS_DIR = "results"
METRICS_FILE = os.path.join(RESULTS_DIR, "all_model_metrics.csv")

def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)

def calculate_metrics(y_true, y_pred, average_method='weighted'):
    """
    Calculates accuracy, precision, recall, and F1-score.
    Returns a dictionary of metrics.
    """
    if y_true is None or y_pred is None or len(y_true) == 0 or len(y_pred) == 0:
        print("Warning: True labels or predicted labels are empty. Cannot calculate metrics.")
        return {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0
        }
    if len(y_true) != len(y_pred):
        print(f"Warning: Mismatch in length of true labels ({len(y_true)}) and predicted labels ({len(y_pred)}). Cannot calculate metrics reliably.")
        # Pad or truncate? For now, returning zeros or handling as error.
        return {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0
        }
    
    accuracy = accuracy_score(y_true, y_pred)
    # zero_division=0 means if a class has no predictions, its precision/recall/F1 will be 0 for that class.
    # This prevents warnings/errors when a class is not present in predictions.
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average_method, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """
    Plots and saves a confusion matrix.
    class_names: list of actual string names for labels for better plot readability.
    """
    ensure_results_dir()
    if y_true is None or y_pred is None or len(y_true) == 0 or len(y_pred) == 0:
        print(f"Warning: Cannot plot confusion matrix for {model_name} due to empty labels.")
        return

    cm = confusion_matrix(y_true, y_pred, labels=np.unique(np.concatenate((y_true, y_pred)))) # Ensure all labels are considered
    
    # If class_names are provided and match the number of unique labels, use them.
    # Otherwise, use unique numerical labels as tick labels.
    unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    if class_names and len(class_names) == len(unique_labels):
        tick_labels = [class_names[i] for i in unique_labels] # Map unique numeric labels to names
        # This assumes class_names is a list where index corresponds to label number.
        # More robust: create a mapping from label number to name if `label_map` is available.
    else:
        tick_labels = unique_labels
        if class_names:
            print(f"Warning: Provided class_names length ({len(class_names)}) does not match unique labels ({len(unique_labels)}) for {model_name}. Using numeric labels for CM.")

    plt.figure(figsize=(12, 12))
    try:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=tick_labels, yticklabels=tick_labels)
    except ValueError as e:
        print(f"Error during heatmap generation for {model_name} (possibly due to tick labels): {e}")
        # Fallback to default tick labels if there was an issue with custom ones
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    save_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix for {model_name} saved to {save_path}")

def save_metrics_to_csv(metrics_dict, model_name, label_map=None):
    """
    Saves a dictionary of metrics for a model to a CSV file.
    Appends to the file if it exists, or creates a new one.
    label_map: dictionary mapping integer labels to person names (optional, for context).
    """
    ensure_results_dir()
    metrics_df = pd.DataFrame([metrics_dict])
    metrics_df['model'] = model_name
    
    # Add label map information if available, for context on which labels were used
    if label_map:
        # Convert label_map to a string to store in CSV
        metrics_df['label_map_preview'] = str({k: label_map[k] for k in list(label_map)[:5]}) # Preview first 5

    if os.path.exists(METRICS_FILE):
        existing_df = pd.read_csv(METRICS_FILE)
        # Remove previous entries for the same model to avoid duplicates if re-running
        existing_df = existing_df[existing_df['model'] != model_name]
        combined_df = pd.concat([existing_df, metrics_df], ignore_index=True)
    else:
        combined_df = metrics_df
    
    # Reorder columns to have 'model' first
    cols = ['model'] + [col for col in combined_df.columns if col != 'model']
    combined_df = combined_df[cols]
    
    combined_df.to_csv(METRICS_FILE, index=False)
    print(f"Metrics for {model_name} saved to {METRICS_FILE}")

def plot_overall_metrics_comparison():
    """
    Reads the all_model_metrics.csv and plots a comparison bar chart for key metrics.
    """
    ensure_results_dir()
    if not os.path.exists(METRICS_FILE):
        print(f"Metrics file {METRICS_FILE} not found. Cannot plot comparison.")
        return

    df = pd.read_csv(METRICS_FILE)
    if df.empty:
        print("Metrics file is empty. No data to plot for comparison.")
        return

    # Metrics to plot - can be customized
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    df_plot = df[['model'] + metrics_to_plot]
    
    df_melted = df_plot.melt(id_vars='model', var_name='metric', value_name='score')

    plt.figure(figsize=(12, 8))
    sns.barplot(x='metric', y='score', hue='model', data=df_melted, palette='viridis')
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xlabel('Metric')
    plt.xticks(rotation=45)
    plt.legend(title='Model')
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "model_metrics_comparison.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Model metrics comparison plot saved to {save_path}")


# Universal evaluation function
def evaluate_model(y_true, y_pred, model_name, class_names=None, label_map=None):
    """
    A common function to evaluate any model, calculate, plot, and save metrics.
    class_names: A list of actual string names for labels (e.g. ['PersonA', 'PersonB', ...])
                 Should correspond to the unique sorted labels in y_true/y_pred.
                 If None, numeric labels are used for CM plot.
    label_map: The original mapping from person name to numeric label (for context in CSV).
    """
    print(f"\n--- Evaluating {model_name} ---")
    if y_true is None or y_pred is None or not hasattr(y_true, '__len__') or not hasattr(y_pred, '__len__') or len(y_true) == 0 or len(y_pred) == 0:
        print(f"Skipping evaluation for {model_name} due to empty or invalid true/predicted labels.")
        # Optionally, save a record indicating no results
        metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0, 'error': 'No prediction data'}
        save_metrics_to_csv(metrics, model_name, label_map)
        return metrics

    if len(y_true) != len(y_pred):
        print(f"Error: Mismatch in length of y_true ({len(y_true)}) and y_pred ({len(y_pred)}) for {model_name}. Evaluation aborted.")
        metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0, 'error': 'Label length mismatch'}
        save_metrics_to_csv(metrics, model_name, label_map)
        return metrics

    metrics = calculate_metrics(y_true, y_pred)
    print(f"Metrics for {model_name}: {metrics}")
    
    # Ensure class_names are correctly derived if not provided, or if they don't match unique labels
    unique_numeric_labels = sorted(list(np.unique(np.concatenate((y_true, y_pred)))))

    effective_class_names = None
    if class_names:
        if len(class_names) == len(unique_numeric_labels):
            # This assumes class_names is already sorted or maps directly to sorted unique_numeric_labels
            # A safer way is to use label_map to get names for unique_numeric_labels
            if label_map:
                # Create a reverse map from numeric label to name
                reverse_label_map = {v: k for k, v in label_map.items()}
                effective_class_names = [reverse_label_map.get(lbl, str(lbl)) for lbl in unique_numeric_labels]
            else:
                effective_class_names = [str(lbl) for lbl in unique_numeric_labels] # Fallback if no label_map
        else:
            print(f"Warning: class_names provided for {model_name} but length mismatch. Using numeric labels for CM.")
            effective_class_names = [str(lbl) for lbl in unique_numeric_labels]
    elif label_map: # If class_names not given, but label_map is, derive names from it
        reverse_label_map = {v: k for k, v in label_map.items()}
        effective_class_names = [reverse_label_map.get(lbl, str(lbl)) for lbl in unique_numeric_labels]
    else:
        effective_class_names = [str(lbl) for lbl in unique_numeric_labels]

    plot_confusion_matrix(y_true, y_pred, class_names=effective_class_names, model_name=model_name)
    save_metrics_to_csv(metrics, model_name, label_map)
    
    return metrics

if __name__ == '__main__':
    ensure_results_dir()
    print("Testing Evaluation module...")

    # Dummy data for testing
    y_true_dummy = np.array([0, 1, 0, 1, 2, 0, 1, 2, 2, 0])
    y_pred_eigen_dummy = np.array([0, 1, 0, 0, 2, 0, 1, 1, 2, 0])
    y_pred_fisher_dummy = np.array([0, 1, 1, 1, 2, 0, 0, 2, 2, 1])
    y_pred_facenet_dummy = np.array([0, 1, 0, 1, 2, 0, 1, 2, 2, 0]) # Perfect dummy score for one

    dummy_label_map = { 'PersonA': 0, 'PersonB': 1, 'PersonC': 2 }
    # Deriving class_names list from label_map for the plot_confusion_matrix
    # Ensure the order of class_names matches the sorted unique labels in y_true/y_pred
    # sorted_unique_labels = sorted(list(np.unique(y_true_dummy)))
    # dummy_class_names = [name for label, name in sorted(dummy_label_map.items(), key=lambda item: item[1]) if label in sorted_unique_labels]
    # The evaluate_model function now handles derivation of class_names more robustly using label_map.

    print("\n--- Evaluating Dummy EigenFaces ---")
    metrics_eigen = evaluate_model(y_true_dummy, y_pred_eigen_dummy, "EigenFaces_Dummy", label_map=dummy_label_map)
    
    print("\n--- Evaluating Dummy FisherFaces ---")
    metrics_fisher = evaluate_model(y_true_dummy, y_pred_fisher_dummy, "FisherFaces_Dummy", label_map=dummy_label_map)

    print("\n--- Evaluating Dummy FaceNet ---")
    metrics_facenet = evaluate_model(y_true_dummy, y_pred_facenet_dummy, "FaceNet_Dummy", label_map=dummy_label_map)

    print("\nAll dummy metrics calculated and saved (if any). Check the 'results' directory.")

    # Plot overall comparison (will use the CSV file generated)
    plot_overall_metrics_comparison()

    print("\nEvaluation module test finished. Check for 'all_model_metrics.csv' and plots in 'results/'") 