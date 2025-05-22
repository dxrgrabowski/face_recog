# Face Recognition System

This project implements multiple face recognition algorithms (EigenFaces, FisherFaces, and FaceNet) and compares their performance on a dataset of celebrity faces.

## Features

- Multiple face recognition algorithms:
  - EigenFaces (OpenCV implementation)
  - FisherFaces (OpenCV implementation)
  - FaceNet (Deep learning-based)
- Face detection using Haar Cascade
- Data preprocessing and augmentation
- Model evaluation and comparison
- Support for skipping training and using existing models

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- opencv-contrib-python (for EigenFaces and FisherFaces)
- tensorflow (for FaceNet)
- scikit-learn (for SVM classifier in FaceNet)
- kagglehub (for dataset and model download)

## Usage

### Full Pipeline (Download, Preprocess, Train, Evaluate)

```bash
python3 -m src.main
```

### Skip Training (Use Existing Models)

```bash
python3 -m src.main --skip-training
```

## Current Results

The system has been tested on a dataset of 31 different people with the following results:

- EigenFaces: 35.87% accuracy
- FisherFaces: 20.29% accuracy
- FaceNet: Currently not working (model loading issues)

Note: These results are lower than typical research results (EigenFaces: 76%, FisherFaces: 88%) due to:
- Larger number of classes (31 people)
- More challenging dataset conditions
- Different preprocessing steps

## Project Structure

```
.
├── data/
│   ├── processed/    # Preprocessed face images
│   ├── train/        # Training set
│   └── test/         # Test set
├── models/           # Saved models
├── results/          # Evaluation results and plots
└── src/
    ├── data_preprocessing.py
    ├── face_detection.py
    ├── recognition_eigen.py
    ├── recognition_fisher.py
    ├── recognition_facenet.py
    ├── evaluation.py
    └── main.py
```

## Known Issues

1. FaceNet model loading issues:
   - The current FaceNet model files from Kaggle appear to be corrupted
   - Need to download a valid model from a trusted source

2. Performance:
   - Current accuracy is lower than research results
   - Potential improvements:
     - Better preprocessing
     - Data augmentation
     - Parameter tuning
     - Using a different FaceNet model

## Contributing

Feel free to submit issues and enhancement requests!

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd face_recog
```

2. Create a virtual environment (recommended):
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
face_recog/
├── data/           # Directory for storing face datasets
├── models/         # Directory for storing trained models
├── results/        # Directory for storing recognition results
├── src/            # Source code
│   ├── main.py                    # Main entry point
│   ├── face_detection.py          # Face detection implementation
│   ├── data_preprocessing.py      # Data preprocessing utilities
│   ├── recognition_facenet.py     # FaceNet implementation
│   ├── recognition_fisher.py      # Fisher Faces implementation
│   ├── recognition_eigen.py       # Eigen Faces implementation
│   └── evaluation.py              # Evaluation metrics and utilities
└── requirements.txt # Project dependencies
```

## Usage

1. Place your face dataset in the `data/` directory.

2. Run the main program:
```bash
python src/main.py
```

The program supports multiple face recognition algorithms:
- FaceNet (default)
- Fisher Faces
- Eigen Faces

## Dependencies

- opencv-python: Computer vision library
- numpy: Numerical computing
- pandas: Data manipulation
- scikit-learn: Machine learning utilities
- tensorflow: Deep learning framework
- matplotlib: Data visualization
- seaborn: Statistical data visualization
- kagglehub: Dataset management

## Notes

- The system uses TensorFlow by default, but you can modify the requirements.txt to use PyTorch instead if preferred.
- Make sure you have sufficient disk space for storing the face datasets and trained models.
- For optimal performance, a GPU is recommended but not required.

## License

[Add your license information here] 