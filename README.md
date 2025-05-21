# Face Recognition System

This project implements a face recognition system using various algorithms including FaceNet, Fisher Faces, and Eigen Faces. The system includes face detection, preprocessing, and recognition capabilities.

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