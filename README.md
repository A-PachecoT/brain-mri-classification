# Brain MRI Classification

A deep learning project for classifying brain MRI scans to detect tumors using TensorFlow.

## Credits

This project is based on the Jupyter notebook by [Anirudh Bansal](https://www.kaggle.com/anibansal) from his Kaggle kernel [Brain MRI Classification](https://www.kaggle.com/code/anibansal/brain-mri-classification). The original work has been restructured into a proper Python package with improved organization and modularity.

## Project Structure

```
.
├── artifacts/          # Generated artifacts
│   ├── models/        # Saved models
│   └── plots/         # Generated plots and visualizations
├── data/              # Data directory
│   ├── processed/     # Processed dataset
│   └── raw/          # Raw dataset
│       ├── yes/      # MRI scans with tumors
│       └── no/       # MRI scans without tumors
├── notebooks/         # Jupyter notebooks
├── src/              # Source code
│   ├── data/         # Data loading and preprocessing
│   ├── models/       # Model architecture
│   ├── utils/        # Utility functions
│   └── main.py       # Main script
└── requirements.txt   # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
conda create -n mri-classification python=3.8
conda activate mri-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the training pipeline:
```bash
python src/main.py
```

## Model Architecture

The model uses a CNN architecture with:
- 4 convolutional layers
- Max pooling layers
- Dropout for regularization
- Dense layers for classification

## Results

The model achieves:
- 75% precision for no-tumor cases
- 71% precision for tumor cases
- 73% overall accuracy

Results are saved in:
- Model weights: `artifacts/models/best_model.h5`
- Training plots: `artifacts/plots/training_history.png`
- Confusion matrix: `artifacts/plots/confusion_matrix.png` 