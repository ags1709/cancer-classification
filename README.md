# Breast Cancer Classification
A machine learning project for binary classification of breast cancer tumors as benign or malignant using the Wisconsin Breast Cancer Dataset (https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).

## Project Overview

This project implements and compares multiple machine learning algorithms for breast cancer classification, achieving good performance through careful model selection, hyperparameter optimization, and evaluation using Leave-One-Out Cross-Validation (LOOCV).

### Key Features
- Implementation of multiple classification algorithms (Logistic Regression, SVM, Random Forest, KNN)
- Custom K-Nearest Neighbors implementation with probability predictions
- Comprehensive evaluation metrics including ROC-AUC, precision, recall, and F1-score
- Hyperparameter optimization using GridSearchCV with LOOCV
- Threshold analysis to facilitate best recall-precision trade-off
- Visualization of results including ROC curves and feature importance

## Results Summary

| Model | ROC-AUC Score | Accuracy | F1-score(avg over classes) | Best Parameters |
|-------|---------------|-----------------|--------------|-----------------|
| Logistic Regression | ~0.995 | 0.984 | 0.984 | C=0.54 |
| Support Vector Machine | ~0.996 | 0.984 | 0.984 | C=3, kernel='rbf' |
| Random Forest | ~0.991 | 0.961 | 0.961 |n_estimators=100, max_depth=None |
| K-Nearest Neighbors | ~0.985 | 0.972 | 0.972 | K=4 |


## Project Structure

```
cancerClassification/
│
├── data/
│   └── Cancer_data.csv          # Wisconsin Breast Cancer Dataset
│
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── baselineModels.py   # Custom KNN implementation
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── modelEvaluation.py  # Evaluation functions
│   │
│   ├── optimization/
│   │   ├── __init__.py
│   │   └── hyperparameterTuning.py  # GridSearchCV optimization
│   │
│   └── analysis/
│       ├── __init__.py
│       └── thresholdAnalysis.py  # Decision threshold analysis
│
├── main.py                      # Main script to run all models
├── requirements.txt             
└── README.md                  
```

## Running the project

First, navigate to the project folder. Make sure all requirements are installed. Check the requirements.txt file, or simply run the command
```bash
pip install -r requirements.txt
```

Once the requirements have been installed, you can run the complete analysis with all models using:
```bash
python main.py
```

Or, to run specific analyses:
```bash
# Run only model evaluation
python main.py --evaluate

# Run hyperparameter optimization (Warning: This takes significant time with LOOCV)
python main.py --optimize

# Run threshold analysis for the logistic regression model
python main.py --threshold-analysis

# Run feature importance analysis for the logistic regression model
python main.py --feature-importance
```



<!-- ## Model Details

### 1. Logistic Regression
- Implemented with L2 regularization
- Optimized regularization parameter C using LOOCV
- Features standardized

### 2. Support Vector Machine (SVM)
- RBF kernel with optimized C and gamma parameters
- Probability predictions enabled for ROC-AUC calculation
- Excellent performance on non-linearly separable data

### 3. Random Forest
- Ensemble method with 100 decision trees
- No maximum depth restriction for capturing complex patterns
- Feature importance analysis available

### 4. K-Nearest Neighbors (Custom Implementation)
- KNN with probability predictions
- Euclidean distance metric
- K=4 neighbors for optimal performance -->

<!-- ## Evaluation Methodology

- **Leave-One-Out Cross-Validation (LOOCV)**: Ensures robust evaluation on small dataset
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualization**: ROC curves, confusion matrices, feature importance plots -->

## Dataset Information

The Wisconsin Breast Cancer Dataset contains:
- 569 samples (357 benign, 212 malignant)
- 30 numeric features computed from digitized images
- Binary classification target (B=Benign, M=Malignant)