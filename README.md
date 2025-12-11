# Cross-Validation Explained: How K-Fold Stabilises Model Performance Estimates

**Author:** Pravinya Rayalapati  
**Student ID:** 24099441

## Overview

This project demonstrates the importance and implementation of K-Fold cross-validation in machine learning model evaluation. Using the Breast Cancer Wisconsin dataset, the notebook compares single train-test splits with K-Fold cross-validation to show how cross-validation provides more stable and reliable performance estimates.

## Project Objectives

- Compare single train-test split with K-Fold cross-validation
- Demonstrate how performance estimates vary with different random splits
- Analyze the effect of different K values (3, 5, 10) on model evaluation
- Compare Logistic Regression and Random Forest classifiers using cross-validation
- Visualize fold-wise scores and variability of accuracy estimates

## Dataset

**Breast Cancer Wisconsin Dataset**
- **Samples:** 569 examples
- **Features:** 30 numeric features (texture, radius, smoothness, etc.)
- **Target:** Binary classification
  - `0` – malignant tumour
  - `1` – benign tumour

This dataset is ideal for demonstrating cross-validation because:
- It's widely used in machine learning literature
- The task is clinically meaningful (cancer diagnosis)
- It's large enough to illustrate variability in performance estimates

## Key Concepts Demonstrated

### 1. Single Train-Test Split
- Traditional approach: split data once into training and test sets
- Shows variability in accuracy estimates based on random state
- Demonstrates the instability of single-split evaluation

### 2. K-Fold Cross-Validation
- Splits data into K folds
- Trains on K-1 folds and tests on the remaining fold
- Repeats for each fold, ensuring every sample is used for both training and testing
- Provides mean and standard deviation of performance across folds

### 3. Effect of K Values
- Explores trade-offs between different K values (3, 5, 10)
- Smaller K: fewer models to train but higher variance
- Larger K: more stable estimates but more computation

### 4. Model Comparison
- Fair comparison of Logistic Regression vs Random Forest
- Uses 5-fold CV for both models
- Visualizes performance distributions using box plots

## Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Open `notebook17.ipynb` in Jupyter Notebook or JupyterLab
2. Run all cells sequentially to reproduce the analysis
3. The notebook includes:
   - Data loading and exploration
   - Single train-test split baseline
   - K-Fold cross-validation implementation
   - Visualization of results
   - Model comparison

## Dependencies

- **numpy** - Numerical computing
- **pandas** - Data manipulation and analysis
- **matplotlib** - Plotting and visualization
- **seaborn** - Statistical data visualization
- **scikit-learn** - Machine learning algorithms and utilities

See `requirements.txt` for specific versions.

## Project Structure

```
.
├── notebook17.ipynb      # Main Jupyter notebook with analysis
├── README.md             # This file
└── requirements.txt      # Python package dependencies
```

## Key Findings

1. **Single Split Variability:** Accuracy estimates from single train-test splits vary significantly with different random states, demonstrating the need for more robust evaluation methods.

2. **Cross-Validation Stability:** K-Fold cross-validation provides more stable performance estimates by averaging across multiple folds, reducing the impact of a single "lucky" or "unlucky" split.

3. **K Value Impact:** Different K values affect both mean accuracy and standard deviation. Larger K values generally provide more stable estimates but require more computation.

4. **Model Comparison:** Cross-validation enables fair comparison between different models (Logistic Regression vs Random Forest) by evaluating them on the same data splits.

## Visualizations

The notebook includes several visualizations:
- Class distribution in the dataset
- Variability of single train-test split accuracy across different random states
- Fold-wise accuracy scores for 5-Fold CV
- Mean CV accuracy vs K values
- Standard deviation of CV estimates vs K values
- Box plots comparing Logistic Regression and Random Forest performance

## Methodology

1. **Data Preprocessing:** StandardScaler is used to normalize features before training Logistic Regression
2. **Model Pipeline:** Scikit-learn Pipeline combines preprocessing and model training
3. **Cross-Validation:** KFold with shuffling and fixed random state for reproducibility
4. **Evaluation Metric:** Accuracy score for binary classification
5. **Visualization:** Matplotlib and Seaborn for creating informative plots

## Notes

- All random seeds are set to 42 for reproducibility
- Stratified splitting is used to maintain class distribution
- The Logistic Regression model uses LBFGS solver with max_iter=500
- Random Forest uses 200 estimators with no maximum depth limit

## References

- Breast Cancer Wisconsin Dataset from scikit-learn
- Scikit-learn documentation on cross-validation
- Machine learning best practices for model evaluation

## License

This project is part of an academic assignment for Machine Learning Neural Networks course.

## Contact

**Pravinya Rayalapati**  
Student ID: 24099441


