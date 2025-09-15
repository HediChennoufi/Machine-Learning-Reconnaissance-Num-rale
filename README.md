# Introduction to Machine Learning - Digits Classification

This project is a comprehensive study and implementation of machine learning techniques applied to the classification of handwritten digits using the `digits` dataset from `sklearn`.

## Project Overview

The goal is to explore, preprocess, and model the handwritten digits dataset to classify images of digits (0-9) with high accuracy.

Key steps include:  
- Data loading and visualization  
- Data normalization and dimensionality reduction (PCA)  
- Feature engineering (zone-based and edge detection features)  
- Building and evaluating models including Support Vector Machine (SVM) and Neural Networks  
- Hyperparameter tuning and cross-validation  
- Comparison of multi-class strategies (One-vs-Rest, One-vs-One)

## Project Structure

- `partie1Metier.py` - Data loading, exploration, preprocessing, PCA and feature engineering  
- `partie2Metier.py` - Dataset splitting, pipeline creation, SVM training, evaluation, hyperparameter tuning, and comparison of OvO/OvR strategies  
- `partie3Metier.py` - Neural network model using Keras for digit classification  
- `RapportMachineLearning.pdf` - Detailed project report including methodology, results, discussions, and conclusions  

## Requirements

- Python 3  
- numpy  
- matplotlib  
- scikit-learn  
- tensorflow (for neural network)  
- scikit-image  

## How to run

1. Install dependencies (e.g., via pip):  # Introduction to Machine Learning - Digits Classification

This project is a comprehensive study and implementation of machine learning techniques applied to the classification of handwritten digits using the `digits` dataset from `sklearn`.

## Project Overview

The goal is to explore, preprocess, and model the handwritten digits dataset to classify images of digits (0-9) with high accuracy.

Key steps include:  
- Data loading and visualization  
- Data normalization and dimensionality reduction (PCA)  
- Feature engineering (zone-based and edge detection features)  
- Building and evaluating models including Support Vector Machine (SVM) and Neural Networks  
- Hyperparameter tuning and cross-validation  
- Comparison of multi-class strategies (One-vs-Rest, One-vs-One)

## Project Structure

- `partie1Metier.py` - Data loading, exploration, preprocessing, PCA and feature engineering  
- `partie2Metier.py` - Dataset splitting, pipeline creation, SVM training, evaluation, hyperparameter tuning, and comparison of OvO/OvR strategies  
- `partie3Metier.py` - Neural network model using Keras for digit classification  
- `RapportMachineLearning.pdf` - Detailed project report including methodology, results, discussions, and conclusions  

## Requirements

- Python 3  
- numpy  
- matplotlib  
- scikit-learn  
- tensorflow (for neural network)  
- scikit-image  


Refer to the PDF report for detailed explanations, results, and analysis.

---

## Results Summary

- The best model achieved approximately **99% accuracy** using an optimized SVM with RBF kernel, PCA, and combined handcrafted features (zones + Sobel edges).  
- Neural networks showed promising results as well with clear training and validation loss convergence.  
- Extensive hyperparameter tuning and comparison of multi-class strategies (OvR vs OvO) are included.  

---

## Authors

- Amiel Metier  
- Hedi Chennoufi  
- Lucas Fert  

Supervised by:  
- Jean-Luc Bouchot  
- Mahmoud Elsawy  



