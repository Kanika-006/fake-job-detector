# FraudSense — Explainable Fake Job Posting Detection

An NLP-based machine learning project for detecting potentially fraudulent job postings and explaining the signals behind the model's prediction.

## Project Overview

Online job platforms can contain fraudulent job advertisements designed to mislead applicants or collect money and personal information.

This project builds an explainable classification system that combines:

- NLP-based machine learning
- TF-IDF text representation
- Linear SVM classification
- Deep learning with BiLSTM
- Rule-based suspicious-signal detection
- Model explainability
- Error analysis
- Streamlit deployment

The goal is not only to classify a job posting, but also to provide understandable evidence behind the prediction.

## Dataset

The project uses the Kaggle Fake Job Postings dataset.

The dataset contains job-posting information such as:

- Job title
- Location
- Company profile
- Description
- Requirements
- Benefits
- Employment type
- Required experience
- Required education
- Industry
- Function
- Fraudulent label

Target variable:

- `0` → Genuine
- `1` → Fraudulent

The dataset contains 17,880 job postings, with fraudulent postings representing a minority of the data.

## Project Workflow

Dataset  
↓  
EDA & Data Cleaning  
↓  
Train / Test Split  
↓  
Text Preprocessing  
↓  
TF-IDF + Logistic Regression  
↓  
TF-IDF + Linear SVM  
↓  
Structured Feature Engineering  
↓  
Hybrid Model Experiment  
↓  
BiLSTM Experiment  
↓  
Model Comparison  
↓  
Error Analysis  
↓  
Explainability  
↓  
Suspicious Signal Detection  
↓  
Streamlit Deployment

## Models

### 1. TF-IDF + Logistic Regression

A baseline NLP classification model using:

- TF-IDF
- Unigrams and bigrams
- Class-weight balancing
- Logistic Regression

### 2. TF-IDF + Linear SVM

A stronger traditional NLP model using:

- TF-IDF
- Unigrams and bigrams
- Class-weight balancing
- Linear SVM

### 3. Hybrid Model

Combines:

- TF-IDF text features
- Missing-value indicators
- Text statistics
- Salary-related features
- Categorical features

This experiment demonstrated that adding structured features does not automatically improve model performance.

### 4. BiLSTM

A deep-learning experiment using:

- Tokenization
- Sequence padding
- Embedding layer
- Bidirectional LSTM
- Dropout
- Class weighting
- Early stopping

The BiLSTM experiment was developed in Google Colab.

## Model Results

| Model | Accuracy | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 98.35% | 77.67% | 92.49% | 84.43% | 0.932 |
| Linear SVM | 99.13% | 94.38% | 87.28% | 90.69% | 0.951 |
| BiLSTM | 98.07% | 77.37% | 84.97% | 80.99% | 0.874 |
| Hybrid Model | 69.55% | 10.72% | 72.25% | 18.67% | — |

Results are based on the experiments performed in this project.

Because fraudulent postings are a minority class, accuracy alone is not sufficient for evaluating the models. Precision, recall, F1-score and PR-AUC were also considered.

## Explainability

The Linear SVM model provides feature-level explanations using TF-IDF feature contributions.

The project identifies terms that push predictions toward:

- Genuine
- Fraudulent

Local explanations can also show which terms in an individual posting contributed toward the model's decision.

These features represent model associations and should not be interpreted as causal evidence of fraud.

## Suspicious Signal Detection

In addition to the machine-learning model, the application performs transparent rule-based checks for patterns such as:

- Payment requests
- Requests for sensitive information
- Urgency language
- Easy-income or work-from-home language
- External links
- Very short job postings

These signals are separate from the ML prediction and are intended as additional warning indicators rather than proof of fraud.

## Error Analysis

The project examines false positives and false negatives to understand where the model makes mistakes.

Borderline predictions are also examined to identify job postings where the model has lower confidence.

## Streamlit Application

The project includes a Streamlit interface called **FraudSense**.

The application allows a user to:

1. Paste a job posting
2. Generate a prediction
3. View the model decision score
4. View explainability information
5. View suspicious warning signals

## Project Structure

```text
fake-job-detector/
│
├── app/
│   └── app.py
│
├── data/
│   └── fake_job_postings.csv
│
├── models/
│   ├── fake_job_svm.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model_evaluation.ipynb
│   └── 03_bilstm_experiment.ipynb
│
├── src/
│   └── predict.py
│
├── requirements.txt
└── README.md