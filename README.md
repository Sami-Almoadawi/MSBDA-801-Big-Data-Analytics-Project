# Scalable Real-time Detection of AI-Generated Arabic Text

## Project Overview
This project presents a scalable distributed pipeline for detecting AI-generated Arabic text using Apache Spark and machine learning techniques.

The system includes:
- Distributed data preprocessing
- Arabic NLP cleaning
- TF-IDF feature engineering
- Stylometric feature extraction
- Machine learning classification
- Spark Structured Streaming simulation

## Technologies Used
- Python
- PySpark
- Apache Spark MLlib
- Scikit-learn
- Google Colab

## Dataset
KFUPM-JRCAI Arabic Generated Abstracts Dataset:
https://huggingface.co/datasets/KFUPM-JRCAI/arabic-generated-abstracts

## Models Used
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

## Best Model Performance
- Accuracy: 97.3%
- F1-score: 97.3%
- ROC-AUC: 99.1%

## Project Structure
The repository is organized as follows:

 ```

project-root/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   └── Arabic_AI_Text_Detection_Project.ipynb
├── reports/
│   ├── figures/
│   └── final project reports/
│   └── presentations/
├── src/
│   ├── data_preparation.py
│   ├── modeling.py
│   ├── streaming_pipeline.py
│   └── utils.py
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt

```

---
