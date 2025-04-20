---

# 📞 Scam Call Detection using Machine Learning

This project uses multiple datasets from Hugging Face to train three machine learning models — Naive Bayes, Support Vector Machine (SVM), and Random Forest — to detect scam conversations. It includes preprocessing logic to normalize differently structured datasets and evaluate model performance on a separate test dataset.

## 📁 Project Structure

```bash
.
├── combine_and_preprocess.py        # Script to load and combine all Hugging Face datasets
├── train_models.py                  # Trains Naive Bayes, SVM, and Random Forest models
├── test_models.py                   # Evaluates saved models on a separate balanced test dataset
├── generate_graphs.py               # Generates confusion matrix and ROC curve graphs
├── models/
│   ├── naive_bayes_model.joblib
│   ├── svm_model.joblib
│   └── random_forest_model.joblib
├── data/
│   ├── combined_dataset.csv         # Combined and cleaned dataset from Hugging Face sources
│   └── balanced_dataset.csv         # Separate dataset for testing models
├── plots/
│   ├── Naive_Bayes_confusion_matrix.png
│   ├── SVM_confusion_matrix.png
│   ├── Random_Forest_confusion_matrix.png
│   ├── Naive_Bayes_roc_curve.png
│   ├── SVM_roc_curve.png
│   └── Random_Forest_roc_curve.png
└── README.md
```

---

## 📦 Requirements

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn datasets joblib
```

---

## ⚙️ Steps to Run the Project

### 1. Combine and Preprocess All Datasets

```bash
python combine_and_preprocess.py
```

This script:
- Loads multiple Hugging Face datasets.
- Extracts text and labels (from differently named columns).
- Normalizes the labels to 0 (Not Scam) and 1 (Scam).
- Saves the cleaned data as `combined_dataset.csv`.

---

### 2. Train Models on the Combined Dataset

```bash
python train_models.py
```

This script:
- Loads `combined_dataset.csv`.
- Trains Naive Bayes, SVM, and Random Forest classifiers.
- Saves the models in the `models/` folder.

---

### 3. Test Models on a Separate Dataset

Place your test file (e.g., `balanced_dataset.csv`) in the `data/` folder with columns:
- `text`: The message/conversation.
- `label`: Either 0 or 1.

Then run:

```bash
python test_models.py
```

This script:
- Loads the balanced test dataset.
- Loads the trained models from the `models/` directory.
- Evaluates them using accuracy and classification report.

---

### 4. Generate Visualizations

```bash
python generate_graphs.py
```

This script:
- Generates and saves confusion matrix heatmaps and ROC curves for all three models.
- Outputs them to the `plots/` directory.

---

## 🧠 Models Used

- **Naive Bayes**: Good baseline for text classification.
- **SVM**: High-performing linear classifier.
- **Random Forest**: Ensemble model that improves accuracy and robustness.

---

## 📊 Results Snapshot

Model | Validation Accuracy | Test Accuracy
------|----------------------|---------------
Naive Bayes | ~95.6% | ~72.0%
SVM         | ~98.5% | ~95.2%
Random Forest | ~98.3% | ~95.4%

---

## 📎 Notes

- Different datasets use different label names and formats — this was handled in preprocessing.
- `final_scam` and others had unique schemas that were mapped to a unified format: `text` and `label`.
- All test evaluations are done on a **clean, balanced**, and **separate** dataset not seen during training.

---
