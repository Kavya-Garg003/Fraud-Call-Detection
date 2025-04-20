# 📞 Scam Call Detection using Machine Learning

This project uses multiple datasets from Hugging Face to train three machine learning models — Naive Bayes, Support Vector Machine (SVM), and Random Forest — to detect scam conversations. It includes preprocessing logic to normalize differently structured datasets and evaluate model performance on a separate test dataset. All the code, including training, testing, and visualization, is contained in a single notebook.

---

## 📁 Project Structure

```bash
.
├── Training_and_Testing.ipynb       # Main notebook: includes dataset loading, model training, testing, and visualization
├── scam_call/                       # Frontend Code (React interface)
├── models/
│   ├── naive_bayes_model.joblib     # Saved Naive Bayes model
│   ├── svm_model.joblib             # Saved SVM model
│   └── random_forest_model.joblib   # Saved Random Forest model
├── datasets/
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

## ⚙️ How to Run the Project

### 1. Open the Jupyter Notebook

Launch Jupyter Notebook or use any environment like VS Code or Google Colab, and open:

```bash
Training_and_Testing.ipynb
```

---

### 2. Run All Cells

- The notebook will:
  - Load and combine multiple scam-related datasets from Hugging Face.
  - Preprocess and normalize them into a unified format.
  - Train three models: Naive Bayes, SVM, and Random Forest.
  - Evaluate them on a clean, balanced test dataset (`balanced_dataset.csv`).
  - Generate confusion matrices and ROC curves for all models.
  - Save the models and plots to their respective directories.

---

## 🧠 Models Used

- **Naive Bayes**: A fast probabilistic classifier suitable for text.
- **SVM**: Excellent generalization for high-dimensional data like TF-IDF vectors.
- **Random Forest**: A robust ensemble model that prevents overfitting.

---

## 📊 Results Snapshot

Model | Validation Accuracy | Test Accuracy
------|----------------------|---------------
Naive Bayes | ~95.6% | ~72.0%
SVM         | ~98.5% | ~95.2%
Random Forest | ~98.3% | ~95.4%

---

## 📎 Notes

- The notebook handles datasets with varying schemas and label formats.
- Unified schema: All datasets are transformed to use `text` and `label` columns where `label` is either 0 (Not Scam) or 1 (Scam).
- A separate, balanced dataset (`balanced_dataset.csv`) is used for evaluation to avoid data leakage.
- Plots for each model are exported as `.png` files to the `plots/` directory.
