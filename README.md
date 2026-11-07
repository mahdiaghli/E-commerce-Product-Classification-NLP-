# E-commerce Product Classification Using NLP

## Overview
This project builds and evaluates a machine learning pipeline for classifying e-commerce product descriptions into product categories using natural language processing (NLP). The goal is to automatically assign each product to its correct category from raw textual product data.

## Problem Statement
E-commerce platforms contain large catalogs with product titles and descriptions written in natural language. Manually categorizing thousands of listings is time-consuming and error-prone. This project explores how text-based features and classical machine learning models can automate this task with high accuracy.

## Dataset
The project uses the dataset stored in `ecommerceDataset.csv`.

- Total rows: 50,425
- Columns: 2
- Categories: 4
  - Household
  - Books
  - Electronics
  - Clothing & Accessories

The original dataset was cleaned and normalized to produce `ecommerce_preprocessed.csv` before model training.

## Project Workflow
The project is implemented in `code.ipynb` and follows a multi-phase workflow:

### Phase 1: Text preprocessing
- Load raw product data
- Detect header issues and normalize structure
- Clean text by removing URLs, emails, HTML tags, punctuation, and extra whitespace
- Convert to lowercase
- Remove common stopwords
- Drop empty or duplicate text entries

### Phase 2: Feature engineering
Two representation strategies were evaluated:
- TF-IDF vectorization
- Mean Word2Vec embeddings

The generated outputs are saved in `phase2_outputs/`:
- `X_tfidf.npz` / `X_tfidf.npy`
- `X_w2v_mean.npy`
- `labels.npy`
- `tfidf_vocab.json`
- `gensim_w2v.model`

### Phase 3: Model training and evaluation
The following classifiers were trained and compared:
- Logistic Regression
- Linear SVC
- Random Forest

Evaluation metrics included:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Per-class classification report

### Phase 4: Statistical validation and comparison
The project also performs deeper evaluation using:
- bootstrap confidence intervals
- McNemar tests for paired model comparison
- model comparison summaries

Outputs are stored in `phase4_outputs/`.

## Best Model Performance
The best-performing model was:

- Representation: TF-IDF
- Model: LinearSVC

Results:

| Metric | Value |
|---|---:|
| Accuracy | 0.9557650471356055 (95.58%) |
| Precision | 0.9558124739164566 (95.58%) |
| Recall | 0.9557650471356055 (95.58%) |
| F1-score | 0.95571122942488 (95.57%) |

Other evaluated models:

- TF-IDF + Logistic Regression: 95.09% accuracy
- TF-IDF + Random Forest: 92.88% accuracy
- W2V + Logistic Regression: 90.79% accuracy
- W2V + LinearSVC: 90.81% accuracy
- W2V + RandomForest: 91.70% accuracy

## Repository Structure

```text
.
├── code.ipynb                   # Main project notebook with all phases
├── ecommerceDataset.csv         # Raw dataset
├── ecommerce_preprocessed.csv   # Cleaned dataset
├── phase2_outputs/             # Feature extraction outputs
├── phase3_outputs/             # Model evaluation results
├── phase4_outputs/             # Advanced comparison and validation results
├── README.md                   # Project documentation
├── report.docx                 # Project report
├── project5.pdf                 # Project summary PDF
└── archive.zip / project5.zip  # Archived project files
```

## Tools and Libraries
The project uses Python libraries including:
- pandas
- NumPy
- scikit-learn
- Gensim
- matplotlib
- scipy

## How to Run
Open `code.ipynb` in Jupyter Notebook or VS Code and run the cells in order:

1. Phase 1: preprocessing
2. Phase 2: TF-IDF and Word2Vec generation
3. Phase 3: model training and evaluation
4. Phase 4: model comparisons and validation

Alternatively, you can reuse the output files in `phase2_outputs/`, `phase3_outputs/`, and `phase4_outputs/` if the pipeline has already been executed.

## Key Takeaways
- TF-IDF features performed significantly better than mean Word2Vec embeddings for this dataset.
- LinearSVC was the strongest-performing classifier in this setup.
- High-quality preprocessing played a crucial role in improving model accuracy.
- The pipeline is suitable for large-scale product categorization and can be extended to more categories or multilingual text.

## Future Improvements
Potential enhancements include:
- Hyperparameter tuning for Logistic Regression and LinearSVC
- Advanced text cleaning with lemmatization
- Class imbalance handling
- Transformer-based models such as BERT or DistilBERT
- Saving the trained model for deployment in a real application or API

## Conclusion
This project demonstrates that classical NLP techniques combined with carefully tuned machine learning models can achieve excellent classification performance for e-commerce product categories. The best model reached approximately 95.6% accuracy, showing strong potential for practical automated product categorization.
