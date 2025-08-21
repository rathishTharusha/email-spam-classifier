# Email Spam Classifier

## Project Overview

This project develops a high-accuracy email spam classifier using Natural Language Processing (NLP) and machine learning. It employs a Support Vector Machine (SVM) model with TF-IDF features, optimized via GridSearchCV, achieving ~98-99% accuracy on the test set. The project addresses class imbalance using SMOTE, includes comprehensive text preprocessing, and is documented in a Jupyter Notebook and Python script for exploration. A Streamlit web app for real-time predictions is planned. Designed as a learning tool, it’s ideal for mastering NLP, machine learning pipelines, and model evaluation.

### Key Features

- **Dataset**: Spam Email Dataset (~5,728 emails, ~76% ham, ~24% spam) from Kaggle.
- **Preprocessing**: Text cleaning with NLTK (lowercase, punctuation removal, tokenization, stopword removal, lemmatization).
- **Model**: SVM classifier with TF-IDF vectorization, tuned with GridSearchCV for optimal hyperparameters (C, kernel, gamma, max_features, ngram_range).
- **Class Imbalance**: Handled using SMOTE to oversample the minority class (spam).
- **Evaluation**: Metrics include accuracy, precision, recall, F1-score (~98%+), and a confusion matrix visualization.
- **Web App**: Planned Streamlit interface for users to input email text and get spam/ham predictions (to be implemented in `app.py`).
- **Future Work**: Explore deep learning models (e.g., LSTM, BERT) for potentially higher accuracy (99%+).

### Deliverables

- Jupyter Notebook: `spam_classifier_model.ipynb` (data loading, preprocessing, modeling, evaluation, visualization).
- Python Script: `spam_classifier_model.py` (equivalent to notebook, for automation).
- Saved Models: `spam_classifier_model.joblib` (SVM model) and `tfidf_vectorizer.joblib` (TF-IDF vectorizer).
- Visualizations: Confusion matrix plot (`cm.png`).
- Web App: `app.py` (Streamlit script, to be added).
- Environment File: `environment.yml` for reproducible setup.
- This README.

## Prerequisites

- Python 3.9 (recommended for compatibility).
- Conda (Miniconda preferred) for environment management.
- Dataset: `emails.csv` from [Kaggle Spam Email Dataset](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset).
- GitHub account to clone the repository.

## Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/rathishTharusha/email-spam-classifier.git
   cd email-spam-classifier
   ```

2. **Set Up Conda Environment**:

   - Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if not already installed.
   - Create and activate the environment:

     ```bash
     conda env create -f environment.yml
     conda activate spam_detector
     ```

   - Or manually create:

     ```bash
     conda create -n spam_detector python=3.9
     conda activate spam_detector
     conda install pandas matplotlib scikit-learn nltk jupyter imbalanced-learn
     pip install streamlit  # For future web app
     ```

3. **Download the Dataset**:

   - Download `emails.csv` from [Kaggle](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset).
   - Place it in the project root folder (same level as `spam_classifier_model.ipynb`).

4. **Verify Setup**:

   - Check library installations:

     ```bash
     python -c "import pandas, matplotlib, sklearn, nltk, imblearn, streamlit; print('All good!')"
     ```

   - If errors occur, install missing libraries with `conda install <library>` or `pip install <library>`.

## Usage

### Running the Jupyter Notebook or Python Script

1. Start Jupyter Notebook:

   ```bash
   cd path/to/project
   jupyter notebook
   ```

2. Open `spam_classifier_model.ipynb` and run all cells (Shift+Enter) to:
   - Load and preprocess `emails.csv`.
   - Apply SMOTE to balance classes.
   - Train an SVM model with TF-IDF features using a Pipeline and GridSearchCV.
   - Evaluate on test set (accuracy, precision, recall, F1-score).
   - Visualize confusion matrix (saved as `cm.png`).
   - Save model and vectorizer (`spam_classifier_model.joblib`, `tfidf_vectorizer.joblib`).
   - Test a sample email (e.g., "Win a free iPhone now!" → Spam).

3. Alternatively, run the Python script:

   ```bash
   python spam_classifier_model.py
   ```

   - Outputs similar results to the notebook, including metrics and saved files.

### Running the Web App (Planned)

1. Ensure the environment is active (`conda activate spam_detector`).
2. Once `app.py` is implemented, run:

   ```bash
   streamlit run app.py
   ```

3. Open the URL (e.g., `http://localhost:8501`) in your browser.
4. Enter email text and click "Predict" to see if it’s spam or ham.

### Expected Outputs

- **Notebook/Script**:
  - Dataset: ~5,728 rows, columns (`text`, `spam`).
  - Class distribution: ~4,360 ham (0), ~1,368 spam (1).
  - Post-SMOTE: Balanced classes (~3,504 each).
  - Model performance: ~98-99% accuracy, precision, recall, F1-score (check notebook output).
  - Confusion matrix plot: Saved as `cm.png`.
  - Sample prediction: "Win a free iPhone now!" → Spam.
- **Web App**: Will display "Spam" or "Ham" (to be implemented).

## Project Structure

```
email-spam-classifier/
├── emails.csv                   # Dataset (download from Kaggle)
├── spam_classifier_model.ipynb  # Jupyter Notebook
├── spam_classifier_model.py     # Python script
├── spam_classifier_model.joblib # Saved SVM model
├── tfidf_vectorizer.joblib     # Saved TF-IDF vectorizer
├── cm.png                      # Confusion matrix plot
├── app.py                      # Streamlit app (to be added)
├── environment.yml             # Conda environment file
└── README.md                   # This file
```

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Ideas for contributions:
- Implement the Streamlit app (`app.py`) with confidence scores or batch processing.
- Add deep learning models (e.g., LSTM, BERT) for higher accuracy.
- Enhance preprocessing with word embeddings (e.g., Word2Vec, GloVe).
- Add visualizations like ROC curves or word clouds.

Please follow the code style in `spam_classifier_model.py` and include tests for new features.

## Future Improvements

- **Advanced Models**: Experiment with Random Forest, XGBoost, or BERT for 99%+ accuracy.
- **Preprocessing**: Use word embeddings (e.g., GloVe, BERT embeddings) for richer text representation.
- **Web App**: Implement `app.py` with features like prediction confidence or email header analysis.
- **Deployment**: Host on Heroku or Streamlit Community Cloud for public access.

## Acknowledgments

- **Dataset**: [Spam Email Dataset](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset).
- **Libraries**: pandas, scikit-learn, NLTK, matplotlib, imbalanced-learn, Streamlit.
- **Author**: Rathish Tharusha (GitHub: [rathishTharusha](https://github.com/rathishTharusha)).
- **Purpose**: Built as a learning project to master NLP and machine learning.

## Contact

For questions, open a GitHub issue or contact [Rathish Tharusha](https://github.com/rathishTharusha).