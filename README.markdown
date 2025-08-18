# Email Spam Classifier

## Project Overview

This project builds an email spam classifier using Natural Language Processing (NLP) techniques to distinguish spam emails from non-spam (ham). It achieves high accuracy (\~98-99%) using a Multinomial Naive Bayes model with TF-IDF features, with plans to explore advanced models like BERT for potentially higher accuracy (99%+). The project includes a Jupyter Notebook for data processing, model training, and evaluation, plus a Streamlit web app for real-time spam detection. It’s designed as a learning tool for understanding NLP and machine learning, with clear documentation for reproducibility.

### Key Features

- **Dataset**: Uses the Spam Email Dataset (\~5,728 emails, \~76% ham, \~24% spam).
- **Preprocessing**: Text cleaning with NLTK (tokenization, stemming, stopword removal).
- **Model**: Multinomial Naive Bayes with TF-IDF vectorization; achieves \~98% accuracy, precision, recall, and F1-score on test data.
- **Evaluation**: Includes confusion matrix visualization and metrics (accuracy, precision, recall, F1-score).
- **Web App**: A Streamlit interface where users input email text to get spam/ham predictions.
- **Future Work**: Option to integrate advanced models like LSTM or BERT for higher accuracy.

### Deliverables

- Jupyter Notebook: `spam_classifier_model.ipynb` (data loading, preprocessing, modeling, visualization).
- Saved Models: `spam_classifier_model.joblib` (Naive Bayes) and `tfidf_vectorizer.joblib` (TF-IDF vectorizer).
- Visualizations: Confusion matrix plot (`cm.png`).
- Web App: `app.py` (Streamlit script for interactive predictions).
- This README and environment setup file (`environment.yml`).

## Prerequisites

To run this project, you need:

- Python 3.9 (recommended for compatibility).
- Conda (Miniconda preferred) for environment management.
- The dataset (`emails.csv`) from Kaggle.
- A GitHub account to clone the repository.

## Setup Instructions

Follow these steps to set up the project locally.

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/email-spam-classifier.git
   cd email-spam-classifier
   ```

2. **Set Up Conda Environment**:

   - Install Miniconda if not already installed.
   - Create and activate the environment:

     ```bash
     conda env create -f environment.yml
     conda activate spam_detector
     ```
   - Alternatively, manually create the environment:

     ```bash
     conda create -n spam_detector python=3.9
     conda activate spam_detector
     conda install pandas matplotlib scikit-learn nltk jupyter streamlit
     ```

3. **Download the Dataset**:

   - Go to the Kaggle dataset page, sign in, and download `emails.csv`.
   - Place `emails.csv` in the project root folder (same level as `spam_classifier_model.ipynb`).

4. **Verify Setup**:

   - Run `python -c "import pandas, matplotlib, sklearn, nltk, streamlit; print('All good!')"` to check library installations.
   - If errors occur, install missing libraries with `conda install <library>` or `pip install <library>`.

## Usage

### Running the Jupyter Notebook

1. Start Jupyter Notebook:

   ```bash
   cd path/to/project
   jupyter notebook
   ```
2. Open `spam_classifier_model.ipynb` in the browser.
3. Run all cells (Shift+Enter) to:
   - Load and preprocess the dataset.
   - Train the Naive Bayes model.
   - Evaluate performance (accuracy, precision, recall, F1-score).
   - Generate a confusion matrix plot (`cm.png`).
   - Save the model and vectorizer (`spam_classifier_model.joblib`, `tfidf_vectorizer.joblib`).
4. Test a sample email (e.g., "Win a free iPhone now!") to see a prediction.

### Running the Web App

1. Ensure the environment is active (`conda activate spam_detector`).
2. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```
3. Open the provided URL (e.g., `http://localhost:8501`) in your browser.
4. Enter an email text in the text area and click "Predict" to see if it’s spam or ham.

### Expected Outputs

- **Notebook**:
  - Dataset stats: \~5,728 rows, 2 columns (`text`, `spam`).
  - Class distribution: \~4,360 ham (0), \~1,368 spam (1).
  - Model performance: \~98% accuracy, precision, recall, F1-score.
  - Confusion matrix plot saved as `cm.png`.
- **Web App**: Displays "Spam" or "Ham" based on input text.

## Future Improvements

- **Model Upgrades**: Experiment with SVM, Random Forest, or BERT for higher accuracy (target 99%+).
- **Preprocessing**: Use lemmatization or word embeddings (e.g., Word2Vec, GloVe) for better text representation.
- **Web App**: Add features like confidence scores or batch email processing.
- **Deployment**: Host on Heroku or Streamlit Sharing for public access.

## Acknowledgments

- Dataset: Spam Email Dataset.
- Libraries: pandas, scikit-learn, NLTK, matplotlib, Streamlit.
- Built as part of a learning project to master NLP and machine learning.

## Contact

For questions or issues, open a GitHub issue or contact \[your-email@example.com\].