# Email Spam Classifier

## Project Overview

This project develops a high-accuracy email spam classifier using Natural Language Processing (NLP) and machine learning. It employs a Support Vector Machine (SVM) model with TF-IDF features, optimized via GridSearchCV, achieving ~98-99% accuracy on the test set. The project addresses class imbalance using class weights in the SVM model, includes comprehensive text preprocessing, and is documented in a Jupyter Notebook and Python script for exploration. A Streamlit web app for real-time predictions is planned. Designed as a learning tool, it’s ideal for mastering NLP, machine learning pipelines, and model evaluation.

### Key Features

- **Dataset**: Spam Email Dataset (~5,728 emails, ~76% ham, ~24% spam) from Kaggle.
- **Preprocessing**: Text cleaning with NLTK (lowercase, punctuation removal, tokenization, stopword removal, lemmatization).
- **Model**: SVM classifier with TF-IDF vectorization, tuned with GridSearchCV for optimal hyperparameters (C, kernel, gamma, max_features, ngram_range, class_weight).
- **Class Imbalance**: Handled using class weights in the SVM model (e.g., 'balanced', custom ratios like 1:2 or 1:3 for ham:spam).
- **Evaluation**: Metrics include accuracy, precision, recall, F1-score (~98%+), and a confusion matrix visualization.
- **Web App**: Planned Streamlit interface for users to input email text and get spam/ham predictions (to be implemented in `app.py`).
- **Future Work**: Explore deep learning models (e.g., LSTM, BERT) for potentially higher accuracy (99%+), or add SMOTE for alternative imbalance handling.

### Deliverables

- Jupyter Notebook: `spam_classifier_model.ipynb` (data loading, preprocessing, modeling, evaluation, visualization).
- Python Script: `spam_classifier_model.py` (equivalent to notebook, for automation).
- Saved Models: `spam_classifier_model.joblib` (SVM model) and `tfidf_vectorizer.joblib` (TF-IDF vectorizer).
- Visualizations: Confusion matrix plot (`cm.png`).
- Web App: `app.py` (Streamlit script, to be added).
- Environment Files: `environment.yml` (for Conda) and `requirements.txt` (for pip).
- Old Versions: `spam_classifier_model_old.ipynb` and `spam_classifier_model_old.py` (archived previous code versions).
- This README.

## Prerequisites

- Python 3.9 (recommended for compatibility).
- Conda (Miniconda preferred) for environment management, or pip for virtual environments.
- Dataset: `emails.csv` from [Kaggle Spam Email Dataset](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset).
- GitHub account to clone the repository.

## Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/rathishTharusha/email-spam-classifier.git
   cd email-spam-classifier
   ```

2. **Set Up Environment**:

   - **Preferred: Using Conda** (for better dependency management, especially with scientific libraries):
     - Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if not already installed.
     - Create and activate the environment using `environment.yml`:

       ```bash
       conda env create -f environment.yml
       conda activate spam_detector
       ```

     - If `environment.yml` is not present or needs updating, manually install dependencies:

       ```bash
       conda create -n spam_detector python=3.9
       conda activate spam_detector
       conda install pandas numpy matplotlib scikit-learn nltk seaborn joblib jupyter
       pip install streamlit  # For future web app
       ```

   - **Alternative: Using pip and Virtual Environment** (if Conda is not preferred):
     - Create a virtual environment:

       ```bash
       python -m venv spam_detector
       source spam_detector/bin/activate  # On Linux/macOS
       # Or on Windows: spam_detector\Scripts\activate
       ```

     - Install dependencies from `requirements.txt`:

       ```bash
       pip install -r requirements.txt
       ```

     - If `requirements.txt` needs updating, it should include: pandas, numpy, nltk, scikit-learn, matplotlib, seaborn, joblib, streamlit. You can generate or update it with:

       ```bash
       pip freeze > requirements.txt
       ```

3. **Download the Dataset**:

   - Download `emails.csv` from [Kaggle](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset).
   - Place it in the project root folder (same level as `spam_classifier_model.ipynb`).

4. **Verify Setup**:

   - Check library installations (adjust for your environment):

     ```bash
     python -c "import pandas, numpy, sklearn, nltk, matplotlib, seaborn, joblib, streamlit; print('All good!')"
     ```

   - If errors occur, install missing libraries with `conda install <library>` (in Conda) or `pip install <library>`.

## Usage

### Running the Jupyter Notebook or Python Script

1. Start Jupyter Notebook:

   ```bash
   cd path/to/project
   jupyter notebook
   ```

2. Open `spam_classifier_model.ipynb` and run all cells (Shift+Enter) to:
   - Load and preprocess `emails.csv`.
   - Train an SVM model with TF-IDF features using a Pipeline and GridSearchCV (including class weights for imbalance).
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

1. Ensure the environment is active (`conda activate spam_detector` or virtual env).
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
  - Model performance: ~98-99% accuracy, precision, recall, F1-score (check notebook output).
  - Confusion matrix plot: Saved as `cm.png`.
  - Sample prediction: "Win a free iPhone now!" → Spam.
- **Web App**: Will display "Spam" or "Ham" (to be implemented).

## Project Structure

```
email-spam-classifier/
├── .gitignore                   # Git ignore file
├── cm.png                       # Confusion matrix plot
├── emails.csv                   # Dataset (download from Kaggle)
├── environment.yml              # Conda environment file
├── README.md                    # This file
├── requirements.txt             # Pip requirements file
├── spam_classifier_model_old.ipynb  # Archived old Jupyter Notebook
├── spam_classifier_model_old.py     # Archived old Python script
├── spam_classifier_model.ipynb  # Current Jupyter Notebook
├── spam_classifier_model.joblib # Saved SVM model
├── spam_classifier_model.py     # Current Python script
├── tfidf_vectorizer.joblib      # Saved TF-IDF vectorizer
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
- **Libraries**: pandas, numpy, scikit-learn, NLTK, matplotlib, seaborn, joblib, Streamlit.
- **Author**: Rathish Tharusha (GitHub: [rathishTharusha](https://github.com/rathishTharusha)).
- **Purpose**: Built as a learning project to master NLP and machine learning.

## Contact

For questions, open a GitHub issue or contact [Rathish Tharusha](https://github.com/rathishTharusha).