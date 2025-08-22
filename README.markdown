# Email Spam Classifier

This project develops a high-accuracy email spam classifier using Natural Language Processing (NLP) and machine learning. It employs a Support Vector Machine (SVM) model with TF-IDF features, optimized via GridSearchCV, achieving ~98-99% accuracy on the test set. The classifier addresses class imbalance using class weights in the SVM model. A Flask-based web application allows users to input email text and receive real-time spam or ham (not spam) predictions. The project is designed as a learning tool for mastering NLP, machine learning pipelines, and model evaluation, with a simple online platform for practical use.

## Project Overview

- **Dataset**: Spam Email Dataset (~5,728 emails, ~76% ham, ~24% spam) from [Kaggle](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset).
- **Preprocessing**: Text cleaning with NLTK, including:
  - Converting to lowercase
  - Removing punctuation and special characters
  - Tokenization
  - Stopword removal
  - Lemmatization
- **Model**: SVM classifier with TF-IDF vectorization, tuned with GridSearchCV for optimal hyperparameters (C, kernel, gamma, max_features, ngram_range, class_weight).
- **Class Imbalance**: Handled using class weights in the SVM model (e.g., 'balanced', custom ratios like 1:2 or 1:3 for ham:spam).
- **Evaluation**: Metrics include accuracy, precision, recall, F1-score (~98%+), and a confusion matrix visualization.
- **Web App**: A Flask-based interface for users to input email text and get spam/ham predictions in real time.
- **Future Work**: Explore deep learning models (e.g., LSTM, BERT) for potentially higher accuracy (99%+), or add SMOTE for alternative imbalance handling.

## Deliverables

- **Jupyter Notebook**: `spam_classifier_model.ipynb` (data loading, preprocessing, modeling, evaluation, visualization)
- **Python Script**: `spam_classifier_model.py` (equivalent to notebook, for automation)
- **Saved Models**: `spam_classifier_model.joblib` (SVM model), `tfidf_vectorizer.joblib` (TF-IDF vectorizer)
- **Visualizations**: Confusion matrix plot (`cm.png`)
- **Web App**: `app.py` (Flask script), `index.html` (HTML template), `style.css` (CSS styling), `preprocess.py` (preprocessing function)
- **Environment Files**: `environment.yml` (Conda), `requirements.txt` (pip)
- **Old Versions**: `spam_classifier_model_old.ipynb`, `spam_classifier_model_old.py` (archived code)
- **GitHub Repository**: [https://github.com/rathishTharusha/email-spam-classifier](https://github.com/rathishTharusha/email-spam-classifier)
- **This README**

## Prerequisites

- **Python 3.9**: Recommended for compatibility with the provided environment.
- **Conda**: Miniconda preferred for environment management, or pip for virtual environments.
- **Dataset**: `emails.csv` from [Kaggle Spam Email Dataset](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset).
- **GitHub**: Clone the repository to access the code.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/rathishTharusha/email-spam-classifier.git
   cd email-spam-classifier
   ```

2. **Set Up Environment**:
   - **Preferred: Using Conda** (for better dependency management):
     ```bash
     conda env create -f environment.yml
     conda activate spam_detector
     ```
     If issues occur, manually install dependencies:
     ```bash
     conda create -n spam_detector python=3.9
     conda activate spam_detector
     conda install pandas numpy scikit-learn nltk matplotlib seaborn joblib
     pip install flask
     ```
   - **Alternative: Using pip and Virtual Environment**:
     ```bash
     python -m venv spam_detector
     source spam_detector/bin/activate  # On Linux/macOS
     # Or on Windows: spam_detector\Scripts\activate
     pip install -r requirements.txt
     ```

3. **Download the Dataset**:
   - Download `emails.csv` from [Kaggle](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset).
   - Place it in the project root folder (same level as `spam_classifier_model.ipynb`).

4. **Verify Setup**:
   - Check library installations:
     ```bash
     python -c "import pandas, numpy, sklearn, nltk, matplotlib, seaborn, joblib, flask; print('All good!')"
     ```
   - If errors occur, install missing libraries with `conda install <library>` or `pip install <library>`.

## Usage

### Running the Jupyter Notebook or Python Script
1. **Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
   - Open `spam_classifier_model.ipynb`.
   - Run all cells to:
     - Load and preprocess `emails.csv`.
     - Train the SVM model with TF-IDF features using a Pipeline and GridSearchCV.
     - Evaluate on the test set (accuracy, precision, recall, F1-score).
     - Visualize the confusion matrix (saved as `cm.png`).
     - Save the model and vectorizer (`spam_classifier_model.joblib`, `tfidf_vectorizer.joblib`).
     - Test a sample email (e.g., "Win a free iPhone now!" → Spam).

2. **Python Script**:
   ```bash
   python spam_classifier_model.py
   ```
   - Produces similar outputs to the notebook, including metrics, saved files, and sample prediction.

### Running the Web Application
1. Ensure the environment is active:
   ```bash
   conda activate spam_detector
   # Or for pip: source spam_detector/bin/activate (Linux/macOS) or spam_detector\Scripts\activate (Windows)
   ```
2. Run the Flask app:
   ```bash
   python app.py
   ```
3. Open `http://127.0.0.1:5000` in a web browser.
4. Enter an email message (e.g., "Win a free iPhone now! Click here to claim your prize!") in the text area and click "Check Spam".
5. View the prediction result ("Spam" or "Ham") displayed on the page.

### Expected Outputs
- **Notebook/Script**:
  - Dataset: ~5,728 rows, columns (`text`, `spam`)
  - Class distribution: ~4,360 ham (0), ~1,368 spam (1)
  - Model performance: ~98-99% accuracy, precision, recall, F1-score
  - Confusion matrix: Saved as `cm.png`
  - Sample prediction: "Win a free iPhone now!" → Spam
- **Web App**:
  - Displays "Spam" or "Ham" for user-input emails
  - Example: "Hi, let’s meet tomorrow at 10 AM" → Ham

## Project Structure
```
email-spam-classifier/
├── .gitignore                   # Git ignore file
├── static/
│   └── style.css                # CSS for web app styling
├── templates/
│   └── index.html               # HTML template for web app
├── spam_classifier_model.joblib # Trained SVM model
├── tfidf_vectorizer.joblib      # TF-IDF vectorizer
├── app.py                       # Flask web application
├── preprocess.py                # Text preprocessing function
├── spam_classifier_model.ipynb  # Jupyter Notebook for model training
├── spam_classifier_model.py     # Python script for model training
├── spam_classifier_model_old.ipynb  # Archived old notebook
├── spam_classifier_model_old.py     # Archived old script
├── emails.csv                   # Dataset (download from Kaggle)
├── environment.yml              # Conda environment
├── requirements.txt             # Pip dependencies
├── cm.png                       # Confusion matrix plot
└── README.md                    # This documentation
```

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Ideas for contributions:
- Enhance the web app with prediction confidence scores or batch processing.
- Add deep learning models (e.g., LSTM, BERT) for higher accuracy.
- Include visualizations like ROC curves or word clouds.
- Deploy the web app on Heroku or Render.

## Future Improvements
- **Advanced Models**: Experiment with Random Forest, XGBoost, or BERT for 99%+ accuracy.
- **Preprocessing**: Use word embeddings (e.g., GloVe, BERT) for richer text representation.
- **Web App**: Add features like prediction confidence or email header analysis.
- **Deployment**: Host on Heroku or Render for public access.

## Acknowledgments
- **Dataset**: [Spam Email Dataset](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset)
- **Libraries**: pandas, numpy, scikit-learn, NLTK, matplotlib, seaborn, joblib, Flask
- **Author**: Rathish Tharusha (GitHub: [rathishTharusha](https://github.com/rathishTharusha))
- **Purpose**: Built as a learning project to master NLP, machine learning, and web development

## Contact
For questions, open a GitHub issue or contact [Rathish Tharusha](https://github.com/rathishTharusha).

## License
MIT License