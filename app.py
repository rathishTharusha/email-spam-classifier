from flask import Flask, request, render_template
import joblib
import nltk
from preprocess import preprocess_text

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load('spam_classifier_model.joblib')
except FileNotFoundError:
    print("Error: Model file not found. Ensure 'spam_classifier_model.joblib' is in the root directory.")
    exit(1)

# Download NLTK data (run once)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get email text from form
    email_text = request.form['email_text']
    
    # Preprocess the email
    processed_email = preprocess_text(email_text)
    
    # Make prediction using the Pipeline (includes TF-IDF and SVM)
    prediction = model.predict([processed_email])
    
    # Convert prediction to label
    result = 'Spam' if prediction[0] == 1 else 'Ham'
    
    # Return result to webpage
    return render_template('index.html', prediction=result, email_text=email_text)

if __name__ == '__main__':
    app.run(debug=True)