# 📧 Spam or Ham Email Classifier

## Overview
This project is a machine learning-based email classifier built using Streamlit and Scikit-learn. It determines whether an email is **Spam** or **Ham** based on its text content.

## Dataset
The model is trained on a labeled dataset containing email messages categorized as **ham (not spam)** or **spam**. The dataset includes:
- `Category`: Label indicating whether the message is spam or ham.
- `Message`: The actual email content.

## Problem Type
This is a **binary classification problem**, where the objective is to classify an email into one of two categories:
- **Spam (1):** Unwanted or fraudulent emails.
- **Ham (0):** Genuine emails.

## Project Workflow
1. **Data Preprocessing:** 
   - Load the dataset.
   - Convert text data into numerical format using `CountVectorizer`.
   - Encode labels (`ham` → 0, `spam` → 1).
2. **Train-Test Split:** 
   - Divide data into training and testing sets.
3. **Model Training & Selection:** 
   - Train multiple classifiers (`Logistic Regression, Naïve Bayes, KNN, Decision Tree, SVM`).
   - Allow users to select a model in the Streamlit app.
4. **Evaluation & Prediction:** 
   - Compute accuracy score.
   - Predict whether user-inputted email text is spam or ham.

## Installation & Usage
### 🔧 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/spam-ham-classifier.git
   cd spam-ham-classifier
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Future Improvements
- Implement **TF-IDF Vectorization** for better text feature extraction.
- Add **Deep Learning models** (LSTMs, BERT) for improved accuracy.
- Enhance UI with **better visualization and explanations**.
- Deploy the model using **Docker & cloud platforms**.

🚀 Happy Coding!

