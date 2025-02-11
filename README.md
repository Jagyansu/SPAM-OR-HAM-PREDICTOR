# 📧Email Spam or Ham Classifier

## Overview
This project is a **Spam or Ham Email Classifier** built using **Streamlit** for an interactive web-based interface and **Scikit-learn** for machine learning. The application allows users to upload a dataset, train different classification models, and classify emails as spam or ham.

## Features
- Upload and process a dataset (CSV format) containing email messages.
- Train and test models using:
  - Logistic Regression
  - Naïve Bayes
  - K-Nearest Neighbors
  - Decision Tree
  - Support Vector Machine
- Compute and display the accuracy of the selected model.
- Enter an email manually to predict whether it is **Spam** or **Ham**.

## Requirements
Make sure you have the following installed:
- Python 3.x
- Required libraries:
  ```bash
  pip install streamlit pandas numpy scikit-learn
  ```

## Installation & Usage
1. Clone this repository or download the source code.
2. Navigate to the project directory and ensure dependencies are installed.
3. Run the Streamlit app using:
   ```bash
   streamlit run app.py
   ```
4. Upload the `spam.csv` dataset and select a model to train.
5. Enter an email to classify it as Spam or Ham.

## Dataset
The application expects a dataset in CSV format with the following columns:
- **Category**: Label indicating spam (1) or ham (0)
- **Message**: Email text content

Ensure the dataset follows this format before uploading.

## File Structure
```
├── app.py                # Main Streamlit application
├── spam.csv              # Sample dataset
├── SPAM OR HAM PREDICTION.ipynb  # Jupyter Notebook with model analysis
├── README.md             # Project documentation
```

## Future Enhancements
- Implement deep learning models for better accuracy.
- Enable dataset uploading via the Streamlit UI.
- Improve preprocessing techniques for better feature extraction.

## Author
Developed by Jagyansu Padhy - Machine Learning Enthusiast 🚀



