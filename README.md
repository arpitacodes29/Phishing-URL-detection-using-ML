# Phishing URL Detection using Machine Learning

This project implements a machine learning–based system to detect phishing URLs using engineered URL features. A Random Forest classifier is trained to classify URLs as phishing or legitimate.

##  Project Overview

Phishing attacks are a major cybersecurity threat where attackers trick users into revealing sensitive information using malicious URLs.  
This project uses machine learning to automatically detect phishing URLs based on lexical and domain-based features.

##  Dataset

- Source: Kaggle
- Type: Pre-engineered phishing URL dataset
- Target column: `phishing`
  - `1` → Phishing
  - `0` → Legitimate

 ## Sample Features
- `length_url`
- `email_in_url`
- `qty_redirects`
- `url_google_index`
- `domain_google_index`
- `url_shortened`

 ## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

##  Project Workflow

1. Load the phishing dataset
2. Separate features and target labels
3. Split data into training and testing sets
4. Train a Random Forest classifier
5. Evaluate model performance using:
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix

---

##  How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/arpitacodes29/Phishing-URL-detection-using-ML.git
cd Phishing-URL-detection-using-ML
