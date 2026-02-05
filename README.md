# social-media-sentiment-analysis-using-ml-models
📌 Overview

This project focuses on performing Sentiment Analysis on Social Media data using multiple Machine Learning (ML) models.
The objective is to classify user-generated text into sentiment categories such as Positive, Negative, and Neutral, analyze platform-wise trends, and compare the performance of different ML algorithms.

The project follows a complete end-to-end ML pipeline, starting from data preprocessing to model evaluation and prediction on new data.

📑 Table of Contents

Overview

Key Features

Dataset Description

Project Architecture

Technologies Used

Data Cleaning and Preprocessing

Exploratory Data Analysis (EDA)

Text Analysis

Feature Extraction

Model Training and Comparison

Model Evaluation

Results

Sentiment Prediction

Folder Structure

Future Improvements

Conclusion

✨ Key Features

End-to-end sentiment analysis pipeline

Robust text preprocessing and cleaning

Exploratory Data Analysis with visualizations

Feature extraction using NLP techniques

Training and comparison of multiple ML models

Evaluation using standard metrics

Sentiment prediction on unseen data

📂 Dataset Description

The dataset consists of social media posts along with engagement and temporal metadata.

Dataset Attributes
Column	Description
Text	User-generated content expressing opinions
Sentiment	Sentiment label (Positive / Negative / Neutral)
Platform	Social media platform of the post
Hashtags	Trending topics and themes
Likes	Number of likes
Retweets	Number of retweets
Country	Geographic origin
Timestamp	Date and time of the post
Year	Year of the post
Month	Month of the post
Day	Day of the post
Hour	Hour of the post
🏗️ Project Architecture

Data Ingestion

Data Cleaning & Preprocessing

Exploratory Data Analysis

Text Analysis

Feature Extraction

Model Training

Model Comparison

Evaluation of Best Model

Sentiment Prediction

🛠️ Technologies Used

Python

Pandas, NumPy – Data manipulation

Matplotlib, Seaborn – Data visualization

Scikit-learn – Machine learning models

WordCloud – Text visualization

NLTK / Regex – Text preprocessing

🧹 Data Cleaning and Preprocessing

The following steps are applied to clean the data:

Handling missing values

Converting text to lowercase

Removing URLs, punctuation, numbers, and special characters

Removing extra whitespaces

Removing duplicate records

📊 Exploratory Data Analysis (EDA)

EDA is performed to:

Understand dataset structure

Analyze sentiment distribution

Visualize platform-wise sentiment trends

Identify engagement patterns across platforms

📝 Text Analysis

Text analysis helps to:

Identify frequent words in each sentiment category

Generate WordClouds for better visualization

Understand dominant themes and opinions

🔢 Feature Extraction

Text data is transformed into numerical features using:

CountVectorizer

TF-IDF Vectorizer

N-gram representation for capturing word context

🤖 Model Training and Comparison

The following machine learning models are trained and evaluated:

Logistic Regression

Linear Support Vector Machine (SVM)

Random Forest Classifier

Multinomial Naive Bayes

Each model is trained on the same dataset to ensure fair comparison.

📈 Model Evaluation

Models are evaluated using:

Accuracy Score

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

🏆 Results

After comparison, Linear SVM achieved the highest accuracy among all tested models, making it the most effective model for this dataset.

🔮 Sentiment Prediction

The final trained model can predict sentiment for new text inputs.

Example:

Input: "This product exceeded my expectations"
Output: Positive

📁 Folder Structure
├── main.ipynb
├── sentimentdataset.csv
├── README.md

🚀 Future Improvements

Use deep learning models such as LSTM or BERT

Apply hyperparameter tuning

Address class imbalance

Deploy model using Flask or Streamlit

✅ Conclusion

This project demonstrates how machine learning techniques can be effectively used for sentiment analysis on social media data.
The comparative study highlights Linear SVM as the most suitable model for this classification task.

👨‍💻 Author

Ayush Kumar
Diploma in Artificial Intelligence & Machine Learning
