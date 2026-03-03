# Classifying-Emotions-in-Tweets-using-Natural-Language-Processing-NLP-techniques
## Project Overview
This project focuses on building a Machine Learning model to automatically classify emotions expressed in tweets using Natural Language Processing (NLP) techniques. With the rapid growth of social media, understanding emotional sentiment in text has become essential for businesses, policymakers, and researchers. This project demonstrates how NLP and supervised learning can be used to analyze public sentiment and extract meaningful emotional insights from short text data.
The model predicts emotions such as:
- Angry  
- Calm  
- Happy  
- Sad  
## Problem Statement
Social media generates large volumes of unstructured text data daily. Manually identifying emotional tone is inefficient and subjective.
The objective of this project is to:
- Preprocess and clean textual tweet data
- Convert text into numerical features using TF-IDF
- Train a classification model
- Evaluate performance using standard ML metrics
- Deploy the trained model for real-time predictions
## Dataset
The dataset ('simulated_tweet_emotions.csv') consists of labeled tweets with corresponding emotion categories.
Each record contains:
- 'Tweet' – Original tweet text
- 'Emotion' – Target emotion label
The dataset is simulated and structured for learning and experimentation purposes.
## Data Preprocessing
Text preprocessing steps included:
- Lowercasing text
- Removing URLs
- Removing mentions and hashtags
- Removing punctuation
- Removing numbers
- Stripping extra whitespace
Cleaned text was stored in a new column: 'Cleaned_Tweet'.
## Feature Engineering
Text data was converted into numerical features using:
- **TF-IDF Vectorization**
  - Stopword removal (English)
  - Sparse matrix representation
  - Term frequency–inverse document frequency weighting
This transformation allowed the model to learn patterns from textual input.
## Model Development
### Algorithm Used:
- Logistic Regression (Scikit-learn)
### Workflow:
1. Train-test split (80/20)
2. Model training using Logistic Regression
3. Performance evaluation using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix
## Model Performance
The Logistic Regression model achieved:
- **Accuracy:** 100.00%
- **Precision:** 1.00
- **Recall:** 1.00
- **F1-Score:** 1.00
- **Test Samples:** 200

The confusion matrix showed perfect classification across all four emotion categories.

 Note: The dataset used was simulated and relatively clean. Real-world social media datasets typically produce lower performance due to noise, slang, sarcasm, and class imbalance.
## Data Visualization
To better understand the dataset, the following visualizations were created:
- Emotion distribution bar chart
- Confusion matrix heatmap
- Word clouds:
  - Happy tweets
  - Combined tweets across all emotions
These visualizations helped explore class balance and frequently occurring words within each emotional category.
## Model Saving & Reusability
The trained model and vectorizer were saved using 'joblib':
- 'emotion_model.pkl'
- 'vectorizer.pkl'
This enables reuse without retraining and supports deployment in applications.
## Deployment (Streamlit App)
Ideveloped a simple Streamlit web application to allow real-time emotion prediction.
Users can:
1. Enter a tweet
2. Click "Predict Emotion"
3. Receive an instant classification result

## Limitations
- Dataset is simulated and may not reflect real-world linguistic complexity.
- Does not handle sarcasm or contextual nuance.
- Performance may drop significantly on noisy real-world Twitter data.
- Model is based on traditional ML rather than deep learning architectures.

## Future Improvements

- Implement LSTM or GRU networks
- Use transformer-based models (e.g. BERT)
- Handle class imbalance using advanced sampling techniques
- Deploy using FastAPI or Docker for production-level systems
- Integrate with cloud platforms (e.g., Google Cloud Vertex AI)

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- WordCloud
- Joblib
- Streamlit

## Key Skills Demonstrated

- Natural Language Processing (NLP)
- Text preprocessing
- Feature engineering (TF-IDF)
- Supervised Machine Learning
- Model evaluation & interpretation
- Model serialization
- Basic ML deployment
- Data visualization

## Author

**Fiona Njeri Kamau**  
BSc Statistics  
Data Science & Artificial Intelligence  
