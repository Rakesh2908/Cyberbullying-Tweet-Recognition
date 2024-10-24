# Cyberbullying Tweet Recognition App

## Project Overview

This project aims to analyze tweets to predict whether a tweet contains cyberbullying content or not. If a tweet is classified as cyberbullying, it further categorizes the nature of the cyberbullying into one of six categories:

- **Age**
- **Ethnicity**
- **Gender**
- **Religion**
- **Other Cyberbullying**

### Dataset

We utilized the [Cyberbullying Classification Dataset](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification) from Kaggle to train and test our models.

## Approach

- **Library Installation and Data Import**
  - Install the necessary libraries using pip and a `requirements.txt` file:
  
    ```bash
    pip install -r requirements.txt
    ```

    **Caution:** Ensure you have pip installed before running this command.
  
  - Import the dataset using `pandas` for further analysis.

- **Data Review and Missing Values**
  - Conduct an initial exploration of the dataset to understand its structure and content.
  - Check for missing values and handle them appropriately (e.g., imputation, removal).

- **Data Preprocessing**
  - Clean the tweet data by performing the following steps:
    - Remove emojis.
    - Convert text to lowercase and remove:
      - Line breaks (`\r`, `\n`)
      - URLs
      - Non-UTF characters
      - Numbers and punctuations
      - Stopwords (common words like "and", "the", "is")
    - Remove contractions (e.g., "can't" -> "cannot").
    - Clean hashtags and filter out special characters.
    - Eliminate multi-space characters.
    - Apply stemming or lemmatization to standardize words to their base form.

- **Handling Duplicates**
  - Identify and remove duplicate entries from the dataset to avoid skewing model training.

- **Exploratory Data Analysis (EDA)**
  - Conduct EDA to gain insights into the distribution and patterns within the data. This can help visualize the data and identify potential relationships between features.

- **Train-Test Split**
  - Split the dataset into training and testing sets for model evaluation. The training set is used to train the model, and the testing set is used to evaluate its performance on unseen data.

- **TF-IDF Vectorization**
  - Transform the text data using Term Frequency-Inverse Document Frequency (TF-IDF) to convert text into numerical features suitable for machine learning algorithms.

- **Model Training**
  - Experiment with various machine learning models to determine the best performing classifier for this task. Examples of models to consider include:
    - Logistic Regression
    - Support Vector Classifier (SVC)
    - Naive Bayes Classifier
    - Decision Tree Classifier
    - Random Forest Classifier
    - AdaBoost Classifier

- **Model Tuning**
  - Fine-tune the hyperparameters of the chosen model to optimize its accuracy and performance.

- **Model Evaluation and Saving**
  - Evaluate the model's performance using appropriate metrics such as accuracy, precision, recall, and F1-score.
  - Save the final model using a library like `pickle` or joblib for future use.

## Libraries Used

The project utilizes the following Python libraries:

- `pandas` (data manipulation)
- `numpy` (numerical computations)
- `matplotlib` and/or `seaborn` (data visualization)
- `scipy` (scientific computing)
- `re` (regular expressions)
- `pickle` or `joblib` (model serialization)
- `string` (text processing)
- `nltk` (Natural Language Toolkit) - for advanced text processing (optional)
- `emoji` (handling emojis) - for advanced text processing (optional)
- `wordcloud` (generating word clouds) - for visualization (optional)
- `streamlit` (web app development) - to deploy the model as a web app (optional)
- `collections` (data structures) - for advanced data manipulation (optional)
- `statsmodels` (statistical modeling) - for advanced analysis (optional)
- `flask` (web framework) - alternative to Streamlit for web app development (optional)

The specific libraries you need might vary depending on your implementation choices.

## How to Run

**Prerequisites:** Ensure you have Python 3 and pip (package manager) installed.

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-repo-link.git
   ```

2. **Navigate to the Project Directory**

   ```bash
   cd your-repo-name
   ```

3. **Install Required Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**

   ```bash
   streamlit run app.py
   ```

## Future Enhancements

- Implement deep learning models for improved accuracy.
- Deploy the model using cloud platforms like AWS or Heroku.
- Add a real-time tweet analysis feature.
