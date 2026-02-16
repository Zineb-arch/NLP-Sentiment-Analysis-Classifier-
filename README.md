# NLP-Sentiment-Analysis-Classifier-

#  Sentiment Analysis using MLP with Lexicon-Based Features

##  Project Overview

This project implements a **sentiment classification system** using a **Multi-Layer Perceptron (MLP)** neural network to predict whether a movie review expresses **positive** or **negative** sentiment.

Instead of traditional bag-of-words or word embeddings, the model relies on **lexicon-based sentiment features**, specifically:
- **VADER polarity score**
- **TextBlob polarity score**

This lightweight approach demonstrates how sentiment lexicons can be effectively used with neural networks for text classification.

---

## 📂 Dataset

📁 **Dataset**: The IMDB movie reviews dataset used in this project is publicly available on Kaggle:  
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

- **Total samples**: 50,000 movie reviews  
- **Classes**:
  - `positive`
  - `negative`

The dataset is split into **80% training** and **20% testing**.

---

##  Feature Engineering

Each movie review is transformed into numerical features using sentiment lexicons:

- **VADER compound score**: captures sentiment intensity and polarity
- **TextBlob polarity score**: captures overall sentiment orientation

These two values form the **input feature vector** for the neural network.

---

##  Model Architecture

- **Model**: Multi-Layer Perceptron (MLP)
- **Input Layer**: 2 neurons (VADER + TextBlob scores)
- **Hidden Layers**:
  - 16 neurons with ReLU activation
  - 8 neurons with ReLU activation
- **Output Layer**: Binary classification (positive / negative)
- **Optimizer**: Adam
- **Loss Function**: Cross-entropy
- **Feature Scaling**: StandardScaler
- **Training Epochs**: up to 500

---

## 📊 Evaluation and Results

The model was evaluated on **10,000 test reviews** using standard classification metrics.

###  Quantitative Metrics

| Metric      | Score |
|------------|-------|
| Accuracy   | **0.7747** |
| Precision | **0.7876** |
| Recall    | **0.7571** |
| F1-score  | **0.7720** |

---

### 📋 Classification Report


| Class | Precision | Recall | F1-score | Support |
|------|----------|--------|----------|---------|
| Negative (0) | 0.76 | 0.79 | 0.78 | 4,961 |
| Positive (1) | 0.79 | 0.76 | 0.77 | 5,039 |
| **Accuracy** |  |  | **0.77** | **10,000** |
| **Macro Avg** | 0.78 | 0.77 | 0.77 | 10,000 |
| **Weighted Avg** | 0.78 | 0.77 | 0.77 | 10,000 |


---

###  Confusion Matrix

The confusion matrix summarizes the model’s predictions:
<img width="522" height="470" alt="image" src="https://github.com/user-attachments/assets/95559df8-72cd-42ea-88e6-25ce2584f646" />


- **True Negatives**: 3932  
- **False Positives**: 1029  
- **False Negatives**: 1224  
- **True Positives**: 3815  

![Confusion Matrix](./confusion_matrix.png)

The model correctly classifies the majority of reviews, with most errors occurring for reviews with neutral or ambiguous sentiment.

---

##  Interpretation

Despite using only two lexicon-based sentiment features, the MLP classifier achieves an accuracy of approximately **77%**. This highlights the effectiveness of sentiment lexicons when combined with neural networks, while keeping the model simple, interpretable, and computationally efficient.

---

##  Limitations

- Lexicon-based features do not capture contextual or semantic meaning
- Sarcasm and complex sentence structures are not well handled
- Performance may plateau without richer text representations

---

##  Future Improvements

- Include VADER positive, negative, and neutral scores
- Compare MLP performance with Logistic Regression or SVM
- Incorporate word embeddings or transformer-based models (e.g., BERT)
- Perform hyperparameter tuning

---

##  Technologies Used

- Python
- Scikit-learn
- NLTK (VADER)
- TextBlob
- Pandas & NumPy
- Matplotlib & Seaborn

---

