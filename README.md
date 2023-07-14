# parkinsons_disease

## Introduction
Parkinson's Disease is the second most common neurodegenerative disorder worldwide, affecting over 10 million people. It leads to the gradual decline of motor and cognitive functions.

Diagnosing Parkinson's disease is challenging because there isn't a single definitive test available. Instead, doctors rely on carefully analyzing the patient's medical history. Unfortunately, this method isn't very accurate, with only a 53% accuracy rate for early-stage diagnosis according to a study by the National Institute of Neurological Disorders. However, early detection is crucial for effective treatment.

To address these diagnostic limitations, this investigation explores the use of machine learning to accurately diagnose Parkinson's disease. The study uses a dataset containing various speech features because speech has proven to be highly indicative of the condition. Many Parkinson's patients experience noticeable changes in their speech, such as tremors and difficulty producing sustained phonations. Using voice analysis has the advantages of being non-invasive, cost-effective, and easily accessible in a clinical setting.


## Performance Metrics

TP = true positive, FP = false positive, TN = true negative, FN = false negative
Copen Kappa score
AUC ROC score
Accuracy, Precision, Recall

An overall report was generated for error analysis.

The dataset was imbalanced and these metrics were chosen accordingly.The dataset 1 and 0 status is shown in the countplot below.


## Algorithms Employed
### Logistic Regression:
Logistic Regression is a suitable model for Parkinson's disease classification, as it models the probability of having the disease based on input features, making it effective for binary classification tasks.

### Bernoulli Naive Bayes: 
Bernoulli Naive Bayes is well-suited for Parkinson's disease classification, particularly when dealing with binary features, as it uses Bayes' theorem to estimate the probability of having the disease based on the presence or absence of certain features.

### Decision Tree: 
Decision Trees offer interpretability and can be employed for Parkinson's disease classification by sequentially making decisions based on feature values to predict the disease status.

### Random Forest: 
Random Forest, an ensemble model, is well-suited for Parkinson's disease classification due to its ability to handle high-dimensional datasets, reduce overfitting, and capture complex relationships between features and disease status.

### XGBoost: 
XGBoost, an advanced gradient boosting algorithm, is highly effective in handling tabular data, making it a strong candidate for Parkinson's disease classification. It optimizes model performance through boosting and provides accurate predictions.

### Gradient Boosting: 
Gradient Boosting, similar to XGBoost, is a powerful algorithm for Parkinson's disease classification. It combines multiple weak learners in an iterative manner to improve prediction accuracy and handle complex relationships within the data

## Engineering Goal

The evaluation of the model emphasized the importance of feature engineering and feature selection. The primary objective was to minimize false negatives, thereby prioritizing a high recall rate. Additionally, the model's accuracy was expected to be substantial, and its overall performance was crucial. Furthermore, achieving a decent Cohen Kappa score of 0.8 and above was also a key consideration during the evaluation process.
Dataset Description

Source: the University of Oxford
195 instances (147 subjects with Parkinson’s, 48 without Parkinson’s)
22 features (elements that are possibly characteristic of Parkinson’s, such as frequency, pitch, amplitude / period of the sound wave)
1 label (1 for Parkinson’s, 0 for no Parkinson’s)
Project Pipeline
pipeline

## Brief Summary of Procedure

The initial steps involved in the analysis process included data comprehension, correlation examination, and generating a profile report on the features. Visualizations and analysis were conducted, including univariate analysis and review of statistical values for highly correlated features. Skewness and Kurtosis were also explored in depth.

Feature selection played a crucial role in understanding the variables, and the mutual information of the top 10 features was visualized in a bar plot.
Following the data preprocessing and wrangling phase, I initially considered performing PCA on the classification data. However, I ultimately decided against it.
Subsequently, the following steps were executed on the prepared data:

Train-test split was performed to divide the data for training and testing purposes.
The data was then scaled using a Standard Scaler to ensure consistency across features.
Various models were fitted using the scaled data.
An error report was generated, and cross-validation scores were obtained for the models.
The performance of the models was compared to assess their effectiveness.
Threshold variance-based feature selection was implemented on the models.
These steps facilitated the evaluation and refinement of the models, enhancing their predictive capabilities.

## Model Performance
The XGBoost and Random Forest models demonstrated strong performance, achieving Cohen Kappa scores of 0.80 and 0.86, respectively. Notably, both the Adaboost and Random Forest models achieved a recall value of 1, indicating no false negatives in their predictions.

When comparing the models, the Random Forest model exhibited the highest overall accuracy. However, when considering threshold variance, the XGBoost model performed exceptionally well.

By applying a threshold value of 0.0005 to the XGBoost model, an impressive accuracy of 97.4359% was achieved. This represents a significant improvement compared to the original accuracy of the model without feature selection, which stood at 94.8718%.

Overall, this corresponds to a remarkable 2.5641% increase in the model's accuracy, highlighting the effectiveness of feature selection in enhancing the performance of the XGBoost model.

## Conclusion and Significance

This report showcases the implementation of a machine learning approach that significantly improves the diagnosis of Parkinson's disease. The study highlights the remarkable accuracy achieved and emphasizes the criticality of early and precise detection for effective treatment.

### Improved Accuracy and Timely Diagnosis:
Compared to current methods, the machine learning model achieves an impressive 97% accuracy, marking a substantial 44% increase. This underscores the importance of early and accurate diagnosis in managing Parkinson's disease.

### Impact on Disease Management:
The machine learning model enables earlier detection, which has the potential to slow down or treat disease progression, leading to improved disease management and enhanced quality of life for patients.

### Significance and Accessibility:
With over 10 million individuals affected globally, the machine learning approach holds significant relevance. It not only provides accurate and early diagnosis but also offers scalability, cost-effectiveness, and accessibility to underserved populations.

### Simplified Diagnostic Process:
The machine learning model simplifies the diagnostic process, requiring only a 10-15 second voice recording and producing an immediate diagnosis.

#### Implementing machine learning techniques enhances Parkinson's disease diagnosis by delivering high accuracy. Early and precise detection is vital for effective treatment, and the accessibility and simplicity of the machine learning model make it a promising solution for improved patient outcomes

## Future Developments
To further enhance the effectiveness of the model, future efforts can focus on expanding the dataset and refining the machine learning algorithm. This can be achieved by collecting additional data, specifically voice recordings, through the development of a dedicated app. By deploying the model to analyze these voice recordings, the collected data can be processed, and the model can determine whether the user exhibits signs of Parkinson's disease or not. This approach would provide a convenient and efficient method for individuals to assess their risk of Parkinson's disease using their own voice recordings.









