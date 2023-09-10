
## Introduction
Parkinson's Disease is the second most common neurodegenerative disorder worldwide, affecting over 10 million people. It leads to the gradual decline of motor and cognitive functions.

Diagnosing Parkinson's disease is challenging because there isn't a single definitive test available. Instead, doctors rely on carefully analyzing the patient's medical history. Unfortunately, this method isn't very accurate, with only a 53% accuracy rate for early-stage diagnosis according to a study by the National Institute of Neurological Disorders. However, early detection is crucial for effective treatment.

### Goals and Significance
Enhance Early Detection: Improve the accuracy of early-stage Parkinson's disease diagnosis, addressing the current limitation of a 53% accuracy rate in early detection.

Minimize False Negatives: A major focus of this project is to significantly reduce false negatives in the diagnosis process. Identifying individuals who have Parkinson's but might be missed by traditional methods is crucial for early intervention and treatment.

Utilize Speech Features: Leverage a dataset containing various speech features, which have proven to be highly indicative of Parkinson's disease. This includes analyzing tremors, phonation difficulties, and other speech-related changes.

### Why speech features?
Speech is very predictive and characteristic of Parkinson’s disease; almost every Parkinson’s patient experiences severe vocal degradation (inability to produce sustained phonations, tremor, hoarseness), so it makes sense to use voice to diagnose the disease. Voice analysis gives the added benefit of being non-invasive, inexpensive, and very easy to extract clinically.

## Data Collection and Preprocessing
### Data collection
The audio data was obtained from UCI (https://archive.ics.uci.edu/dataset/174/parkinsons). This dataset is composed of a range of biomedical voice measurements from 31 people, 23 with Parkinson's disease (PD). Each column in the table is a particular voice measure, and each row corresponds to one of 195 voice recordings from these individuals ("name" column). The main aim of the data is to discriminate healthy people from those with PD, according to the "status" column, which is set to 0 for healthy and 1 for PD.

### Preprocessing
The dataset contains 24 attributes.
22 are float data types.
"Status," i.e., the target variable, is the only integer variable here.
"Name" is the only object variable.
The data we deal with is mostly numerical.
The dataset has no missing data values.
Exploratory Data Analysis (EDA)
Analysis of the target variable:

Out of the 195 data rows, 147 have PD, and 48 people are healthy. The distribution is not uniform. This is a 75-25 dataset with 75% of people having PD and 25% being healthy.

### Correlation matrix:

Correlation analysis was performed to identify highly correlated features in the dataset. Features with a correlation coefficient greater than 0.99 were identified and considered for potential removal to reduce multicollinearity.

### Profile Report using pandas_profiling:

A profile report was generated using the pandas_profiling library, providing an overview of the dataset's key statistics, data types, missing values, and more.

### Frequency Distributions Analysis:

Frequency distributions were analyzed using bar plots and histograms, focusing on the 'status' variable. Features related to vocal fundamental frequency were examined:

MDVP:Fo(Hz) - Average vocal fundamental frequency
MDVP:Fhi(Hz) - Maximum vocal fundamental frequency
MDVP:Flo(Hz) - Minimum vocal fundamental frequency
Skewness and Kurtosis Analysis:

The dataset's skewness and kurtosis were computed:

Skewness: -1.1785714285714286
Kurtosis: -0.6109693877551026
Handling Data Imbalance:

Instead of oversampling or undersampling, techniques like Synthetic Minority Over-sampling Technique (SMOTE) or generating synthetic data were considered to address class imbalance while preserving data integrity. Model evaluation metrics like precision, recall, and F1-score, which are more informative in imbalanced datasets, were focused on.

## Feature Extraction
### Outlier Removal:

Outliers can significantly impact statistical measures like mean, standard deviation, and regression coefficients, potentially leading to incorrect conclusions. Removing outliers of the data can make it more representative and generalizable, increasing the likelihood that findings can be applied safely and effectively to a broader patient population. Domain knowledge plays a crucial role in this process.

### Mutual Information Classification:

It was employed for feature selection, helping to identify the most informative features that contribute the most to the target variable's prediction. It is capable of capturing both linear and non-linear dependencies between features.

## Model Selection
Many features are highly correlated with each other, which can impact the model's performance due to multicollinearity. Algorithms like decision trees and boosted trees algorithms are immune to multicollinearity by nature.

Algorithms for a binary classification task were chosen:

Logistic Regression: Suitable for Parkinson's disease classification, modeling the probability of having the disease based on input features, making it effective for binary classification tasks.

Bernoulli Naive Bayes: Well-suited for Parkinson's disease classification, particularly when dealing with binary features, as it uses Bayes' theorem to estimate the probability of having the disease based on the presence or absence of certain features.

Decision Tree: Offers interpretability and can be employed for Parkinson's disease classification by sequentially making decisions based on feature values to predict the disease status.

Random Forest: Effective for Parkinson's disease classification due to its ability to handle high-dimensional datasets, reduce overfitting, and capture complex relationships between features and disease status.

XGBoost: Chosen due to its exceptional performance on the dataset. It optimizes model performance through boosting and provides accurate predictions.

Gradient Boosting: Similar to XGBoost, a powerful algorithm for Parkinson's disease classification, combining multiple weak learners to improve prediction accuracy and handle complex relationships within the data.

### Why XG Boost?

XGBoost (XGB) was chosen for the Parkinson's disease classification problem due to its exceptional performance on the dataset. Initially, when tested on all algorithms, XGB exhibited remarkable results, achieving a Matthews coefficient score of 0.914853 and a Cohen Kappa of 0.911243. Notably, it also demonstrated an impressive precision of 1.00, indicating a very low rate of false positives.

The results of performing variance threshold on various models were consistent, maintaining the Matthews correlation coefficient at an impressive value of 0.914853.

After testing different ranges of feature selection through a variance threshold, algorithms like decision tree and AdaBoost could reach the value of Matthew's correlation coefficient of 0.914853.

Feature selection may not work for XGBoost:

Loss of Discriminative Power: It may discard features with low variance that could be important for distinguishing between disease classes.

Information Redundancy: Some low-variance features might be correlated with other crucial features, and their removal can lead to a loss of valuable information.

Overfitting Risk: Removing low-variance features might lead to overfitting, particularly when the dataset is small, as the model may become too specific to the training data.

Data Characteristics Matter: The effectiveness of variance thresholding depends on the dataset's characteristics. If the data is noisy or features have weak correlations with the target variable, this method may not yield significant benefits.

## Model Evaluation
The XGBoost (XGB) model initially performed exceptionally well with a Matthews correlation coefficient and Cohen's kappa score both at 0.914853. However, after fine-tuning through a randomized search, the model's performance decreased slightly to a Matthews correlation coefficient and Cohen's kappa score of 0.8136645962732919.

The drop in the Matthews correlation coefficient (MCC) and Cohen's kappa score after fine-tuning through a randomized search can occur for several reasons:

Overfitting: Fine-tuning hyperparameters can lead to overfitting if the model becomes too specialized for the training data. This overfit model may perform exceptionally well on the training data but generalize poorly to unseen data, resulting in a decrease in performance metrics like MCC and Cohen's kappa.

Hyperparameter Configuration: The randomized search explores various hyperparameter configurations, and not all of them may lead to improved performance. It's possible that some combinations of hyperparameters resulted in a less effective model.

Randomness: Randomized search introduces an element of randomness, and sometimes it may not converge to the absolute best hyperparameters, leading to a drop in performance.

## Deployment and Practical Application

### Real-World Use Cases and Benefits:

Early Diagnosis: The deployed model can aid in the early detection of Parkinson's disease, allowing healthcare professionals to intervene sooner and potentially slow the progression of the disease.

Data Insights: By analyzing data from a large patient population, healthcare professionals can gain insights into disease trends, treatment effectiveness, and potential risk factors.

## Future Directions

Simplified Diagnostic Process:

In the future, the Parkinson's disease detection model could revolutionize diagnosis by streamlining the process to just a 10-15 second voice recording. This simplified approach would provide swift and accurate diagnostic results, offering patients and healthcare professionals a convenient and efficient means of early disease detection, potentially leading to quicker intervention and improved patient outcomes.


### Libraries used: Numpy, Seaborn, Matplotlib, Shap, scikit-learn, Pandas, Scipy, and many more.

### Frameworks: Flask
