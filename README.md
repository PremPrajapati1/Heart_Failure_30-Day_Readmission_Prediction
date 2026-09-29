# Heart Failure 30-Day Readmission Prediction

A machine learning project that predicts whether a heart failure patient is likely to be readmitted within 30 days. The project includes data preprocessing, categorical encoding, feature scaling, model training, hyperparameter experimentation, and performance evaluation using multiple classification models.

# 📌 Overview

Hospital readmission is an important healthcare problem. This project uses machine learning classification techniques to analyze patient information and predict the possibility of 30-day readmission.

The project compares three machine learning algorithms:

Logistic Regression
K-Nearest Neighbors (KNN)
Decision Tree

The models are evaluated using:

Accuracy
Precision
Recall
F1-Score
ROC-AUC
Confusion Matrix
# 🎯 Objectives
Analyze the heart failure dataset.
Perform data preprocessing and cleaning.
Remove unnecessary patient identification information.
Encode categorical features.
Scale numerical features.
Split the dataset into training and testing sets.
Train multiple machine learning classification models.
Perform hyperparameter experiments for KNN and Decision Tree.
Compare model performance using different evaluation metrics.
Select a final model based on the experimental results.
# 📊 Dataset

The dataset contains 12,000 patient records with 36 columns.

Target Variable
Readmitted_30_Days

The target represents whether the patient was readmitted within 30 days.

Identifier Removed
Patient_ID

Patient_ID was removed because it is an identifier and does not provide useful predictive information for the model.

# 🔄 Project Workflow
Dataset
   ↓
Data Exploration
   ↓
Data Preprocessing
   ↓
Categorical Encoding
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Hyperparameter Experimentation
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Final Model
# 🧹 Data Preprocessing
1. Target Separation

The target variable was separated from the input features:

y = df["Readmitted_30_Days"]
2. Remove Patient ID
df = df.drop("Patient_ID", axis=1)
3. Categorical Encoding

LabelEncoder was used for:

Gender
Smoking_Status
Alcohol_Consumption

pd.get_dummies() was used for:

Heart_Failure_Type
4. Train-Test Split

The dataset was divided into:

80% training data
20% testing data
random_state=42

For 12,000 records:

Training samples: 9,600
Testing samples: 2,400
5. Feature Scaling

StandardScaler was used to scale the features.

The scaler was fitted on the training data and then used to transform the test data.

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

This prevents information from the test set being used during the scaling process.

# 🤖 Machine Learning Models
1. Logistic Regression

The final Logistic Regression configuration was:

LogisticRegression(C=1, max_iter=1000)

Logistic Regression was used as a binary classification model for predicting 30-day readmission.

2. K-Nearest Neighbors

Different values of k were experimented with:

3, 5, 7, 9, 11, 15, 21

5-fold cross-validation was used during the experiment.

The final model used:

KNeighborsClassifier(n_neighbors=15)
3. Decision Tree

Different max_depth values were experimented with:

2, 3, 4, 5, 6, 8, 10, 15

5-fold cross-validation was used during the experiment.

The final model used:

DecisionTreeClassifier(max_depth=5)

# 📈 Model Performance
Model	Train Accuracy	Test Accuracy	Precision	Recall	F1-Score	ROC-AUC
Logistic Regression	89.44%	89.50%	85.24%	78.61%	81.79%	95.62%
KNN	88.57%	83.96%	82.91%	58.61%	68.67%	87.85%
Decision Tree	85.14%	83.83%	75.78%	67.78%	71.55%	89.07%
# 📊 Evaluation Metrics
Accuracy

Measures the percentage of total predictions that were classified correctly.

Precision

Measures how many of the patients predicted as positive were actually positive.

Recall

Measures how many of the actual positive cases were correctly identified.

F1-Score

Provides a combined measure of Precision and Recall.

ROC-AUC

Measures the model's ability to distinguish between the two classes across different classification thresholds.

Confusion Matrix

The confusion matrix provides:

True Positives (TP)
True Negatives (TN)
False Positives (FP)
False Negatives (FN)

# 🔍 Results
Logistic Regression
Test Accuracy: 89.50%
Precision: 85.24%
Recall: 78.61%
F1-Score: 81.79%
ROC-AUC: 95.62%

Training accuracy was 89.44%, while testing accuracy was 89.50%, showing a very small difference between the two.

KNN
Test Accuracy: 83.96%
Precision: 82.91%
Recall: 58.61%
F1-Score: 68.67%
ROC-AUC: 87.85%

The training accuracy was 88.57%, compared with a test accuracy of 83.96%.

Decision Tree
Test Accuracy: 83.83%
Precision: 75.78%
Recall: 67.78%
F1-Score: 71.55%
ROC-AUC: 89.07%

The training accuracy was 85.14%, compared with a test accuracy of 83.83%.

🏆 Final Model

Based on the reported experimental results, Logistic Regression was selected as the final model.

Final Performance
Test Accuracy : 89.50%
Precision     : 85.24%
Recall        : 78.61%
F1-Score      : 81.79%
ROC-AUC       : 95.62%

# 🛠️ Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Jupyter Notebook

# Machine Learning Techniques
Logistic Regression
K-Nearest Neighbors
Decision Tree
Label Encoding
One-Hot Encoding using pd.get_dummies()
Standard Scaling
5-Fold Cross-Validation
Classification Evaluation

Adjust the filenames and folders according to the actual files in your repository.

# 🚀 How to Run
1. Clone the Repository
git clone https://github.com/your-username/Heart_Failure_30-Day_Readmission_Prediction.git
2. Open the Project
cd Heart_Failure_30-Day_Readmission_Prediction
3. Create Virtual Environment
python -m venv venv
4. Activate Virtual Environment

Windows:

venv\Scripts\activate

Linux/macOS:

source venv/bin/activate
5. Install Dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
6. Start Jupyter Notebook
jupyter notebook

Open the .ipynb file and execute the cells in order.

# 🔮 Future Improvements
- Test additional machine learning algorithms.
- Perform more extensive hyperparameter optimization.
- Explore feature selection techniques.
- Analyze feature importance.
- Investigate class imbalance techniques.
- Test the model on an external dataset.
- Create a web-based prediction interface.
- Build an API for model prediction.
Deploy the trained model.

# ⚠️ Disclaimer
This project is developed for educational and machine learning demonstration purposes. The predictions generated by this model should not be considered medical advice or used as a substitute for professional clinical judgment.

# 👨‍💻 Author
Prem Kumar

BCA Student | Software Development | Machine Learning 
Skills demonstrated: Python • Machine Learning • Data Preprocessing • Classification • Model Evaluation

# ⭐ Project Summary
This project demonstrates an end-to-end machine learning workflow for 30-day heart failure readmission prediction, covering data preprocessing, categorical encoding, feature scaling, model training, hyperparameter experimentation, and model evaluation.

Three classification algorithms—Logistic Regression, KNN, and Decision Tree—were implemented and compared using multiple performance metrics.
