# Heart Disease Prediction Project

## Project Overview
This project is focused on predicting the presence of heart disease in patients using machine learning techniques. Utilizing clinical parameters such as age, sex, cholesterol levels, and blood pressure, the project aims to provide insights into key factors that contribute to heart disease and predict patient outcomes.

## Dataset
- **Source**: UCI Machine Learning Repository
- **Description**: The dataset consists of 303 individuals with 14 attributes, including the target variable.
- **Features**: Key features include age, sex, chest pain type (cp), resting blood pressure (trestbps), cholesterol levels (chol), fasting blood sugar (fbs), and resting electrocardiographic results (restecg).

# Data Description:
- **id**: (Unique id for each patient)
- **age**: (Age of the patient in years)
- **origin**: (place of study)
- **sex**: (Male/Female)
- **cp chest pain type**: ([typical angina, atypical angina, non-anginal, asymptomatic])
- **trestbps resting blood pressure**: (resting blood pressure (in mm Hg on admission to the hospital))
- **chol**: (serum cholesterol in mg/dl)
- **fbs**: (if fasting blood sugar > 120 mg/dl)
- **restecg**: (resting electrocardiographic results)
- **Values**: [normal, stt abnormality, lv hypertrophy]
- **thalach**: maximum heart rate achieved
- **exang**: exercise-induced angina (True/ False)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: the slope of the peak exercise ST segment
- **ca**: number of major vessels (0-3) colored by fluoroscopy
- **thal**: [normal; fixed defect; reversible defect]
- **num**: the predicted attribute

## Preprocessing

The preprocessing stage is crucial in preparing the dataset for effective machine learning model training. In this project, several preprocessing steps were undertaken:
  
Preprocessing steps included:
- Handling missing values.
- Encoding categorical variables such as sex, cp, and restecg.
- Normalizing numerical features for certain models.
- 
- **Removing Columns with High Missing Values**: Columns 'slope', 'ca', and 'thal' were dropped due to a high proportion of missing values. Keeping these columns with substantial missing data could have introduced bias or inaccuracies in the models.

- **Imputing Missing Values**: For columns with fewer missing values ('trestbps', 'chol', 'thalch', 'oldpeak'), missing data was imputed using the median value of each column. Median imputation was chosen as it is less sensitive to outliers than mean imputation, thus preserving the integrity of the dataset more effectively.

- **Encoding Categorical Variables**: 
  - The 'sex', 'fbs', and 'exang' columns were binary categorical variables and were encoded as 1 and 0. This conversion was necessary to transform these categorical values into a format that can be efficiently processed by machine learning algorithms.
  - The 'cp', 'restecg', and 'dataset' columns were one-hot encoded, creating a separate binary column for each category. One-hot encoding is a standard approach for handling nominal categorical data, as it allows models to better interpret and use the information without an implied order.

- **Dropping the 'id' Column**: The 'id' column was removed as it is a unique identifier for each patient and does not contribute to the predictive power of the models.

- **Splitting the Dataset**: The dataset was split into training and testing sets, with 80% of the data used for training and 20% for testing. This split ensures that the model can be trained on a large subset of the data while also being validated on an unseen subset, providing insights into its generalization capability.

- **Re-Imputing NaN Values in Training and Test Sets**: Any NaN values that emerged during the train-test split were imputed using the same median strategy, ensuring consistency and data integrity.

Each of these preprocessing steps was critical to ensure the data was clean, relevant, and structured in a way that maximizes the performance and accuracy of the machine learning models.

## Data Visualization

In this project, various data visualization techniques were employed to gain insights into the dataset and the results of our predictive models. These visualizations aid in understanding the underlying distribution of data, identifying patterns, and interpreting model outputs.

 - **Histograms**: Used to display the distribution of data for each numeric feature. Histograms help in understanding the spread (such as range and central tendency) and identifying skewness (if the data is skewed to the left or right). For example, histograms of `age` and `cholesterol levels` provide insights into the age distribution of the dataset and the typical cholesterol levels among the patients.

- **Countplots (Bar Graphs)**: These are useful for categorical data. They show the frequency of each category within a feature, helping us understand the balance or imbalance in categorical variables. For instance, countplots of `sex` and `cp (chest pain type)` reveal the proportion of males vs. females in the dataset and the prevalence of different types of chest pain among patients.

- **Correlation Heatmap**: This visualization shows the correlation coefficients between pairs of numeric variables. It helps in identifying potentially significant relationships between features. For instance, in ours we see that age plays a role in resting blood pressure however doesn't play a role in cholestoral or  max heart rate.

- **Box Plots**: These are used to examine the distribution of numeric variables across different categories. Box plots for `age` across different types of `cp` show us thatc age groups 45-60 are more prone to their specific types of chest pain.

- **Pair Plots or Scatter Plots**: Pair plots provide pairwise scatter plots for numeric variables, offering a comprehensive view of how each variable relates to others. Scatter plots can be particularly insightful for observing relationships between key features like `age` and `max heart rate achieved` with respect to the presence of heart disease to which we saw the severity of the heart disease from ages 40 onwards got significantly worse.

## Models Used
I employed several machine learning models for this project:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

Each model was tuned for optimal performance using techniques like grid search.

## Evaluation Metrics
Model performance was evaluated based on metrics such as accuracy, precision, recall, F1-score, and confusion matrices.

## Results
The models achieved varying levels of accuracy, with the Random Forest model showing the most promising results. Feature importance analysis revealed that factors like age, max heart rate, resting blood pressure, and cholesterol levels play a significant role in heart disease prediction.

There was some overlap in the 95% confidence interval however SVM and KVM models performed signifcantly worse having their high bound the low bound of the Random Forest model and the lowbound of the Logistic Regression and Decision Tree models was a lot lower than the Random Forest's along with their high bounds being lower than the Random Forest so this model outperformed all others.

## How to Run the Project
To run this project:
1. Install Python and necessary libraries: Pandas, Scikit-learn, Matplotlib, Seaborn.
2. Clone the repository and navigate to the project directory.
3. Run the Jupyter notebook.

## Dependencies
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Contact
For any queries regarding this project, please reach out to me at marcos_remy@uri.edu.

## Acknowledgements
Special thanks to the UCI Machine Learning Repository for providing the dataset and to my teacher who provided invaluable insights and feedback on this project.
