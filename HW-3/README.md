# Heart Disease Prediction Project

## Project Overview
This project is focused on predicting the presence of heart disease in patients using machine learning techniques. Utilizing clinical parameters such as age, sex, cholesterol levels, and blood pressure, the project aims to provide insights into key factors that contribute to heart disease and predict patient outcomes.

## Dataset
- **Source**: UCI Machine Learning Repository
- **Description**: The dataset consists of 303 individuals with 14 attributes, including the target variable.
- **Features**: Key features include age, sex, chest pain type (cp), resting blood pressure (trestbps), cholesterol levels (chol), fasting blood sugar (fbs), and resting electrocardiographic results (restecg).

## Preprocessing
Preprocessing steps included:
- Handling missing values.
- Encoding categorical variables such as sex, cp, and restecg.
- Normalizing numerical features for certain models.

## Models Used
We employed several machine learning models for this project:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

Each model was tuned for optimal performance using techniques like grid search.

## Evaluation Metrics
Model performance was evaluated based on metrics such as accuracy, precision, recall, F1-score, and confusion matrices.

## Results
The models achieved varying levels of accuracy, with the Random Forest model showing the most promising results. Feature importance analysis revealed that factors like age and cholesterol levels play a significant role in heart disease prediction.

## How to Run the Project
To run this project:
1. Install Python and necessary libraries: Pandas, Scikit-learn, Matplotlib, Seaborn.
2. Clone the repository and navigate to the project directory.
3. Run the Jupyter notebooks or Python scripts provided.

## Dependencies
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Contact
For any queries regarding this project, please reach out to [Your Name] at [Your Email].

## Acknowledgements
Special thanks to the UCI Machine Learning Repository for providing the dataset and to my teachers and peers who provided invaluable insights and feedback throughout this project.

