
# User Behavior Anomaly Detection Using Machine Learning

## About the Project

This project focuses on detecting unusual user behavior and identifying potential security threats using machine learning algorithms.

The main goal is to analyze user activity data, detect anomalous behavior, and compare the performance of different machine learning models.

This project was developed as my Bachelor's degree project.

## Dataset

The project uses the `cybersecurity_intrusion_data.csv` dataset.

The dataset contains information about user sessions and network activities, including:

- Network packet size
- Session duration
- IP reputation score
- Login attempts and failed logins
- Protocol type
- Encryption method
- Browser type
- Unusual time access

The target column is `attack_detected`, which indicates whether an attack was detected.

## Project Workflow

The main steps of this project are:

1. **Data Cleaning:** Removing duplicate records and handling missing values.
2. **Feature Encoding:** Converting categorical features into numerical values.
3. **Feature Engineering:** Creating new features such as login failure ratio, normalized traffic, weak encryption, and risk score.
4. **Feature Scaling:** Applying MinMaxScaler to scale the features.
5. **Model Training and Evaluation:** Training different machine learning models and comparing their performance.

## Machine Learning Algorithms

The following algorithms were implemented and evaluated:

- Isolation Forest
- One-Class SVM
- Local Outlier Factor (LOF)
- K-Nearest Neighbors (KNN)
- K-Means
- Autoencoder

## Ensemble Model

In addition to evaluating individual models, a hybrid model combining **KNN and Autoencoder** was developed.

The purpose of this approach is to combine the predictions of both models to improve anomaly detection performance.

## Evaluation Metrics

The models are evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC

Different visualizations, including confusion matrices and ROC curves, are also used to compare model performance.

## Visualizations

The project includes several visualizations to better understand the data and model results.

### Feature Importance Analysis

### Model Performance Comparison

### Model Comparison Heatmap

### Isolation Forest
- Confusion Matrix
- t-SNE Visualization

### One-Class SVM
- Confusion Matrix
- t-SNE Visualization

### Local Outlier Factor (LOF)
- Confusion Matrix
- t-SNE Visualization

### K-Nearest Neighbors (KNN)
- Confusion Matrix
- t-SNE Visualization

### K-Means
- Confusion Matrix
- t-SNE Visualization

### Autoencoder
- Confusion Matrix
- t-SNE Visualization

### ROC Curve Comparison

### Anomaly Count Comparison

### Ensemble Model (KNN + Autoencoder)
- Confusion Matrix
- ROC Curve
- Performance Heatmap
- t-SNE Visualization
- Anomaly Count Comparison

## Technologies and Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn

## How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
   ```

2. Install the required libraries:

   ```bash
   pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
   ```

3. Download the dataset and place it in your project folder.

4. Update the dataset file path in the Python code.

5. Run the Python script.

## Future Improvements

- Improving the data preprocessing and evaluation process.
- Experimenting with additional machine learning algorithms.
- Further tuning the ensemble model.
- Organizing the project into a Jupyter Notebook.

## Author

**Golnoosh Nasereslami**

Bachelor's Degree in Computer Science
