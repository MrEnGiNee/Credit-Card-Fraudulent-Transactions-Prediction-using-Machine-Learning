# Credit-Card-Fraudulent-Analysis
This GitHub repository contains code and tools for analyzing credit card transactions, aiming to identify potential cases of fraud and determine the correlation between fraud and normal transactions.

The repository includes techniques to segregate the Kaggle datasets into fraud and normal transactions, transforming the non-uniform data into uniform data for further analysis.

To ensure effective model training, the data is separated into input features (X) and target variables (Y). Subsequently, test and training data are created to evaluate the logistic regression model's performance accurately.

The logistic regression model serves as the primary predictive tool to identify fraudulent transactions with a high level of accuracy for new, unseen real-world data. The model demonstrates impressive performance, achieving an accuracy rate of approximately 92-94% on both training and test datasets.

The analysis begins with exploratory data analysis, where visualizations are utilized to gain insights into the transaction class distribution, transaction amounts, and occurrence times. This step helps in better understanding the dataset and identifying potential patterns or anomalies related to fraudulent activities.

To address the issue of imbalanced data, under-sampling is applied, leading to the creation of a balanced dataset with an equal number of fraudulent and normal transactions. This balanced dataset ensures that the logistic regression model does not favor the majority class, contributing to more reliable predictions for both classes.

Overall, this GitHub repository provides a comprehensive approach to credit card fraud detection using logistic regression and exploratory data analysis. By leveraging machine learning techniques and data preprocessing methods, the model can effectively identify fraudulent transactions and provide valuable insights for enhancing security measures. The achieved accuracy showcases the model's robustness in handling real-world credit card transaction data, making it a valuable tool for credit card companies and customers to safeguard against fraudulent activities.
