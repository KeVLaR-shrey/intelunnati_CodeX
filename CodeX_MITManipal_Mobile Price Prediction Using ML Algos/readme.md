# Mobile Price Prediction: A Machine Learning Approach

## Introduction

This project aims to predict the price of mobile phones using various features such as RAM, weight, screen size, and more. The dataset used for this project contains 700 rows and 40 columns, each representing a unique feature of a mobile phone. The data was scraped from various online sources using beautifulsoup and selenium and cleaned for the purpose of this project.

## Technologies Used

The project is implemented in Python, leveraging several libraries for data manipulation, numerical computations, data visualization, statistical modeling, and machine learning algorithms. Here are the main libraries used:

Pandas: For data manipulation and analysis.
Numpy: For numerical computations.
Matplotlib and Seaborn: For data visualization.
Plotly and Bokeh: For creating interactive plots.
Statsmodels and Seaborn: For statistical modeling.
Scikit-learn: For machine learning algorithms and related tasks (like splitting the dataset, feature scaling, etc.).

## Methodology

### Data Cleaning and Preprocessing

The raw data was initially cleaned by removing any irrelevant columns and handling missing values. The missing values were either removed if minimal or filled with the obvious value as interpreted from website. The data was then preprocessed to make it suitable for machine learning models. This involved transforming certain features to reduce skewness and improve the distribution of the data. For instance, right-skewed columns were log-transformed, while left-skewed columns were square transformed. Outliers were handled with minimal manipulation as they represented valuable data of mobile phone market volatility.


![image](https://github.com/KeVLaR-shrey/intelunnati_CodeX/assets/91597263/685bb483-e4a5-40d1-852c-86193a70308f)



### Exploratory Data Analysis (EDA)



![image](https://github.com/KeVLaR-shrey/intelunnati_CodeX/assets/91597263/09de6a18-5771-4e6d-a872-2a2f34b582b0)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


![image](https://github.com/KeVLaR-shrey/intelunnati_CodeX/assets/91597263/fe6b15ea-ab9e-430a-8e3a-c6b1758195df)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;




The EDA process involved analyzing the data to understand the relationships between different features and their impact on the target variable, i.e., the price of the mobile phone. This was done using various statistical techniques and data visualization tools. 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;




![image](https://github.com/KeVLaR-shrey/intelunnati_CodeX/assets/91597263/4ccc52ed-2054-41eb-a94a-ea5d24c22004)
&nbsp;&nbsp;&nbsp;
![image](https://github.com/KeVLaR-shrey/intelunnati_CodeX/assets/91597263/fda5b263-fd61-4caa-b5a2-f1e7faa2f71f)
&nbsp;&nbsp;&nbsp;
![image](https://github.com/KeVLaR-shrey/intelunnati_CodeX/assets/91597263/f6665c47-2acb-4191-bf55-efe756d9dec9)
&nbsp;&nbsp;&nbsp;
![image](https://github.com/KeVLaR-shrey/intelunnati_CodeX/assets/91597263/a12b9275-3dce-4d7d-9f0e-cfcdadad2628)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;



The correlation between different features was analyzed, and the features with high correlation were identified. The distribution of different features was also studied to understand their spread and skewness.The correlation between different features was analyzed, and the features with high correlation were identified. The distribution of different features was also studied to understand their spread and skewness.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


![image](https://github.com/KeVLaR-shrey/intelunnati_CodeX/assets/91597263/17ea270e-0b09-40af-b342-4628e7a04d27)

&nbsp;&nbsp;&nbsp;
### Feature Engineering

Feature engineering was performed to create new features that could potentially improve the performance of the machine learning models. This included creating interaction features, which are new features formed by combining two or more existing features. For instance, the total number of pixels in a phone's display was calculated by multiplying the resolution in the X-axis by the resolution in the Y-axis.

### Model Selection and Training

The model selection process involved training various regression models and tuning their hyperparameters to find the best performing model. The models used included Linear Regression, Support Vector Regressor, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, and several others. The hyperparameters of these models were tuned using RandomizedSearchCV, which performs a random search on hyperparameters to find the best parameters for the model.

### Model Evaluation

The performance of the models was evaluated using three metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R2). 

The MSE and MAE provide a measure of the prediction error of the models, with lower values indicating better performance. The R2 score provides a measure of how well the model explains the variance in the data, with higher values indicating better performance.

## Results




![image](https://github.com/KeVLaR-shrey/intelunnati_CodeX/assets/91597263/848f5f17-5146-4b4d-a55d-da0de06e77dd)

&nbsp;&nbsp;&nbsp;


Model Performance Metrics
The performance metrics of the models are as follows:

- **Linear Regression**: 
  - MSE = 0.150
  - MAE = 0.306
  - R2 = 0.848

- **SVR**: 
  - MSE = 0.128
  - MAE = 0.272
  - R2 = 0.870

- **Decision Tree**: 
  - MSE = 0.212
  - MAE = 0.351
  - R2 = 0.786

- **Random Forest**: 
  - MSE = 0.157
  - MAE = 0.295
  - R2 = 0.842

- **Gradient Boosting**: 
  - MSE = 0.123
  - MAE = 0.265
  - R2 = 0.876

- **XGBoost**: 
  - MSE = 0.114
  - MAE = 0.252
  - R2 = 0.885

- **LightGBM**: 
  - MSE = 0.137
  - MAE = 0.278
  - R2 = 0.862

- **CatBoost**: 
  - MSE = 0.115
  - MAE = 0.254
  - R2 = 0.884

- **HistGradientBoosting**: 
  - MSE = 0.135
  - MAE = 0.272
  - R2 = 0.864

- **Bagging**: 
  - MSE = 0.138
  - MAE = 0.269
  - R2 = 0.861

- **K-Neighbors**: 
  - MSE = 0.161
  - MAE = 0.295
  - R2 = 0.838

### Learning Curves for top models



![image](https://github.com/KeVLaR-shrey/intelunnati_CodeX/assets/91597263/17c04965-bc48-4b25-8cb9-3e50590d4c9c)

![image](https://github.com/KeVLaR-shrey/intelunnati_CodeX/assets/91597263/7af71f7d-489c-47eb-9c6b-cee9882d58e0)

![image](https://github.com/KeVLaR-shrey/intelunnati_CodeX/assets/91597263/9538fed4-0bf2-49a9-998a-d06ad86db258)

![image](https://github.com/KeVLaR-shrey/intelunnati_CodeX/assets/91597263/2a9b2cb3-469a-45cb-968a-25e7fd8805a0)

![image](https://github.com/KeVLaR-shrey/intelunnati_CodeX/assets/91597263/bd395dcb-70d9-4677-bf13-67490b82b9b0)
&nbsp;&nbsp;&nbsp;


The best performing model was the XGBoost Regressor, which achieved the lowest MSE and MAE, and the highest R2 score. However, other models such as the CatBoost Regressor, Gradient Boosting Regressor, Support Vector Regressor, Bagging Regressor, and LightGBM Regressor also performed closely.

## Conclusion and Future Work

This project demonstrates the potential of machine learning in predicting mobile phone prices with a high degree of accuracy. The best performing model, XGBoost, was able to explain a significant portion of the variance in the data, indicating its effectiveness in capturing the underlying patterns in the data.

In terms of future work, there are several avenues that can be explored to further improve the model's performance. This includes collecting more data, especially for mobile phones with unique features that are not well-represented in the current dataset. Additionally, more advanced feature engineering techniques can be applied to create new features that can potentially improve the model's predictive power. Finally, more advanced machine learning models and ensemble methods can be explored to further improve the model's performance.

