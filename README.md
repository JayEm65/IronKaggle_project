# IronKaggle Project - Machine Learning

The team was tasked to undertake a project related to the King County dataset for the housing market. The challenge aimed at pushing our boundaries in the topic of Machine Learning and specifically the Regression models.

## Dataset Structure

The dataset is structured as per below:

- **id**: A unique identifier for a house.
- **date**: The date on which the house was sold.
- **price**: The sale price of the house (prediction target).
- **bedrooms**: Number of bedrooms in the house.
- **bathrooms**: Number of bathrooms in the house, per bedroom.
- **sqft_living**: Square footage of the interior living space.
- **sqft_lot**: Square footage of the land space.
- **floors**: Number of floors (levels) in the house.
- **waterfront**: Whether the house has a waterfront view.
- **view**: Number of times the house has been viewed.
- **condition**: The overall condition of the house.
- **grade**: The overall grade given to the house, based on the King County grading system.
- **sqft_above**: Square footage of the house apart from the basement.
- **sqft_basement**: Square footage of the basement.
- **yr_built**: The year the house was built.
- **yr_renovated**: The year the house was renovated.
- **zipcode**: ZIP code area.
- **lat**: Latitude coordinate.
- **long**: Longitude coordinate.
- **sqft_living15**: The interior living space for the nearest 15 neighbors in 2015.
- **sqft_lot15**: The land spaces for the nearest 15 neighbors in 2015.
- **TARGET --> Price**: Our primary focus is to understand which features most significantly impact the house price. Additionally, we aim to explore properties valued at $650K and above for more detailed insights.

## Process Overview

Upon receiving the dataset for the task, the team aligned on the steps each of the members would focus on during the whole process.

At the beginning, each of us had the freedom to clean the data as we preferred, and once the EDA process was completed, the team aligned again to discuss the next steps.

Together, we defined the correlation matrix and selected the features which would have the most impact on the Regression models. These are:

- **sqft_living**
- **grade**
- **sqft_above**
- **bathrooms**

**id**, **date**, and **zipcode**: These columns were removed because they had low correlation with the target variable (price), meaning they did not provide meaningful information for predicting house prices.

**sqft_above** and **sqft_basement**: These were dropped due to redundancy. Combined, they provided the same information as the **sqft_living** column, making them unnecessary for the analysis.

After removing these features, the correlation matrix was recalculated to assess the relationships between the remaining variables and the target variable (price).

At this point, Marta and Mirko took on the task to create a baseline `LinearRegression` model without making any further feature selection or engineering.

### Baseline Model Results:

- **R²** = 0.7
- **RMSE** = 43223918065.38
- **MAE** = 127752.47

Although the overall R² score was able to explain 70% of the variations (which is a fairly good result considering no feature engineering had been applied), the root squared error and the mean absolute error showed extreme results, indicating the model was unstable and further feature preparation was required.

Marta and Mirko plotted the numerical features and shared information with Marc, who in the meantime had already learned about model parameters for `LinearRegression`, `DecisionTree`, and `XGB`.

Each team member was tasked with handling the features to make the models more stable, focusing on different regression models.

## Outlier Detection and Feature Engineering

The team noticed that the feature with the most extreme outliers was the target feature: **price**.

![Price Distribution](https://github.com/user-attachments/assets/f6852e05-fd3e-4968-97a1-00afa266a5c1)

The price feature appears skewed on the left and has extreme outliers on the right. The team focused on reducing the noise introduced by these outliers.

Marc found the cap for the numerical features, resulting in the best regression results. The following code was used to cap outliers:

```python
# Cap outliers at 10th and 90th percentiles:
numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns  
lower_limit = df_cleaned[numeric_columns].quantile(0.10)  
upper_limit = df_cleaned[numeric_columns].quantile(0.90)
```

This code caps outliers in numerical columns at the 10th and 90th percentiles, ensuring that extreme values don't skew the dataset.

After this feature engineering process, the results achieved with the `RandomForest` model were:

- **R²** = 0.8795
- **RMSE** = 85258.4252
- **MAE** = 57231.8505
- **MSE** = 7268999073.8572

### Final XGB Model Results:
An even better result was achieved with the `XGB` model:

- **R²** = 0.9941
- **RMSE** = 8.6931
- **MAE** = 6.3539
- **MSE** = 75.5693

To ensure the model wasn't overfitted, cross-validation was implemented. The results showed a mean cross-validation RMSE of 7.9858.

#### Cross-Validation RMSE Scores:
- [7.62387532 8.48389226 7.88998129 8.18427953 7.74694593]

#### Mean Cross-Validation RMSE:
- **7.985794867160695**

This makes our `XGB` model a good model to predict house values, with an average margin of error under $10,000.

## Conclusion
For further information on the results and the models used, please refer to the presentation attached in the repository.

### Team Members:
- [Marta](https://github.com/martasamuel)  
- [Mirko](https://github.com/MC993)  
- [Marc](https://github.com/JayEm65)
