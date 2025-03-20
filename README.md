# IronKaggle_project - Machine Learning

The team was tasked to undertake a project related to the King County dataset for the housing market.
The challenge aimed at pushing our boundaries in the topic of Machine Learning and specifically the Regression models.

The dataset is structured as per below:

- id: A unique identifier for a house.
- date: The date on which the house was sold.
- price: The sale price of the house (prediction target).
- bedrooms: Number of bedrooms in the house.
- bathrooms: Number of bathrooms in the house, per bedroom.
- sqft_living: Square footage of the interior living space.
- sqft_lot: Square footage of the land space.
- floors: Number of floors (levels) in the house.
- waterfront: Whether the house has a waterfront view.
- view: Number of times the house has been viewed.
- condition: The overall condition of the house.
- grade: The overall grade given to the house, based on the King County grading system.
- sqft_above: Square footage of the house apart from the basement.
- sqft_basement: Square footage of the basement.
- yr_built: The year the house was built.
- yr_renovated: The year the house was renovated.
- zipcode: ZIP code area.
- lat: Latitude coordinate.
- long: Longitude coordinate.
- sqft_living15: The interior living space for the nearest 15 neighbors in 2015.
- sqft_lot15: The land spaces for the nearest 15 neighbors in 2015.
- TARGET --> Price: Our primary focus is to understand which features most significantly impact the house price. Additionally, we aim to explore properties valued at $650K and above for more detailed insights.

Upon receiving the dataset for the task the team has aligned on the steps each of the members would focus on during the whole process.

At the beginning each of us had the freedom to clear the data as they preferred and once the EDA process was completed, the team has aligned once again to discuss the next steps.

All together we defined the correlation matrix and selected the features which would have the most impact on the Regression models, these being the following:

- sqft_living
- grade
- sqft_above
- bathrooms

  The rest of the features had a weak correlation.

  At this point Marta and Mirko took on the task to create a baseline LinearRegression model without making any further feature selection or engineering.

  The results were the following:

R2 =  0.7
RMSE =  43223918065.38
MAE =  127752.47

Although the overall R2 score was able to explain 70% of the variations (which is a fairly good result considering no feature engineering has been applied at this point) the root squared error and the mean absolute error were showing extreme results which means that the model is unstable and further feature preparation is required.

Marta and Mirko plotted the numerical features and shared information with Marc who in the meantime had already learned about model parameters for LinearRegression, DecisionTree and XGB.

Each of the team member was tasked to deal with the features to make the models more stable and each of us focused on a different regression model.

The team noticed that the feature which had the most extreme outliers was the target feature: price.

![image](https://github.com/user-attachments/assets/f6852e05-fd3e-4968-97a1-00afa266a5c1)

The price feature appears skewed on the left and has therefore extreme outliers on the right.

The team has therefore focused on reducing the noise introduced by the outliers.

While doing so, Marc has found the cap for the numerical features which resulted into the best regression results.
The results were achieved via the below string of code:

# Cap outliers at 5th and 95th percentiles:
numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns  
lower_limit = df_cleaned[numeric_columns].quantile(0.05)  
upper_limit = df_cleaned[numeric_columns].quantile(0.95) 

The above code caps outliers in numerical columns at the 5th and 95th percentiles, ensuring that extreme values don't skew the dataset.

After this feature engineering process the result achieved with the RandomForest model is the following:

R² = 0.8867
RMSE = 85258.4252
MAE = 57231.8505
MSE = 7268999073.8572
  



 
