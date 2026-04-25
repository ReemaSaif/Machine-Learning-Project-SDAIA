# Machine-Learning-Project-SDAIA

## Bike Rental Demand Prediction (Random Forest + Gradio App)


**The Goal:**
The goal of this model is to predict how many bikes will be rented per hour (cnt) given weather and seasonal conditions, so the bike‑sharing company can better plan bikes allocation, staffing, and maintenance.

**The Dataset Coulmns:**
- instant: record index
- dteday : date
- season : season (1 : springer, 2 : summer, 3 : fall, 4 : winter)
- yr : year (0: 2011, 1:2012)
- mnth : month ( 1 to 12)
- hr : hour (0 to 23)
- holiday : weather day is holiday or not 
- weekday : day of the week
- workingday : if day is neither weekend nor holiday is 1, otherwise is 0
-  weathersit : 
	1. Clear, Few clouds, Partly cloudy, Partly cloudy
	2. Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
	3. Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
	4.  Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
- temp : Normalized temperature in Celsius. The values are divided to 41 (max)
- atemp: Normalized feeling temperature in Celsius. The values are divided to 50 (max)
- hum: Normalized humidity. The values are divided to 100 (max)
- windspeed: Normalized wind speed. The values are divided to 67 (max)
- casual: count of casual users
- registered: count of registered users
- cnt: count of total rental bikes including both casual and registered


### **Problem Framing For The Model:**

1.	State the goal
*Current/non-ML solution:*

Current / non‑ML solution the company likely uses:
   - Human decision-making: Operators use experience to estimate how many bikes to deploy in each area.
   - Historical data insights: Average rentals by month or season.
   - Heuristic rules: For example, weekends are busier than weekdays.
   - Period comparisons: Demand patterns for similar times (e.g., same day last week or last year).
   - Weather-based rules: For example, rain reduces bike demand.


These methods suffer from subjective decision‑making, limited scalability, an inability to capture complex interactions, and lower accuracy that often leads to over‑ or under‑supply.


*Application, Goal, Description*

* **Application:** Demand forecasting for a bike-sharing system
* **Goal:** Predict hourly bike rental demand.
* **Description:** Using the available features in the dataset, the model predicts how many bikes will be rented at a specific hour.


*ML Task:*

* **Type:** Supervised Learning
* **Task:** Regression
* **Input features:** season, yr, mnth, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed
* **Target Variable:** cnt 	(count of total rental bikes = casual + registered)

2.	Clear use case for ML: Difference, Cost, Maintenance, and Expertise requirements.

**Difference: How much better do you think an ML solution can be?**
* An ML solution performs better because it can detect complex, nonlinear relationships between weather, seasonality, holidays, and daily demand, and it continuously refine its predictions as new data arrives. In similar real‑world rental systems, ML models have reduced forecasting errors by about 20–40%, which leads to fewer bike shortages, less over‑supply, and smoother day‑to‑day operations.
  
**Cost: How expensive is the ML solution in both the short- and long-term?**
* In the short term, an ML solution requires investment data cleaning and preprocessing, feature engineering, model training and evaluation, and deployment. Long‑term costs come from ongoing model retraining (monthly or quarterly), monitoring for data drift, and maintaining the deployment infrastructure to ensure the model remains accurate and reliable over time.
  
**Maintenance: How much maintenance will the solution require?**
* ML models require ongoing maintenance, including scheduled retraining (monthly or quarterly), continuous monitoring for data drift caused by changes in weather patterns, holidays, or user behavior, and updating features whenever new or improved data sources become available.
  
**Expertise: Does your product have the resources to support training or hiring people with ML expertise?** 
* Yes — the product has the resources to support training or hiring people with the required ML expertise.


3.	Does ART apply to the data?
* **Available:** Yes, All required input features are available at prediction time, such as season, yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed. And the target variable (cnt) is correctly excluded from the input features.
* **Representative:** Yes, The dataset covers two full years (2011–2012) with all seasons, holidays, weekdays/weekends, and weather conditions, making it representative of real-world bike rental demand.
* **Trusted:** Yes, The data comes from reliable sources, including the bike-sharing system’s own logs and trusted external weather providers, making it suitable and trustworthy for modeling.

4.	What is the quantity and quality of the data?

*Quantity*
      * 17,379 rows (hourly records for 2 years)
      * 16 columns (including weather, calendar, and demand variables)

*Quality*
    •	No missing values reported in the original dataset description.
    •	Features are already normalized for temp, atemp, hum, and windspeed.
    •	Categorical variables are encoded as integers (season, yr, mnth, holiday, weekday, workingday, weathersit).
    •	The data is clean, consistent, and requires minimal preprocessing for modeling.

5.	What features have been engineered?
   
   * is_weekend — 1 if Saturday/Sunday, else 0
   * is_rush_hour — 1 if the hour falls within 7–9 AM or 16–19 PM, else 0
   * season_peak — 1 if the season is fall (season = 3), else 0
   * temperature_bin— bin temperature into cold / mild / hot groups

6.	Which features have the most predictive power?
Based on the model feature importance results, the feature with the highest predictive power is: 
*is_rush_hour:* This feature has the strongest impact on the model prediction, indicating that commuting patterns are the main factor of hourly bike rental demand.

7.	What is the prediction of the model, and how is the decision based on it?

The model predicts the number of bikes rented in a given hour using inputs such as weather, season, temperature, humidity, and weekday/weekend indicators, and outputs a forecasted demand value. These predictions help the company decide how many bikes to deploy, how to schedule staff, and when to perform maintenance, ensuring efficient and reliable operations.

8.	What are the model’s metrics?

* The model is evaluated using RMSE, MAE, and R².
* RMSE and MAE measure the size of the prediction errors, while R² measures how well the model explains variation in hourly bike rental demand. Together, these metrics provide a complete view of model accuracy and performance. In this project, the Random Forest Regressor performs better because it can capture non‑linear patterns and complex interactions in the bike rental data, whereas Linear Regression cannot.


9.	What are the success/failure criteria?

**Success Criteria:**
    * Low RMSE and MAE
    * High R²
    * Model outperforms the baseline (Linear Regression)
    * Predictions are reliable enough to support company decision-making for bike allocation, staffing, and maintenance
    
**Failure Criteria:**
   * High RMSE and MAE
   * Low R²
   * Model does not outperform the baseline method
   * Predictions are inconsistent or not useful for operational decision‑making



### Setup and Usage

This section explains how to set up the environment, load the trained model, and run predictions using the Gradio UI.

**Required Files:**
Ensure the following files are located in the project directory:

* randomforest_model.pkl — trained Random Forest model
* app.py — Gradio deployment script
* requirements.txt — dependency list


**1. Environment Setup:**
Create a new environment and install the required packages:
```
pip install -r requirements.txt
```

**2. Loading the Trained Model:**
You can load the saved model using joblib:
```
import joblib
model = joblib.load("randomforest_model.pkl")
```

**3. Running the Gradio App (Model UI Deployment):**
```
Click the URL: https://8f49be505f6816cefd.gradio.live
```
A browser window will open with an interactive interface where you can enter feature values and get predictions instantly.


