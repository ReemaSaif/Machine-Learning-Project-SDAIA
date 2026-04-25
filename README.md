# Machine-Learning-Project-SDAIA

## Bike Rental Demand Prediction (Random Forest + Gradio App)


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


