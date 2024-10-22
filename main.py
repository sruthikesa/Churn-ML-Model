from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import logging
import os
import uvicorn

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load the model
try:
    with open('xgb_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    loaded_model = None  # Set to None or handle as appropriate

# Define Pydantic model for input data
class CustomerData(BaseModel):
    CreditScore: float
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Geography: str
    Gender: str

def preprocess_data(customer_dict):
    input_dict = {
        'CreditScore': customer_dict.CreditScore,
        'Age': customer_dict.Age,
        'Tenure': customer_dict.Tenure,
        'Balance': customer_dict.Balance,
        'NumOfProducts': customer_dict.NumOfProducts,
        'HasCrCard': customer_dict.HasCrCard,
        'IsActiveMember': customer_dict.IsActiveMember,
        'EstimatedSalary': customer_dict.EstimatedSalary,
        'Geography_France': 1 if customer_dict.Geography == 'France' else 0,
        'Geography_Germany': 1 if customer_dict.Geography == 'Germany' else 0,
        'Geography_Spain': 1 if customer_dict.Geography == 'Spain' else 0,
        'Gender_Male': 1 if customer_dict.Gender == 'Male' else 0,
        'Gender_Female': 1 if customer_dict.Gender == 'Female' else 0,
    }
    
    customer_df = pd.DataFrame([input_dict])
    
    logger.info("Customer DataFrame:\n%s", customer_df)
    
    return customer_df

def get_predictions(customer_dict):
    if loaded_model is None:
        raise Exception("Model is not loaded.")
    
    preprocessed_data = preprocess_data(customer_dict)
    prediction = loaded_model.predict(preprocessed_data)
    probability = loaded_model.predict_proba(preprocessed_data)
    
    return prediction, probability

@app.post("/predict")
async def predict(data: CustomerData):
    try:
        prediction, probabilities = get_predictions(data)
        return {
            "prediction": prediction.tolist(),
            "probabilities": probabilities.tolist(),
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction error")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
