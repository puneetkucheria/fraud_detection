from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd

app = FastAPI()

#import model
model = joblib.load('model_classifier.pkl')

class Fraud_input_parms(BaseModel):
    # if all type are 0 the transection is type CASH_IN
    step: int = 0
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    isFlaggedFraud: int
    type_CASH_OUT: bool
    type_DEBIT: bool
    type_NOTYPE: bool
    type_PAYMENT: bool
    type_TRANSFER: bool
    
@app.get('/')
async def root():
    return{"status":"Online"}

@app.post('/fraud/')
async def fraud(parms:Fraud_input_parms):
    # print("Parameters : " + str(parms.step))
    re = model.predict(pd.DataFrame([[parms.step,parms.amount,parms.oldbalanceOrg,parms.newbalanceOrig,parms.oldbalanceDest,parms.newbalanceDest,parms.isFlaggedFraud,parms.type_CASH_OUT,parms.type_DEBIT,parms.type_NOTYPE,parms.type_PAYMENT,parms.type_TRANSFER]]))
    # print(re)
    return {"is_fraud": str(re[0])}