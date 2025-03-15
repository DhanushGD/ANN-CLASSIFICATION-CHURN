import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle 
import numpy as np


### Load the ANN trained mode,scaler pickle and onehotencoded pickle file
model = tf.keras.models.load_model('DAY - 4 ANN project/model.h5')
with open('DAY - 4 ANN project/label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender=pickle.load(file)
with open('DAY - 4 ANN project/onehot_encoder_geo.pkl', 'rb') as file:
    label_encoder_geo=pickle.load(file)
with open('DAY - 4 ANN project/scaler.pkl','rb') as file:
    scaler=pickle.load(file)

# Example input data
input_data = {
    'CreditScore': 600,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000
}
geo_encoded = label_encoder_geo.transform([[input_data['Geography']]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))
input_df=pd.DataFrame([input_data])

## Encode categorical variables
input_df['Gender']=label_encoder_gender.transform(input_df['Gender'])
input_df=pd.concat([input_df.drop("Geography",axis=1),geo_encoded_df],axis=1)

## Scaling the input data
input_scaled=scaler.transform(input_df)

## PRedict churn
prediction=model.predict(input_scaled)

prediction_proba = prediction[0][0]
if prediction_proba > 0.5:
    print('The customer is likely to churn.')
else:
    print('The customer is not likely to churn.')
