"""
Tensorflow is open source library to create a neural networks like simple RNN,LSTM,ANN etc
Keras is a api will be on top of tensorflow ,like rapper for tendorflow

if we have a tabular data,classification or regression problem we can use ANN
problem statement - is to find whether a person will leave the bank or not based on previous past data
 

When to use ANN (Artificial Neural Network):
Use an ANN when you need to model complex patterns or relationships in data, like image recognition, language processing, or prediction tasks.

Bias:A bias is a value added to the output of a neuron to shift the activation function, helping the network learn more effectively.
Activation function:It transforms the neuron’s input into an output that will either "fire" or "not fire.
Weights:Weights are values that determine the strength of the connection between two neurons, influencing the output of a neural network.
Forward Propagation:Forward propagation is the process of passing input data through the network to get the output (prediction).
Backward Propagation:Backward propagation is the process of adjusting the weights and biases based on the error (difference between predicted and actual output) to improve the network’s performance.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle  #for reusing the model/file when deploying

data =pd.read_csv("DAY - 4 ANN project\data.csv")

## preprocess the data
data=data.drop(['RowNumber','CustomerId','Surname'],axis=1)  #axis =1 is column wise

## encode catergorical variables
label_enocoder_gender = LabelEncoder()
data['Gender']=label_enocoder_gender.fit_transform(data['Gender'])

## onehot enocode geography column since it has many different variables
onehot_encoder_geo=OneHotEncoder()
geo_encoder=onehot_encoder_geo.fit_transform(data[['Geography']]).toarray()
onehot_encoder_geo.get_feature_names_out(['Geography'])
geo_encoded_df=pd.DataFrame(geo_encoder,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

## Combine one hot encoder columns with the original data
data=pd.concat([data.drop('Geography',axis=1),geo_encoded_df],axis=1)

## save the encoders and scaler

with open('DAY - 4 ANN project/label_encoder_gender.pkl','wb') as file:
    pickle.dump(label_enocoder_gender,file)
with open('DAY - 4 ANN project/onehot_encoder_geo.pkl','wb') as file:
    pickle.dump(onehot_encoder_geo,file)

## divide the dataset into dependent and independent features
X=data.drop('Exited',axis=1)  #independent features/variables
y=data['Exited'] #dependent features/variables

"""
X = data.drop('Exited', axis=1):
We are creating the independent features X by dropping the 'Exited' column from the dataset (data). 
The axis=1 means we are dropping a column, not a row.
X will contain all the columns except 'Exited', which represents 
the features that will be used to predict the target variable.

y = data['Exited']:
We are creating the dependent variable y, which is the target we are trying to predict. 
In this case, 'Exited' is the target variable (whether a customer left or not, for example).
y will contain just the values of the 'Exited' column.
"""

## split the data in training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# print(X_train,X_test,y_train,y_test)
## scaling the features for independent features
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train) #fit() learns the parameters (mean, standard deviation for scaling). & transform() applies the learned parameters to the data.
X_test=scaler.transform(X_test) #transform() applies the learned parameters to the data.

with open('DAY - 4 ANN project/scaler.pkl','wb') as file:
    pickle.dump(scaler,file)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense   #for creating dense node(hidden layer and output layer)
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
import datetime
"""
basically in ANN there will be, for example 2 i/p layer multiple hidden layer and 1 o/p layer,and in tensorflow ANN is called as sequential model
Sequential Model = Layers stacked linearly.
ANN = Can be structured sequentially or more complexly depending on the architecture.
"""

## build our ANN model
model=Sequential([
    Dense(64,activation='relu',input_shape=(X_train.shape[1],)),  ##1st hidden layer connected with input layer
    Dense(32,activation='relu'), ##2nd hidden layer - no need to mention input because all the hidden layers are interconnected,so it knows
    Dense(1,activation='sigmoid')  ##output layer
]
)
print(model.summary())   #we can see the total number of parameters

import tensorflow
opt=tensorflow.keras.optimizers.Adam(learning_rate=0.01)

## compile the model
model.compile(optimizer=opt,loss="binary_crossentropy",metrics=['accuracy'])

## Set up the Tensorboard
import os
base_dir = "DAY - 4 ANN project"
log_dir = os.path.join(base_dir, "logs", "fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(log_dir, exist_ok=True)
tensorflow_callback=TensorBoard(log_dir=log_dir,histogram_freq=1)

## Set up Easly stopping
early_stopping_callback=EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)


## Train the model
history=model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,callbacks=[tensorflow_callback,early_stopping_callback])  #Epoch = 1 pass through the entire dataset.
model.save('DAY - 4 ANN project/model.h5')

## Load Tensorboard extension
# tensorboard --logdir="DAY - 4 ANN project/logs/fit"  in terminal

"""
TensorBoard is a visualization tool used to monitor and analyze the 
training process of machine learning models by displaying metrics like loss, accuracy, and other visualizations such as graphs and histograms.
"""



