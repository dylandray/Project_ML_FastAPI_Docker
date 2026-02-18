from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder #transformers of scikit learn (L,H,m encoded)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib #save the model

# fetch dataset from the api directly, just to save memory on my computer
dataset = fetch_ucirepo(id=601) 
  
# data (as pandas dataframes) 
X = dataset.data.features.copy()
Y = dataset.data.targets["Machine failure"]

print(X.head())
print(Y.head())

#Encoding, L < M < H
encoder = OrdinalEncoder(categories=[["L","M","H"]])
X[["Type"]] = encoder.fit_transform(X[["Type"]])

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state= 23) #the model uses 1-0.2 lines of data to learn the logic and to train. It will then pass the "exam" on the 20% remaining
#_train : training phase, _test: testing phase
model = RandomForestClassifier(n_estimators= 100, random_state=23)
model.fit(X_train, Y_train) #Find links and draw conclusions for the training phase

Y_pred = model.predict(X_test) #now predicts the remaining lines
print(classification_report(Y_test, Y_pred)) #do a report 

joblib.dump(model, "maintenance_model.joblib") #save the joblib files in order to use them in main.py
joblib.dump(encoder, "type_encoder.joblib")
