from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder #transformers de scikit learn (low, high -> chiffres)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib #sauvegarde le modèle

# fetch dataset 
dataset = fetch_ucirepo(id=601) 
  
# data (as pandas dataframes) 
X = dataset.data.features.copy()
Y = dataset.data.targets["Machine failure"]

print(X.head())
print(Y.head())

#Encodage, L < M < H
encoder = OrdinalEncoder(categories=[["L","M","H"]])
X[["Type"]] = encoder.fit_transform(X[["Type"]])

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state= 23) #le modèle utilise 1-0.2 données pour apprendre la logique, se train, et se testera sur les 20% restants
#_train : training phase, _test: testing phase
model = RandomForestClassifier(n_estimators= 100, random_state=23)
model.fit(X_train, Y_train) #il analyse les lignes d'entrainements et tire des conclusions, trouve des liens

Y_pred = model.predict(X_test) #maintenant il prédit les lignes restantes
print(classification_report(Y_test, Y_pred)) #fait un rapport 

joblib.dump(model, "maintenance_model.joblib")
joblib.dump(encoder, "type_encoder.joblib")
