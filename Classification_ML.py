"""
Created on Tue Jul  1 09:11:22 2025

@author: Jules Malavieille 
"""

# L'objectif ici est de faire de la classification avec toutes les méthodes possibles de ML
# On utilisera donc regression logistique, Random Forest, XGBoost et Reseau de neurones
# On comparera ensuite quels sont les méthodes les plus efficaces 
# Le data set contient :
# 10 mesure de spectrophotometrie à des longueurs d'onde différentes
# A quel classe appartient le phytoplancton associé à ces mesures (0, 1, 2, 3)
# 0 = Synechococcus 
# 1 = Prochlorococcus
# 2 = Thalassiosira (Diatomé)
# 3 = E. huxleyii (Cocolithophore)

import numpy as np
import matplotlib.pyplot as plt 
import xgboost as xgb
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier


def SPG_categorical(model, X, Y, k=5, lam=0.5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    SPG = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        model.fit(X_train, Y_train)
        
        score_train = model.score(X_train, Y_train)
        score_test = model.score(X_test, Y_test)
        
        spg_score = score_train - lam*abs(score_train - score_test)
        SPG.append(spg_score)
    
    return np.mean(SPG), np.std(SPG)


def model_change():
    input_size = X_train.shape[1]
    output_size = 4
    hidden_layer_size = 20

    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(input_size,)),
                                 tf.keras.layers.Dense(hidden_layer_size, activation="relu"),
                                 tf.keras.layers.Dense(hidden_layer_size, activation="relu"),
                                 tf.keras.layers.Dense(output_size, activation="softmax")
                                 ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
    

data = np.genfromtxt("phytoplankton_spectra_dataset.csv", delimiter=",", skip_header=True)

variables = data[:,:-1]
target = data[:,-1]

X_train, X_test, Y_train, Y_test = train_test_split(variables, target, test_size=0.1)

"""Regression logistique"""
LR = LogisticRegression()
LR.fit(X_train, Y_train)

print()
print("L'accuracy du modèle logistique sur les données train :", LR.score(X_train, Y_train))
print("L'accuracy du modèle logistique sur les données test :", LR.score(X_test, Y_test))
print()


"""Random forest"""
params = {"n_estimators":[100, 200], "max_depth":[None, 5, 10], "min_samples_split":[2,5], "min_samples_leaf":[1, 3, 6]}

# RF_param = RandomForestClassifier(random_state=42)
# grid = GridSearchCV(RF_param, params, cv=5, scoring="accuracy")
# grid.fit(X_train, Y_train)
# Paramètres sont trouvé dans grid.best_params_

RF = RandomForestClassifier(random_state=42, min_samples_leaf=1, min_samples_split=2, n_estimators=200)
RF.fit(X_train, Y_train)

print("L'accuracy de la Random Forest sur les données train :", RF.score(X_train, Y_train))
print("L'accuracy de la Random Forest sur les données test :", RF.score(X_test, Y_test))
print()


"""XGBoost"""
params = {"n_estimators":[100, 200], "max_depth":[2, 3, 4], "learning_rate":[0.01,0.05, 0.1], "subsample":[0.6, 0.8, 1.0], "colsample_bytree":[0.6, 0.8, 1.0]}

# RF_param = xgb.XGBClassifier(random_state=42)
# grid = GridSearchCV(RF_param, params)
# grid.fit(X_train, Y_train)
# Paramètres sont trouvé dans grid.best_params_

XGB = xgb.XGBClassifier(random_state=42, subsample=0.6, n_estimators=200, max_depth=3, learning_rate=0.05, colsample_bytree=0.6)
XGB.fit(X_train, Y_train)

print("L'accuracy du XGBoost sur les données train :", XGB.score(X_train, Y_train))
print("L'accuracy du XGBoost sur les données test :", XGB.score(X_test, Y_test))
print()


"""Réseau de neurones"""
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.111)

input_size = X_train.shape[1]
output_size = 4
hidden_layer_size = 20

model = tf.keras.Sequential([tf.keras.layers.Input(shape=(input_size,)),
                             tf.keras.layers.Dense(hidden_layer_size, activation="relu"),
                             tf.keras.layers.Dense(hidden_layer_size, activation="relu"),
                             tf.keras.layers.Dense(output_size, activation="softmax")
                             ])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

stop = EarlyStopping(patience=5, restore_best_weights=True)

n_epo = 100
model.fit(X_train, Y_train, epochs=n_epo, validation_data=(X_val, Y_val), callbacks=[stop])

print()
print("L'accuracy du réseau de neurones sur les données train :", model.evaluate(X_train, Y_train)[1])
print("L'accuracy du réseau de neurones sur les données test :", model.evaluate(X_test, Y_test)[1])
print()

"""Sélection du meilleur model"""
variable_scaled = scaler.fit_transform(variables)

model_sk = KerasClassifier(model=model_change, epochs=50, batch_size=32)

spg_reg, std_reg = SPG_categorical(LR, variables, target, lam=1)
spg_rf, std_rf = SPG_categorical(RF, variables, target, lam=1)
spg_xgb, std_xgb = SPG_categorical(XGB, variables, target, lam=1)
spg_rn, std_rn = SPG_categorical(model_sk, variable_scaled, target, lam=1)

print("Score SPG, régression ML =", spg_reg,"+/_",std_reg)
print("Score SPG, Random forest =", spg_rf,"+/_",std_rf)
print("Score SPG, XGBosst =", spg_xgb,"+/_",std_xgb)
print("Score SPG, Réseau de neurones =", spg_rn,"+/_",std_rn)
print()

# Donc ici tous les modèles sont presque identiques en performance 
# Avec un très leger avantage pour le modèle logistique













