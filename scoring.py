from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import joblib
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

output_model_path = os.path.join(config['output_model_path']) 
test_data_path = os.path.join(config['test_data_path']) 


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    tbl_test=pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    model=joblib.load(os.path.join(output_model_path, "trainedmodel.joblib"))
    X_test=tbl_test.drop(columns=['corporation', 'exited']) 
    y_true=tbl_test['exited'].values
    y_pred=model.predict(X_test)
    f1_score=metrics.f1_score(y_true,y_pred)
    with open(os.path.join(output_model_path,'latestscore.txt'),'a') as f:
        f.write("\n")
        f.write(str(f1_score))
    return f1_score
if __name__ == '__main__':
    score_model()
