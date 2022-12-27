from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
#import create_prediction_model
#import diagnosis 
#import predict_exited_from_saved_model
import json
import os
from scoring import score_model
from diagnostics import model_predictions, dataframe_summary, dataframe_missing_values, execution_time, outdated_packages_list


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 




#######################Prediction Endpoint
# , methods=['POST','OPTIONS']
@app.route("/prediction")
def predict():        
    #call the prediction function you created in Step 3
    # "testdata.csv"
    file_name=request.args.get('filename')
    prediction_model = model_predictions(file=file_name)
    return {'predictions': str(prediction_model)}
    #return str(file_name)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():        
    #check the score of the deployed model
    f1_score=score_model()
    return {'f1_score': str(f1_score)}

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    stats=dataframe_summary()
    return {'summary_stats': str(stats)}

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    missing_values=dataframe_missing_values()
    exec_time=execution_time()
    outdated_packages=outdated_packages_list()
    return {
        'missing_values': str(missing_values), 
        'execution_time': str(exec_time),
        'dependency_check': str(outdated_packages)
    }

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
