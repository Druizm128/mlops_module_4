import os
import sys
import json
import numpy as np
from ingestion import merge_multiple_dataframe
from training import train_model
from scoring import score_model
from deployment import store_model_into_pickle
from reporting import score_model_report
from diagnostics import model_predictions
from apicalls import get_api_responses, write_api_responses

##################Check and read new data
def read_new_data():
    #first, read ingestedfiles.txt
    with open(os.path.join(prod_deployment_path, 'ingestedfiles.txt'), 'r') as f:
        ingested_files=[line.strip() for line in f.readlines()]

    print(ingested_files)
    #second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    input_files=os.listdir(input_folder_path)

    new_input_files=[]
    for input_file in input_files:
        for ingested_file in ingested_files:
            if input_file != ingested_file and input_file not in new_input_files:
                new_input_files.append(input_file)
    print(new_input_files)
    return new_input_files

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
def is_there_new_data_available(new_input_files):
    if len(new_input_files) > 0:
        return True
    else:
        return False

#################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
def check_drift(f1_previous, f1_new):
    drift=False
    if f1_new < f1_previous:
        drift=Tue
    return drift

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
#if you found evidence for model drift, re-run the deployment.py script
#run diagnostics.py and reporting.py for the re-deployed model

if __name__ == "__main__":
    # Loading configurations
    print("Loading configuration paths ...")
    with open('config.json','r') as f:
        config = json.load(f) 
    
    prod_deployment_path = config['prod_deployment_path']
    input_folder_path = config['input_folder_path']
    output_model_path = os.path.join(config['output_model_path']) 
    test_data_path = os.path.join(config['test_data_path']) 
    prod_deployment_path = os.path.join(config['prod_deployment_path']) 
    model_path = os.path.join(config['output_model_path']) 
 
    #Specify a URL that resolves to your workspace
    URL = "http://127.0.0.1:8000"
    
    print("Checking for new files ...")
    new_input_files=read_new_data()
    print(new_input_files)
    bool_new_data=is_there_new_data_available(new_input_files)
    print(bool_new_data)
    
    if bool_new_data==False:
        print("There is no new data, ending program ...")
        sys.exit()
    
    print("There is new data, training new model")
    merge_multiple_dataframe()
    train_model()
    score_model()
    
    print("Checking drift between new and deployed model ...")
    with open(os.path.join(prod_deployment_path, "latestscore.txt")) as f:
        f1_score_deployed_model=f.readline().strip()
        f1_score_deployed_model=np.float64(f1_score_deployed_model)

    with open(os.path.join(output_model_path, "latestscore.txt")) as f:
        f1_score_new_model=f.readline().strip()
        f1_score_new_model=np.float64(f1_score_deployed_model)
    
    model_drift_found=check_drift(f1_score_deployed_model, f1_score_new_model)
    
    if model_drift_found==False:
        print("There is no model drift, ending program ...")
        print(f"Deployed model {f1_score_deployed_model}")
        print(f"New model {f1_score_new_model}")
        print(f"Drift {model_drift_found}")
        sys.exit()
    
    print("There was a model drift, deploying new model ...")
    store_model_into_pickle()
    model_predictions(file="testdata.csv")
    score_model_report(file="testdata.csv", fig_name="confusionmatrix2.png")

    print("Tesing api responses")
    api_responses=get_api_responses(URL, file_name='testdata.csv')
    write_api_responses(api_responses, model_path, file_out_name='apireturns2.txt')
    print("Success")