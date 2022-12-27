
import pandas as pd
import numpy as np
import timeit
import os
import json
import joblib
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path']) 
diagnostics_path = os.path.join(config['diagnostics_path']) 

##################Function to get model predictions
def model_predictions(file):
    #read the deployed model and a test dataset, calculate predictions
    tbl_data=pd.read_csv(os.path.join(test_data_path, file))
    X = tbl_data.drop(columns=['corporation', 'exited']) 
    y = tbl_data['exited']
    model=joblib.load(os.path.join(model_path, "trainedmodel.joblib"))
    y_pred=list(model.predict(X))
    return y_pred

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    tbl_data=pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    X = tbl_data.drop(columns=['corporation', 'exited']) 
    means=list(X.mean())
    medians=list(X.median())
    std_devs=list(X.std())
    return [{'means': means}, {'medians': medians}, {'std_dev': std_devs}]

##################Function to get the missing values
def dataframe_missing_values():
    tbl_data=pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    X = tbl_data.drop(columns=['corporation', 'exited'])    
    nas=list(X.isna().sum())
    napercents=[nas[i]/len(X.index) for i in range(len(nas))]
    return [{'nas': nas}, {'napercents': napercents}]


##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    def ingestion_timing():
        starttime = timeit.default_timer()
        os.system('python3 ingestion.py')
        timing=timeit.default_timer() - starttime
        return timing
    
    def training_timing():
        starttime = timeit.default_timer()
        os.system('python3 training.py')
        timing=timeit.default_timer() - starttime
        return timing
    
    ingestion_timings=[]
    training_timings=[]

    for idx in range(20):
        ingestion_timings.append(ingestion_timing())
        training_timings.append(training_timing())
        
    final_output=[]
    
    final_output.append(np.mean(ingestion_timings))
    final_output.append(np.std(ingestion_timings))
    final_output.append(np.min(ingestion_timings))
    final_output.append(np.max(ingestion_timings))
    final_output.append(np.mean(training_timings))
    final_output.append(np.std(training_timings))
    final_output.append(np.min(training_timings))
    final_output.append(np.max(training_timings))
    
    return final_output

##################Function to check dependencies
def outdated_packages_list():
    #get a list of
    outdated_packages=subprocess.check_output(['pip', 'list', '--outdated']) #.decode(sys.stdout.encoding)
    with open(os.path.join(diagnostics_path, 'outdated_packages.txt'), 'wb') as f:
        f.write(outdated_packages)
    return outdated_packages

if __name__ == '__main__':
    file_name="testdata.csv"
    model_predictions(file=file_name)
    dataframe_summary()
    dataframe_missing_values()
    execution_time()
    outdated_packages_list()





    
