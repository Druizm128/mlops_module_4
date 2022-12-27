import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path']) 



##############Function for reporting
def score_model_report(file, fig_name):
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    tbl_data=pd.read_csv(os.path.join(test_data_path, file))
    y_true=list(tbl_data['exited'])
    y_pred=model_predictions(file)
    cf_matrix=metrics.confusion_matrix(y_true, y_pred)
    fig, ax=plt.subplots()
    sns.heatmap(cf_matrix, annot=True, cmap='YlOrRd')
    plt.savefig(
        os.path.join(model_path, fig_name), 
        format="png"
    )




if __name__ == '__main__':
    file_name="testdata.csv"
    fig_name="confusion_matrix.png"
    score_model(file_name, fig_name)
