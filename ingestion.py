import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 
    
input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
    
#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    files=os.listdir(input_folder_path)
    files.sort()
    tbl_union=pd.DataFrame()
    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'),'a') as f:
        for file in files:
            df=pd.read_csv(os.path.join(input_folder_path, file))
            tbl_union=tbl_union.append(df, ignore_index=True)
            f.write("\n")
            f.write(file)
        
    tbl_union=tbl_union.drop_duplicates()
    tbl_union.to_csv(os.path.join(output_folder_path, "finaldata.csv"), index=False)


if __name__ == '__main__':
    merge_multiple_dataframe()
