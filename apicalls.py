import os
import json
import subprocess

#combine all API responses
def get_api_responses(URL, file_name):
    #Call each API endpoint and store the responses
    response1=subprocess.run(['curl', f'{URL}/prediction?filename={file_name}'],capture_output=True).stdout
    response2=subprocess.run(['curl', f'{URL}/scoring'],capture_output=True).stdout
    response3=subprocess.run(['curl', f'{URL}/summarystats'],capture_output=True).stdout
    response4=subprocess.run(['curl', f'{URL}/diagnostics'],capture_output=True).stdout
    return {
        'response1': response1, 
        'response2': response2, 
        'response3': response3, 
        'response4': response4
    }
#write the responses to your workspace
def write_api_responses(api_responses, path, file_out_name):
    with open(os.path.join(path, file_out_name), 'w') as f:
        f.write(str(api_responses))

if __name__ == "__main__":
    print("Testing: apicalls.py ...")
    
    with open('config.json','r') as f:
        config = json.load(f) 
    model_path=config['output_model_path'] 
    #Specify a URL that resolves to your workspace
    URL = "http://127.0.0.1:8000"
    
    api_responses=get_api_responses(URL, file_name='testdata.csv')
    write_api_responses(api_responses, model_path, file_out_name='apireturns.txt')
    print("Success")