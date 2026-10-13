#!/usr/bin/env python


import h5py
import numpy as np
from pathlib import Path
import pickle
from scipy.stats import pearsonr

        
# Data directory
data_dir = Path(r"C:\Users\kv5274\Workspace_HP\Projects\AISSAI\artifact-hackathon-2026\challenges\optimisation\data\clear")
    
    # Get first file
h5_files = sorted(data_dir.glob("*.h5"))
    

keys_scalar = ['BCM_GUN', 'DHG0130', 'HUMID1', 'HUMID2', 'HUMID3', 'HUMID4', 'SNH_110', 'SNI_120', 'TEMP1', 'TEMP2', 'TEMP3', 'TEMP4', 'clex_laser_power', 'lab_laser_power', 'pc_time', 'shot']

def get_pearson(x,y, degree):

    coefficients = np.polyfit(x, y, deg=degree)
    y_fit = np.polyval(coefficients, x)
    
    corr, p = pearsonr(y, y_fit)
    
    return corr,p 


degree = 2

coll = {key:[] for key in keys_scalar}
pears = {key:{ } for key in keys_scalar[1:]}


for i_file, file in enumerate(h5_files): 
    
    with h5py.File(file, 'r') as f:
        
    
        for key in keys_scalar :
            data = f[key][:]
            
            coll[key].extend(data.tolist())
            
for i_key, key in enumerate(keys_scalar[1:]):

    
    x = coll['BCM_GUN']
    y = coll[key]
    
    corr,p = get_pearson(x,y, degree)
    pears[key] = {'correlation':corr, 'pearson':p }



# Save to a pickle file
with open("pears.pkl", "wb") as file:  # Note: "wb" for binary mode
    pickle.dump(pears, file)



