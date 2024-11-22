import sys
import numpy as np
import scipy
import subprocess
from natsort import natsorted
import itertools
import pandas as pd
import pickle
sys.path.append("../utls")
sys.path.append("../utls")
sys.path.append("../.")
import os
from preprocessing import *
from InterfaceDeclaration import LPBFInterface
from torch.utils.data.dataset import Subset
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, StandardScaler,LabelEncoder
from models.MLUtls import LPBFDataset

def generate_hist_name(model_name, acoustic_type, context_type,output_type):
    context_combinations = []
    for r in range(len(context_type) + 1): 
        context_combinations.extend(itertools.combinations(context_type, r))

    # Generate all combinations of acoustic_type with context_combinations
    all_combinations = []
    for _output in output_type:
        for model in model_name:
            for acoustic in acoustic_type:
                for context in context_combinations:
                    if len(context)>0:
                        inputs = {acoustic}+{'+'.join(list(context))}
                        all_combinations.append(f"{model}_classification_input_{acoustic}+{'+'.join(list(context))}_output_{_output}")
                    else:
                        inputs = {acoustic}
                        all_combinations.append(f"{model}_classification_input_{acoustic}_output_{_output}")
                    #     all_combinations.append(f"{model}_classification_input_{acoustic}")
    return all_combinations

def find_non_float_index(lst):
    for i, item in enumerate(lst):
        try:
            float(item)
        except TypeError:
            return i
    return -1 

def get_hist_data(hist_dir, model_name,inputs,output_type,folds,max_epochs):
    train_acc = []
    test_acc = []
    for i in range(folds):
        file_path = f"{model_name}_classification_input_{inputs}_output_{output_type}_roi_time10_roi_radius3_fold{i}_of_folds10.csv"
        file_path = os.path.join(hist_dir,file_path)
        df = pd.read_csv(file_path,index_col=0)
        df = df[~df.index.get_level_values(0).duplicated(keep='last')]
        train_acc.append((df['Train Accuracy'][max_epochs]))
        test_acc.append((df['Test Accuracy'][max_epochs]))
    return train_acc,test_acc

def get_hist_data_path(hist_dir, model_name,inputs,output_type,fold=0):
    file_path = f"{model_name}_classification_input_{inputs}_output_{output_type}_roi_time10_roi_radius3_fold{fold}_of_folds10.csv"
    file_path = os.path.join(hist_dir,file_path)
    return file_path

def generate_hist_df(hist_dir,model_name, acoustic_type, context_type,output_type,folds,max_epochs):
    context_combinations = []
    for r in range(len(context_type) + 1): 
        context_combinations.extend(itertools.combinations(context_type, r))
    train_acc_list = []
    test_acc_list = []
    fold_i_list = []
    inputs_list = []
    outputs_list = []
    model_list = []

    for _output in output_type:
        for model in model_name:
            for acoustic in acoustic_type:
                for context in context_combinations:
                    if len(context)>0:
                        inputs = f"{acoustic}+{'+'.join(list(context))}"
                    else:
                        inputs = acoustic

                    train_acc,test_acc = get_hist_data(hist_dir,model,inputs,_output,folds,max_epochs)
                    for i in range(folds):
                        fold_i_list.append(i)
                        train_acc_list.append(train_acc[i])
                        test_acc_list.append(test_acc[i])
                        inputs_list.append(inputs)
                        outputs_list.append(_output)
                        model_list.append(model)

    df = pd.DataFrame()
    df['Model'] = model_list
    df['Train Acc'] = train_acc_list
    df['Test Acc']  = test_acc_list
    df['Fold index'] = fold_i_list
    df['Input type'] = inputs_list
    df['Output type'] = outputs_list

    new_df = pd.concat([df,df])
    new_df['Acc'] = pd.concat([df['Train Acc'], df['Test Acc']])
    new_df['Acc type']= ['Train'] * len(df) + ['Test'] * len(df)
    new_df.index = range(len(new_df))
    return new_df

def get_dataset(roi_time=10, roi_radius=3):
    project_name = ["MuSIC", "MaPS", "MuSIC_EXP1"]
    if os.name == "posix":
        data_dir = subprocess.getoutput("echo $DATADIR")
    elif os.name == "nt":
        data_dir = subprocess.getoutput("echo %datadir%")
    music_dir = os.path.join(data_dir, "MuSIC")
    if not os.path.exists(music_dir):
        project_name[0] = "2024-MUSIC"
    daq_dir = os.path.join(data_dir, *project_name, "Acoustic Monitoring")
    lmq_dir = os.path.join(data_dir, *project_name, "LMQ Monitoring")
    del music_dir

    with open(os.path.join(os.path.dirname(daq_dir),'intermediate',f"lpbf_line_wise_data.pkl"), 'rb') as handle:
        lpbf_data = pickle.load(handle)

    sc_power = StandardScaler().fit(np.unique(lpbf_data.laser_power).astype(float).reshape(-1,1))
    # sc_direction = StandardScaler().fit(np.unique(lpbf_data.print_vector[1]).astype(float).reshape(-1,1))
    le_direction = LabelEncoder().fit(np.unique(np.asarray(np.round(lpbf_data.print_vector[1]),dtype=str)))
    le_speed = LabelEncoder().fit(np.asarray(lpbf_data.scanning_speed,dtype=str))
    le_region = LabelEncoder().fit(np.asarray(lpbf_data.regime_info,dtype=str))

    laser_power = sc_power.transform(np.asarray(lpbf_data.laser_power).astype(float).reshape(-1,1)).reshape(-1)
    # print_direction = sc_direction.transform(np.asarray(lpbf_data.print_vector[1]).astype(float).reshape(-1,1)).reshape(-1)
    print_direction = le_direction.transform(np.asarray(np.round(lpbf_data.print_vector[1]),dtype=str)).astype(int)
    scanning_speed = le_speed.transform(np.asarray(lpbf_data.scanning_speed).astype(float))
    regime_info = le_region.transform(np.asarray(lpbf_data.regime_info,dtype=str))

    dataset = LPBFDataset(lpbf_data.cube_position,laser_power,lpbf_data.scanning_speed,regime_info,print_direction,lpbf_data.microphone, lpbf_data.AE, lpbf_data.defect_labels)
    return dataset,sc_power,le_direction,le_speed,le_region
