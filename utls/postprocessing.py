import sys
import numpy as np
import scipy
import subprocess
from natsort import natsorted
import itertools
import pandas as pd
sys.path.append("../utls")
sys.path.append("../utls")
sys.path.append("../.")
import os
from preprocessing import *
from InterfaceDeclaration import LPBFInterface

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

