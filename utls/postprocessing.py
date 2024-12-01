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
from models.MLModels import SVMModel, CNN_Base_1D_Model, ResNet15_1D_Model

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

def make_model(model_name, input_type, output_type, time_series_length=5888):
    meta_data_size = len(input_type.split("+"))-1
    # time_series_length = 5888
    # print(f"Input type: {input_type}")
    # print(f"Output type: {output_type}")
    # print(f"Lenght of time series data: {time_series_length}")
    # print(f"Lenght of context data: {meta_data_size}")
    # print(f"Model name: {model_name}")

    num_classes = 1
    if output_type == "regime":
        num_classes = 4
    if output_type == "defect":
        num_classes = 2
    if output_type == "direction":
        num_classes = 5
    if output_type == "position":
        num_classes = 5

    if model_name == "SVM":
        model = SVMModel(time_series_length+meta_data_size,num_classes=num_classes).double()
    if model_name == "CNN":
        model = CNN_Base_1D_Model(time_series_length=time_series_length, meta_data_size=meta_data_size,num_classes=num_classes).double()
    if model_name == "Res15":
        model = ResNet15_1D_Model(time_series_length=time_series_length, meta_data_size=meta_data_size,num_classes=num_classes).double()
    return model

def read_trained_model(snap_dir,model_name, acoustic_type, context_type,output_type,folds,max_epochs):
    snap_list = []
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
        for _model in model_name:
            for acoustic in acoustic_type:
                for context in context_combinations:
                    if len(context)>0:
                        inputs = f"{acoustic}+{'+'.join(list(context))}"
                    else:
                        inputs = acoustic
                    
                    for fold in range(folds):
                        file_path = f"{_model}_classification_input_{inputs}_output_{_output}_roi_time10_roi_radius3_fold{fold}_of_folds10.pt"
                        file_path = os.path.join(snap_dir,file_path)
                        model = make_model(_model, inputs, _output)
                        snap_list.append(file_path)
    return model,snap_list

def get_confution_matrix(model, snap_dir, _model_name,_inputs,_outputs,class_num=5,folds_num = 10):
    import torch
    from sklearn.metrics import confusion_matrix
    _model_name= "CNN"
    _inputs = "ae"
    _outputs = "direction"
    meta_list = []
    pred_list = []
    label_list = [] 
    cf = np.zeros((folds_num,class_num,class_num),dtype=np.int64)
    for f_i in range(folds_num):
        snap_name = f"{_model_name}_classification_input_{_inputs}_output_{_outputs}_roi_time10_roi_radius3_fold{f_i}_of_folds10.pt"
        snap_name = os.path.join(snap_dir,snap_name)
        snapshot = torch.load(snap_name, map_location=f"cuda:0", weights_only=True)
        _state_dict = snapshot["model_state_dict"]
        model.load_state_dict(_state_dict)
        model = model.to('cuda')
        # _cube_position, _laser_power, _scanning_speed, _regime_info, _print_direction, _mic, _ae, _defect_labels = next(iter(data_loader))
        with torch.no_grad():
            for _cube_position, _laser_power, _scanning_speed, _regime_info, _print_direction, _mic, _ae, _defect_labels in data_loader:
                time_series = (transform_ft()(standardize_tensor(_ae))).double()
                meta_list.append(_laser_power.double())
                logits = model(time_series.to('cuda'),meta_list)
                probs = torch.sigmoid(logits)
                preds = torch.argmax(probs,axis=1).clone().int().detach().cpu()
                pred_list.extend(preds.cpu().numpy())
                label_list.extend(_print_direction.cpu().numpy())
        cf[f_i,:,:] = confusion_matrix(label_list,pred_list)
    return cf
