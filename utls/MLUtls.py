import numpy as np
import librosa
import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
import torchvision
import torchvision.transforms
from scipy.interpolate import griddata
from tqdm import tqdm
global device
global device_ids



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.cuda.device_count() > 1:
#     # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
#     num_of_gpus = torch.cuda.device_count()
#     print(f"number of gpus:{num_of_gpus}")
#     device = torch.device(f"cuda:3")

def one_hot_encode(y, num_classes):
    """
    Convert labels to one-hot encoding.
    Args:
        y (torch.Tensor): Labels with shape (batch_size,)
        num_classes (int): Number of classes
    Returns:
        torch.Tensor: One-hot encoded labels with shape (batch_size, num_classes)
    """
    y_onehot = torch.zeros((y.size(0), num_classes), device=y.device)
    y_onehot.scatter_(1, y.unsqueeze(1), 1)
    return y_onehot

def standardize_array(arr):
    """
    Standardize a NumPy array.

    Parameters:
        arr (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Standardized array.
    """
    mean = np.mean(arr)
    std = np.std(arr)
    standardized_arr = (arr - mean) / std
    return standardized_arr

def standardize_tensor(arr):
    """
    Standardize a torch tensor.

    Parameters:
        arr (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Standardized array.
    """
    # arr = torch.tensor(arr)
    mean = torch.mean(arr)
    std = torch.std(arr)
    standardized_arr = (arr - mean) / std
    return standardized_arr

def split_dataset( dataset_size, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=910420):
    """
    Splits a dataset into training, validation, and test sets using NumPy.

    Args:
        dataset_size: The total size of the dataset.
        train_ratio: Ratio of the dataset to be used for training (default: 0.7).
        val_ratio: Ratio of the dataset to be used for validation (default: 0.15).
        test_ratio: Ratio of the dataset to be used for testing (default: 0.15).

        seed: Random seed for reproducibility (default: 910420).

    Returns:
        train_idx: A list of indices for the training set.
        valid_idx: A list of indices for the validation set.
        test_idx: A list of indices for the test set.
    """

    np.random.seed(seed)  # Set the random seed for reproducibility

    # Calculate the number of samples in each set
    num_train = int(dataset_size * train_ratio)
    num_val = int(dataset_size * val_ratio)
    num_test = int(dataset_size * test_ratio)

    # Ensure all indices are used and no overlap occurs (adjust ratios if needed)
    # assert (
    #     num_train + num_val + num_test == dataset_size
    # ), "Ratios must sum to 1. Adjust to avoid exceeding dataset size."

    # Permute all indices for random shuffling
    all_indices = np.arange(dataset_size,dtype=int)
    np.random.shuffle(all_indices)

    # Split indices into training, validation, and test sets
    train_idx = all_indices[:num_train]
    valid_idx = all_indices[num_train : num_train + num_val]
    test_idx = all_indices[num_train + num_val :]

    return list(train_idx), list(valid_idx), list(test_idx)

def train_model(model, dataloader, optimizer, test_loader=None, transform=None, num_epochs=10, save_model=None,log_path=None):
    # Define Losses and Optimizer
    criterion_reg = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()

    model.to(device)
    for epoch in range(num_epochs):
        print("Start training...")
        model.train()
        train_loss = 0.0
        for i, data in tqdm(enumerate(dataloader)):
            data_mic, label_power, label_powder, label_carrier, label_shielding, label_defect, label_exhausted = data
            regression_targets = torch.stack((label_power, label_powder, label_carrier, label_shielding),dim=1).to(device,dtype=torch.float)
            inputs = transform(data_mic.numpy()).to(device)

            optimizer.zero_grad()

            regression_output, class_output1, class_output2 = model(inputs)

            # One-hot encode classification targets
            class_target1 = one_hot_encode(label_defect.to(device, dtype=torch.int64), 3)  # Defect type
            class_target2 = one_hot_encode(label_exhausted.to(device, dtype=torch.int64), 2)  # Exhaust on off, binary classification

            # Compute loss
            loss_reg = criterion_reg(regression_output, regression_targets)
            loss_class1 = criterion_class(class_output1, class_target1)
            loss_class2 = criterion_class(class_output2, class_target2)

            loss = loss_reg*1e4 + loss_class1 + loss_class2

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Print statistics
            train_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {train_loss / 10:.3f}")
                train_loss = 0.0
        if save_model:
            if (epoch) % 10 == 0:
                torch.save(model.state_dict(), f"./lfs/weights/{save_model}.pt") 

        if test_loader:
            # Evaluate on test set
            test_loss = evaluate_model(model, test_loader, criterion_reg, criterion_class,transform=transform)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Test Loss: {test_loss:.3f}")
        if log_path:
            train_log(log_path,epoch=epoch,train_loss=train_loss,test_loss=test_loss)
    print("Finished Training")

def evaluate_model(model, dataloader, criterion_reg, criterion_class,transform=None):
    """
    Evaluates the model on the provided dataset.
    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the test set.
        criterion_reg (nn.Module): Loss function for regression.
        criterion_class (nn.Module): Loss function for classification.
    Returns:
        float: Average loss over the test set.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    with torch.no_grad():
        for data in dataloader:

            data_mic, label_power, label_powder, label_carrier, label_shielding, label_defect, label_exhausted = data
            regression_targets = torch.stack((label_power, label_powder, label_carrier, label_shielding),dim=1).to(device,dtype=torch.float)
            inputs = transform(data_mic.numpy()).to(device)
            
            # Forward pass
            regression_output, class_output1, class_output2 = model(inputs)
            
            # One-hot encode classification targets
            class_target1 = one_hot_encode(label_defect.to(device, dtype=torch.int64), 3)  # Defect type
            class_target2 = one_hot_encode(label_exhausted.to(device, dtype=torch.int64), 2)  # Exhaust on off, binary classification
            
            # Compute loss
            loss_reg = criterion_reg(regression_output, regression_targets)
            loss_class1 = criterion_class(class_output1, class_target1)
            loss_class2 = criterion_class(class_output2, class_target2)
            
            loss = loss_reg*1e4 + loss_class1 + loss_class2
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def train_log(save_path,epoch,train_loss,train_acc, test_loss, test_acc):
    # Check if the file exists to determine whether to write headers
    file_exists = os.path.isfile(save_path)
    # Open the CSV file in append mode
    with open(save_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write headers if the file is new
        if not file_exists:
            writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy'])
        writer.writerow([epoch + 1, float(train_loss), float(train_acc), float(test_loss), float(test_acc)])

def getKFoldCrossValidationIndexes(n, k, seed=1):
    """
    Perform k-fold cross-validation.
    Returns:
        List of tuples: Each tuple contains the indices of the training and testing sets.
    """
    fold_size = n // k
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)
    folds = []
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i < k - 1 else n
        test_indices = indices[start:end]
        train_indices = np.concatenate((indices[:start], indices[end:]))
        folds.append((train_indices, test_indices))
    return folds

def logSpectrogram(
    sampling_rate,
    data=None,
    n_fft=1024,
    hop_length=320,
    window_type="hann",
    fmin=0,
    fmax=None,
    display=False,
    save=False,
    ):
    """
    n_fft=1024 # Size of the Fast Fourier Transform (FFT), which will also be used as the window length

    hop_length=320 # Step or stride between windows. If the step is smaller than the window length, the windows will overlap
    window_type ='hann' # Specify the window type for FFT/STFT
    """
    y = data
    # Check for non-finite values
    y = standardize_array(y)
    y = np.nan_to_num(y)
    # Check for non-finite values

    # Calculate the spectrogram as the square of the complex magnitude of the STFT
    spectrogram = (
        np.abs(
            librosa.stft(
                y,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window_type,
            )
        )
        ** 2
    )

    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # display
    return spectrogram

def load_train_data(data_dir):
    # load the dataset
    with open(os.path.join('intermediate',f"indices"), "rb") as fp:
        indices = pickle.load(fp)
    with open(os.path.join('intermediate',f"power_labels"), "rb") as fp:
        power_labels = pickle.load(fp)
    with open(os.path.join('intermediate',f"powder_labels"), "rb") as fp:
        powder_labels = pickle.load(fp)
    with open(os.path.join('intermediate',f"carrier_labels"), "rb") as fp:
        carrier_labels = pickle.load(fp)
    with open(os.path.join('intermediate',f"speed_labels"), "rb") as fp:
        speed_labels = pickle.load(fp)
    with open(os.path.join('intermediate',f"shielding_labels"), "rb") as fp:
        shielding_labels = pickle.load(fp)
    with open(os.path.join('intermediate',f"exhaust_labels"), "rb") as fp:
        exhaust_labels = pickle.load(fp)
    with open(os.path.join('intermediate',f"defect_labels"), "rb") as fp:
        defect_labels = pickle.load(fp)
    file_name_list = ['test2','test3','test4','test5','test6','test7','test1','carrier1','carrier2','carrier3','gas','gas2','exhaust','idle']
    dataset = LCVDataset(power_labels,powder_labels,carrier_labels,shielding_labels,defect_labels,exhaust_labels,indices,file_name_list,data_dir)
    return dataset

def load_model(model_name, model):
    checkpoint_path = f"./lfs/weights/{model_name}.pt"
    snapshot = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(snapshot["model_state_dict"])
    # optimizer.load_state_dict(snapshot['optimizer_state_dict'])
    epochs_run = snapshot["EPOCHS_RUN"]
    print(f"Loading model from {checkpoint_path}")
    return model

def load_model_epochs(checkpoint_name,model):
    loc = f"cuda"
    checkpoint_path = f"./lfs/weights/{checkpoint_name}.pt"
    snapshot = torch.load(checkpoint_path, map_location=loc, weights_only=True)
    epochs_run = snapshot["EPOCHS_RUN"]
    print(f"Resuming training from snapshot at Epoch {epochs_run}")
    return epochs_run

def fade_in_out(data, fade_length):
    # Convert to numpy array for processing
    # Apply fade-in
    audio = data.clone()
    fade_in = torch.linspace(0, 1, fade_length)
    audio[:fade_length] *= fade_in
    # Apply fade-out
    fade_out = torch.linspace(1, 0, fade_length)
    audio[-fade_length:] *= fade_out
    return audio

def transform_pad(maximum_size,fad_in_out_length=16):
    return torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(
                        lambda data:
                        torch.nn.functional.pad(data, (0, maximum_size - data.shape[-1])),
                    ),
                    torchvision.transforms.Lambda(
                        lambda data:
                        fade_in_out(data,fad_in_out_length),
                    ),
                ])

def transform_ft():
    return torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(
                        lambda data:
                        torch.abs(torch.fft.fft(data))
                    ),
                    torchvision.transforms.Lambda(
                        lambda data:
                        data[:,:data.shape[-1]//2]
                    ),
                ]
            )

def transform_spec():
    return torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(
                        lambda data:
                        torch.nn.functional.pad(torch.tensor(data), (0, mic_maximum_size - data.shape[-1])),
                    ),

                    torchvision.transforms.Lambda(
                        lambda data:
                        fade_in_out(data,16),
                    ),
                    torchvision.transforms.Lambda(
                        lambda data:
                        torch.stft(data)
                    ),
                ]
            )

class CylinderDataset(Dataset):
    def __init__(
        self,
        dataDir,
        labelArray,
        roi_time,
        on_pad = True,
        maximum_size=3840,
        fade_in_out_length = 16,
        
        transformation1 = None,
        transformation2 = None,
        transformation3 = None,
    ):
        """
        For LCV dataset
        - Inputs:
            Pre-constructed labels
                - layer index
                - start index of segmentation
                - laser power
                - moving speed
                - direction
                - if keyhole defect 
                - if lack of fusion defect

            DataDir

        - Outputs:
            - Regression:
                - Laser power
                - Powder feed rate
                - Carrier
                - Shielding

            - Classification:
                LoF defect (Binary)
                Keyhole defect (Binary)
        Args:
            microphone_data (ndarray): Microphone data inputs.
            laser_power (ndarray): Laser power values (regression).
            powder_feed_rate (ndarray): Powder feed rate values (regression).
            carrier (ndarray): Carrier values (regression).
            shielding (ndarray): Shielding values (regression).
            defect (ndarray): Defect classification labels (3 classes).
            exhausted (ndarray): Exhausted classification labels (binary).
        """
        self.dataDir = dataDir
        self.labels = labelArray
        self.roi_time = roi_time
        self.on_pad = on_pad
        self.maximum_size = maximum_size
        self.fade_in_out_length = fade_in_out_length
        
        self.transformation1 = transformation1
        self.transformation2 = transformation2
        self.transformation3 = transformation3

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        label_i = self.labels[idx,:]
        layer_i = int(label_i[0])
        i_start = int(label_i[1])
        power=label_i[2]
        velocity=label_i[3]
        direction=label_i[4]
        if_defect_KH=label_i[5]
        if_defect_LoF=label_i[6]
        mic_path = os.path.join(self.dataDir,'intermediate',f"roi{self.roi_time}ms_layer{layer_i}_{i_start}.npy")
        # np.load(os.path.join(data_dir))
        mic = np.load(mic_path)
        if self.on_pad == True:
            t = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(
                        lambda data:
                        torch.nn.functional.pad(torch.tensor(data), (0, self.maximum_size - data.shape[-1])),
                    ),
                    torchvision.transforms.Lambda(
                        lambda data:
                        fade_in_out(data,self.fade_in_out_length),
                    ),
                ])
            mic = t(mic)
        
        if self.transformation1 != None:
            mic = self.transformation1(mic)
        if self.transformation2 != None:
            mic = self.transformation2(mic)
        if self.transformation3 != None:
            mic = self.transformation3(mic)

        return (
            mic,
            power,
            velocity,
            direction,
            if_defect_LoF,
            if_defect_KH,
        )

class CylinderDataset_RAM(Dataset):
    def __init__(
        self,
        dataDir,
        labelArray,
        roi_time,
        mic_list,
        total_labels,
        on_pad = True,
        maximum_size=3840,
        fade_in_out_length = 16,
        
        transformation1 = None,
        transformation2 = None,
        transformation3 = None,
    ):
        """
        For LCV dataset
        - Inputs:
            Pre-constructed labels
                - layer index
                - start index of segmentation
                - laser power
                - moving speed
                - direction
                - if keyhole defect 
                - if lack of fusion defect

            DataDir

        - Outputs:
            - Regression:
                - Laser power
                - Powder feed rate
                - Carrier
                - Shielding

            - Classification:
                LoF defect (Binary)
                Keyhole defect (Binary)
        Args:
            microphone_data (ndarray): Microphone data inputs.
            laser_power (ndarray): Laser power values (regression).
            powder_feed_rate (ndarray): Powder feed rate values (regression).
            carrier (ndarray): Carrier values (regression).
            shielding (ndarray): Shielding values (regression).
            defect (ndarray): Defect classification labels (3 classes).
            exhausted (ndarray): Exhausted classification labels (binary).
        """
        self.dataDir = dataDir
        self.labels = labelArray
        self.roi_time = roi_time
        self.on_pad = on_pad
        self.maximum_size = maximum_size
        self.fade_in_out_length = fade_in_out_length
        self.total_labels = total_labels
        
        self.transformation1 = transformation1
        self.transformation2 = transformation2
        self.transformation3 = transformation3

        self.mic_list =mic_list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        label_i = self.labels[idx,:]
        layer_i = int(label_i[0])
        i_start = int(label_i[1])
        power=label_i[2]
        velocity=label_i[3]
        direction=label_i[4]
        if_defect_KH=label_i[5]
        if_defect_LoF=label_i[6]
        # np.load(os.path.join(data_dir))
        mic_idx = np.where((self.total_labels[:,0]==layer_i) & (self.total_labels[:,1]==i_start))[0][0]
        mic = self.mic_list[mic_idx]
        if self.on_pad == True:
            t = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(
                        lambda data:
                        torch.nn.functional.pad(torch.tensor(data), (0, self.maximum_size - data.shape[-1])),
                    ),
                    torchvision.transforms.Lambda(
                        lambda data:
                        fade_in_out(data,self.fade_in_out_length),
                    ),
                ])
            mic = t(mic)
        
        if self.transformation1 != None:
            mic = self.transformation1(mic)
        if self.transformation2 != None:
            mic = self.transformation2(mic)
        if self.transformation3 != None:
            mic = self.transformation3(mic)

        return (
            mic,
            power,
            velocity,
            direction,
            if_defect_LoF,
            if_defect_KH,
        )

class LCVDataset(Dataset):
    def __init__(
        self,
        laser_power,
        powder_feed_rate,
        carrier,
        shielding,
        defect,
        exhausted,
        indices,
        file_name_list,
        data_dir,
    ):
        """
        For LCV dataset
        - Inputs:
            Microphone data

        - Outputs:
            - Regression:
                - Laser power
                - Powder feed rate
                - Carrier
                - Shielding

            - Classification:
                Defect (3-classes)
                Exhausted (Binary)
        Args:
            microphone_data (ndarray): Microphone data inputs.
            laser_power (ndarray): Laser power values (regression).
            powder_feed_rate (ndarray): Powder feed rate values (regression).
            carrier (ndarray): Carrier values (regression).
            shielding (ndarray): Shielding values (regression).
            defect (ndarray): Defect classification labels (3 classes).
            exhausted (ndarray): Exhausted classification labels (binary).
        """
        self.laser_power = laser_power
        self.powder_feed_rate = powder_feed_rate
        self.carrier = carrier
        self.shielding = shielding
        self.defect = defect
        self.exhausted = exhausted
        self.data_dir = data_dir
        self.file_name_list = file_name_list
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        id_file, id_start_i = self.indices[idx]
        _mic = np.load(
            os.path.join(
                self.data_dir,
                "intermediate",
                f"{self.file_name_list[id_file]}-{id_start_i}.npy",
            )
        )

        return (
            _mic,
            self.laser_power[idx],
            self.powder_feed_rate[idx],
            self.carrier[idx],
            self.shielding[idx],
            self.defect[idx],
            self.exhausted[idx],
        )

def dataset_by_cross_validation(roi_time, train_labels, test_labels, total_labels, ram=False):
    import subprocess
    import pickle
    project_name = ["MuSIC", "FlandersMake","Cylinder14"]

    if os.name == "posix":
        data_dir = subprocess.getoutput("echo $DATADIR")
    elif os.name == "nt":
        data_dir = subprocess.getoutput("echo %datadir%")
    music_dir = os.path.join(data_dir, "MuSIC")
    if not os.path.exists(music_dir):
        project_name[0] = "2024-MUSIC"
    data_dir = os.path.join(data_dir, *project_name)
    del music_dir

    labels = np.vstack([train_labels,test_labels])
    train_idx = np.arange(0,len(train_labels),1,dtype=int)
    test_idx = np.arange(len(train_labels),len(train_labels)+len(test_labels),1,dtype=int)
    if ram == False:
        dataset = CylinderDataset(dataDir=data_dir,labelArray=labels,on_pad =True,  roi_time=roi_time)
    else:
        with open(os.path.join(data_dir,'intermediate',f"roi{roi_time}ms_mic_list"), 'rb') as handle:
            mic_list = pickle.load(handle)
        dataset = CylinderDataset_RAM(dataDir=data_dir,labelArray=labels,on_pad =True, mic_list=mic_list, total_labels=total_labels, roi_time=roi_time)
        
    return dataset, train_idx, test_idx

def labels_by_classes(roi_time, roi_radius):
    import subprocess
    import pickle
    from sklearn.preprocessing import StandardScaler,MinMaxScaler
    project_name = ["MuSIC", "FlandersMake","Cylinder14"]

    if os.name == "posix":
        data_dir = subprocess.getoutput("echo $DATADIR")
    elif os.name == "nt":
        data_dir = subprocess.getoutput("echo %datadir%")
    music_dir = os.path.join(data_dir, "MuSIC")
    if not os.path.exists(music_dir):
        project_name[0] = "2024-MUSIC"
    data_dir = os.path.join(data_dir, *project_name)
    del music_dir

    start_layer = 33
    end_layer = 481
    shell_exception_layers = [114, 208,423,476]
    errlayers = [60 , 61 , 62 , 90 , 120, 150, 180, 210, 240, 270, 300, 330 , 360 , 390, 420, 450 , 470, 471, 472]
    normal_layers = np.arange(end_layer+1)

    _mask = np.ones(len(normal_layers), dtype=bool)
    _mask[shell_exception_layers] = False
    _mask[:start_layer] = False

    _mask = np.ones(len(normal_layers), dtype=bool)
    _mask[errlayers] = False
    _mask[:start_layer] = False
    normal_layers = normal_layers[_mask,...]
    normal_labels = []
    for i in normal_layers:
        with open(os.path.join('lfs/labels',f"roi_{roi_time}ms",f'window_results_layer_{i}'), 'rb') as handle:
            _label = pickle.load(handle)
        normal_labels+=_label
    normal_labels = np.array(normal_labels)

    err_labels = []
    for i in errlayers:
        with open(os.path.join('lfs/labels',f"roi_{roi_time}ms",f'window_results_layer_{i}_roi_radius{roi_radius}'), 'rb') as handle:
            _label = pickle.load(handle)
        err_labels+=_label
    err_labels = np.array(err_labels)
    labels = np.vstack([normal_labels,err_labels])

    p_set = np.unique(labels[:,2]).reshape(-1, 1)
    v_set = np.unique(labels[:,3]).reshape(-1, 1)

    scaler_power = StandardScaler()
    laser_power_standardized = scaler_power.fit_transform(p_set)
    scaler_speed = StandardScaler()
    laser_speed_standardized = scaler_speed.fit_transform(v_set)
    scaler_direction = MinMaxScaler(feature_range=(0, 1))
    _ = scaler_direction.fit_transform([[-180],[180]])
    
    err_labels[:,2] =     scaler_power.transform(err_labels[:,2].reshape(-1,1)).squeeze()
    err_labels[:,3] =     scaler_speed.transform(err_labels[:,3].reshape(-1,1)).squeeze()
    err_labels[:,4] = scaler_direction.transform(err_labels[:,4].reshape(-1,1)).squeeze()

    normal_labels[:,2] =     scaler_power.transform(normal_labels[:,2].reshape(-1,1)).squeeze()
    normal_labels[:,3] =     scaler_speed.transform(normal_labels[:,3].reshape(-1,1)).squeeze()
    normal_labels[:,4] = scaler_direction.transform(normal_labels[:,4].reshape(-1,1)).squeeze()
    return normal_labels, err_labels, scaler_power, scaler_speed, scaler_direction

def textFinder(text,start,end):
    start = text.find(start) + len(start)
    end = text.find(end)
    result = text[start:end]
    return result

def get_current_fold_and_hist(model_name,input_type,output_type,folds,rou_time, roi_radius,max_epochs):
    from MLModels import SVMModel, CNN_Base_1D_Model, ResNet15_1D_Model
    from natsort import natsorted
    import pandas as pd
    checkpoint_name0 = f'{model_name}_classification_input_{input_type}_output_{output_type}_roi_radius{roi_radius}_fold'
    # trained_weights =natsorted([i.split(".")[0] for i in os.listdir(os.path.join(os.getcwd(),"lfs","weights")) if i.split(".")[-1]=='pt' and i.split("_")[0]==model_name and i.split("_")[-1]==f'folds{folds}.pt'])
    # trained_weights_hist =natsorted([i for i in os.listdir(os.path.join(os.getcwd(),"lfs","weights","hist")) if i.split(".")[-1]=='csv' and i.split("_")[0]==model_name and i.split("_")[-1]==f'folds{folds}.csv'])
    trained_weights =natsorted([i.split(".")[0] for i in os.listdir(os.path.join(os.getcwd(),"lfs","weights")) if i.split(".")[-1]=='pt' and checkpoint_name0 in i])
    trained_weights_hist =natsorted([i.split(".")[0] for i in os.listdir(os.path.join(os.getcwd(),"lfs","weights","hist")) if i.split(".")[-1]=='csv' and checkpoint_name0 in i])

    if len(trained_weights) >= 1:
        print(f"Resuming from {trained_weights[-1]}")
        trained_path = os.path.join(os.getcwd(),"lfs","weights",trained_weights[-1])
        # trained_weights,trained_weights_hist

        if model_name == "SVM":
            # if rank ==0:
            print("Using SVM")
            trained_model = SVMModel(1920+4).double()
        if model_name == "CNN":
            # if rank ==0:
            print("Using CNN")
            trained_model = CNN_Base_1D_Model(time_series_length=1920, meta_data_size=4).double()
        if model_name == "Res15":
            # if rank ==0:
            print("Using Res15")
            trained_model = ResNet15_1D_Model(time_series_length=1920, meta_data_size=4).double()

        trained_epochs = load_model_epochs(trained_weights[-1],trained_model)
        trained_hist = pd.read_csv(os.path.join(os.getcwd(),"lfs","weights","hist",trained_weights_hist[-1]+".csv"),index_col=0)
        train_hist_drop = trained_hist.drop(trained_hist[trained_hist.index>trained_epochs].index)
        train_hist_drop.to_csv(os.path.join(os.getcwd(),"lfs","weights","hist",trained_weights_hist[-1]+".csv"))
        # current_fold = get_numbers_from_string(trained_weights[-1])[-2]
        current_fold = int(textFinder(trained_weights[-1],"_fold","_of_folds"))
        print(f"Roll back the hist .csv file from trained epochs: [{trained_epochs}] from fold [{current_fold+1}/{folds}]")
        print(f"====================\n")
    
        return int(current_fold), int(trained_epochs)
    else:
        print(f"{folds}-Folds Cross Validation for {model_name}")
        print(f"====================\n")
        return 0,0

def get_numbers_from_string(s):
    import re
    # Find all numbers in the string using regular expression
    numbers = re.findall(r'\d+', s)
    return numbers

def get_max_length(array_list):
    """
    Calculates the maximum length among all NumPy arrays in a given list.

    Args:
        array_list: A list of NumPy arrays.

    Returns:
        The maximum length found among the arrays.
    """

    if not array_list:
        return 0  # Handle empty list case

    max_length = max(len(array) for array in array_list)
    return max_length

def get_windowed_data(data, index, window_size=101, fade_length=16):
    """
    Gets a window of data around a given index, with zero padding and fade in/out.

    Args:
    data: A 1D numpy array or torch tensor.
    index: The index of the data point to center the window around.
    window_size: The desired size of the window.
    fade_length: The length of the fade in/out regions.

    Returns:
    A torch tensor of size (window_size,) containing the windowed data.
    """

    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)

    # Calculate start and end indices
    start_index = max(0, index - window_size // 2)
    end_index = min(len(data), start_index + window_size)

    # Calculate padding lengths
    left_pad = max(0, start_index - index + window_size // 2)
    right_pad = max(0, window_size - (end_index - start_index))

    # Create windowed data with padding
    windowed_data = torch.cat([
    torch.zeros(left_pad),
    data[start_index:end_index],
    torch.zeros(right_pad)
    ])

    # Apply fade in/out
    fade_in = torch.linspace(0, 1, fade_length)
    fade_out = torch.linspace(1, 0, fade_length)
    windowed_data[:fade_length] *= fade_in
    windowed_data[-fade_length:] *= fade_out

    return windowed_data[:window_size]

def augment_abnormal_data(data,gpu_id,ratio=4):
    """Augments abnormal data by random cropping and stretching.

    Args:
        data: The input data tensor.
        target_length_ratio: The target length ratio for cropping.
        stretch_compress_ratio: The stretch/compress ratio.

    Returns:
        A list of augmented data tensors.
    """
    
    target_length_ratio = torch.abs(0.9 + (torch.rand(1) * (0.95 - 0.9)))
    augmented_data = []
    stretch_compress_ratio = torch.abs(0.98 + (torch.rand(1) * (1.02 - 0.98)))

    for _ in range(ratio):
        # Randomly crop the data
        crop_start = int(torch.rand(1) * (data.shape[1] - data.shape[1] * target_length_ratio))
        crop_end = crop_start + int(data.shape[1] * target_length_ratio)
        cropped_data = data[:, crop_start:crop_end]

        # Stretch or compress the data
        stretched_data = torch.nn.functional.interpolate(cropped_data.unsqueeze(0), size=data.shape[1], mode='linear', align_corners=False).squeeze(0)
        stretched_data *= stretch_compress_ratio
        augmented_data.append(stretched_data)
    return augmented_data

def augment_dataset(label, time_series, meta_list,gpu_id, ratio=4):
    idx = torch.where(label==1)[0].to('cpu')
    data = time_series[idx]
    _m = augment_abnormal_data(data,gpu_id,ratio)
    augmented_data = torch.vstack([torch.vstack(_m),time_series])

    _aug_meta_list = []
    if len(meta_list)>0:
        for _meta in meta_list:
            _m = torch.vstack(list(_meta[idx])*ratio).squeeze()
            _aug_meta_list.append(torch.cat([_m,_meta]))
    _l = torch.cat([torch.vstack(list(label[idx])*ratio).squeeze(),label]) 
    return augmented_data, _aug_meta_list,_l

class LPBFDataset(Dataset):
    def __init__(
        self,
        cube_position,
        laser_power,
        scanning_speed,
        regime_info,
        print_direction,
        microphone,
        ae,
        defect_labels,
        fad_in_out_length = 16
    ):
        self.cube_position = cube_position
        self.laser_power = laser_power
        self.scanning_speed = scanning_speed
        self.regime_info = regime_info
        self.print_direction = print_direction
        self.microphone = microphone
        self.ae = ae
        self.defect_labels = defect_labels
        self.max_length = get_max_length(self.microphone)
        self.fad_in_out_length = fad_in_out_length

    def __len__(self):
        return len(self.cube_position)

    def __getitem__(self, idx):
        mic = self.microphone[idx]
        ae = self.ae[idx]

        mic = torch.nn.functional.pad(torch.tensor(mic), (0, self.max_length - mic.shape[-1]))
        ae = torch.nn.functional.pad(torch.tensor(ae), (0, self.max_length - ae.shape[-1]))
        mic = fade_in_out(mic,self.fad_in_out_length)
        ae = fade_in_out(ae,self.fad_in_out_length)

        return (
            self.cube_position[idx],
            self.laser_power[idx],
            self.scanning_speed[idx],
            self.regime_info[idx],
            self.print_direction[idx],
            mic,
            ae,
            self.defect_labels[idx]
        )

class LPBFPointDataset(Dataset):
    def __init__(
        self,
        ids,
        cube_position,
        laser_power,
        scanning_speed,
        regime_info,
        print_direction,
        microphone,
        ae,
        window_size,
        fad_in_out_length = 16
    ):
        self.ids = ids
        self.cube_position = cube_position
        self.laser_power = laser_power
        self.scanning_speed = scanning_speed
        self.regime_info = regime_info
        self.print_direction = print_direction
        self.microphone = microphone
        self.ae = ae
        self.window_size = window_size
        self.fad_in_out_length = fad_in_out_length

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        item_id, _layer_idx, _line_idx, daq_idx, _point_label = self.ids[idx]

        defect_label = _point_label
        _mic = self.microphone[item_id]
        _ae = self.ae[item_id]

        mic = get_windowed_data(_mic,daq_idx,window_size = self.window_size, fade_length=self.fad_in_out_length)
        ae = get_windowed_data(_ae,daq_idx,window_size = self.window_size, fade_length=self.fad_in_out_length)

        return (
            self.cube_position[item_id],
            self.laser_power[item_id],
            self.scanning_speed[item_id],
            self.regime_info[item_id],
            self.print_direction[item_id],
            mic,
            ae,
            defect_label
        )

def fill_nan_scipy(arr):
    x, y = np.meshgrid(np.arange(arr.shape[1]), np.arange(arr.shape[0]))
    points = np.vstack((x[~np.isnan(arr)].flatten(), y[~np.isnan(arr)].flatten())).T
    values = arr[~np.isnan(arr)].flatten()
    arr[np.isnan(arr)] = griddata(points, values, (x[np.isnan(arr)], y[np.isnan(arr)]), method='nearest')


