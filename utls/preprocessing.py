import os
import glob
import subprocess
import string
import seedir
from natsort import natsorted
from tqdm import tqdm
import numpy as np
import copy
import scipy
import time
import pandas as pd
import math
import scienceplots
import librosa
import shutil
import cv2
import dask
import dask.array as da
import multiprocessing
from multiprocessing import shared_memory, Lock
import struct
import zmq
from scipy.interpolate import griddata

from nptdms import TdmsFile

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.colors as colors
from matplotlib.ticker import FuncFormatter
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

cm_std = ["#0C5DA5", "#00B945", "#FF9500", "#FF2C00", "#845B97", "#474747", "#9e9e9e"]
centimeter: float = 1 / 2.54
mpl.rcParams["figure.dpi"] = 300
plt.style.use(["science", "nature"])
# plt.rcParams['figure.constrained_layout.use'] = False
plt.rcParams['text.usetex']= True if shutil.which('latex') else False

one_column = {                      # setup matplotlib to use latex for output
    # "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    # "font.family": "serif",
    # "font.serif": [],                   # blank entries should cause plots 
    # "font.sans-serif": [],              # to inherit fonts from the document
    # "font.monospace": [],
    # "axes.labelsize": 7,               # LaTeX default is 10pt font.
    # "font.size": 7,
    # "legend.fontsize": 7,               # Make the legend/label fonts 
    # "xtick.labelsize": 7,               # a little smaller
    # "ytick.labelsize": 7,
    "figure.figsize": (3.5,2.625),     # default fig size for single colum
    "figure.dpi" : 300,
    "text.latex.preamble": "\n".join([ # plots will use this preamble
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{siunitx}",
        r"\usepackage{upgreek}",
        r"\usepackage{txfonts}",
        ])
    }

two_column = {                      # setup matplotlib to use latex for output
    # "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    # "font.family": "serif",
    # "font.serif": [],                   # blank entries should cause plots 
    # "font.sans-serif": [],              # to inherit fonts from the document
    # "font.monospace": [],
    # "axes.labelsize": 7,               # LaTeX default is 10pt font.
    # "font.size": 7,
    # "legend.fontsize": 7,               # Make the legend/label fonts 
    # "xtick.labelsize": 7,               # a little smaller
    # "ytick.labelsize": 7,
    "figure.figsize": (6.8,6.1),     # default fig size for single colum
    "figure.dpi" : 300,
    "text.latex.preamble": "\n".join([ # plots will use this preamble
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{siunitx}",
        r"\usepackage{upgreek}",
        r"\usepackage{txfonts}",
        ])
    }

# mpl.rcParams.update(one_column)

def check_lists_length(*lists):
    if not lists:
        print("No lists provided.")
        return

    # Get the length of the first list
    base_length = len(lists[0])

    # Flag to check if all lists have identical lengths
    all_identical = True

    # Iterate over all lists
    for i, lst in enumerate(lists):
        if len(lst) != base_length:
            all_identical = False
            print(f"List {i + 1} does not have the same length: {len(lst)}")

    if all_identical:
        print(f"All lists have the identical length: {base_length}")

def lists_to_dataframe(*lists):
    if not lists:
        print("No lists provided.")
        return None

    # Get the length of the first list (header)
    header = lists[0]

    # Check if all lists have the same length as the header
    for i, lst in enumerate(lists[1:], start=2):
        if len(lst) != len(header):
            print(f"List {i} does not have the same length as the header: {len(lst)}")
            return None

    # Convert lists to DataFrame
    data = {header[i]: [lst[i] for lst in lists[1:]] for i in range(len(header))}
    df = pd.DataFrame(data).T

    print("DataFrame created successfully.")
    # print(df)
    return df

def read_bin_file(file_path:str):
    """Reads a binary file and returns its contents as a bytes object.

    Args:
        file_path: The path to the binary file.

    Returns:
        The contents of the binary file as a bytes object.
    """
    # dtype = np.float32 
    dtype = "<i4"
    num_channels = 8 #  the number of channels
    samples_per_channel = int(np.fromfile(file_path, dtype=dtype).shape[0] / num_channels) # Calculate the number of samples per channel
    data = np.fromfile(file_path, dtype=dtype).reshape(-1,num_channels) # Load the data into a NumPy array
    return data

def print_tdms_structure(file_path):
    """Prints the structure of a TDMS file in a tree-like format.
    Args:
        file_path: Path to the TDMS file.
    """
    tdms_file = TdmsFile(file_path)
    print(f"{tdms_file.properties['name']}")

    for group in tdms_file.groups():
        print(f"├── {group.name}")
        for channel in group.channels():
            print(f"│   ├── {channel.name}")
            for prop_name, prop_value in channel.properties.items():
                print(f"│   │   ├── {prop_name}: {prop_value}")
        for prop_name, prop_value in group.properties.items():
            print(f"│   └── Group Property {prop_name}: {prop_value}")

    for prop_name, prop_value in tdms_file.properties.items():
        print(f"└── File Property {prop_name}: {prop_value}")

def linearSpectrogram(
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
    y = standardize_array(y)

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

    # display
    if display == True:
        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots(figsize=(10 * centimeter, 7 * centimeter))
            librosa.display.specshow(
                spectrogram,
                sr=sampling_rate,
                x_axis="time",
                y_axis="linear",
                hop_length=hop_length,
                fmax=fmax,
                fmin=fmin,
            )
            ax.set_title(f"Linear Spectrogram")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            plt.colorbar()
            plt.tight_layout()
            plt.show()
            if save == True:
                plt.savefig(save, dpi=300)
    return spectrogram

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
    y = standardize_array(y)

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
    if display == True:
        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots(figsize=(10 * centimeter, 7 * centimeter))
            img = librosa.display.specshow(
                spectrogram,
                sr=sampling_rate,
                x_axis="time",
                y_axis="linear",
                hop_length=hop_length,
                fmax=fmax,
                fmin=fmin,
            )
            ax.set_title(f"Log-frequency power spectrogram")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            plt.colorbar(format="%+2.0f dB")
            plt.tight_layout()
            plt.show()
            if save == True:
                plt.savefig(save, dpi=300)
    return spectrogram

def melSpectrogram(
    data,
    sampling_rate,
    n_fft=1024,
    hop_length=320,
    window_type="hann",
    mel_bins=64,
    fmin=0,
    fmax=None,
    mel_power=2.0,
    display=False,
    save=False,
):
    """
    mel_bins = 64 # Number of mel bands
    fmin = 0 # lowest frequency (in Hz)
    fmax= None # highest frequency (in Hz). If None, use fmax = sr / 2.0
    mel_power=2 # Exponent for the magnitude melspectrogram. e.g., 1 for energy, 2 for power
    """
    y = data
    y = standardize_array(y)

    spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=sampling_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window_type,
        n_mels=mel_bins,
        power=mel_power,
        fmax=fmax,
        fmin=fmin,
    )

    # display
    if display == True:
        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots(figsize=(10 * centimeter, 7 * centimeter))
            librosa.display.specshow(
                spectrogram,
                sr=sampling_rate,
                x_axis="time",
                y_axis="linear",
                hop_length=hop_length,
                fmax=fmax,
                fmin=fmin,
            )
            ax.set_title(f"Mel Spectrogram")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            plt.colorbar()
            plt.tight_layout()
            plt.show()
            if save == True:
                plt.savefig(save, dpi=300)
    return spectrogram

def logMelSpectrogram(
    data,
    sampling_rate,
    n_fft=1024,
    hop_length=320,
    window_type="hann",
    mel_bins=64,
    fmin=0,
    fmax=None,
    mel_power=2.0,
    display=False,
    save=False,
):
    """
    mel_bins = 64 # Number of mel bands
    fmin = 0 # lowest frequency (in Hz)
    fmax= None # highest frequency (in Hz). If None, use fmax = sr / 2.0
    mel_power=2 # Exponent for the magnitude melspectrogram. e.g., 1 for energy, 2 for power
    """
    y = data
    y = standardize_array(y)

    spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=sampling_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window_type,
        n_mels=mel_bins,
        power=mel_power,
        fmax=fmax,
        fmin=fmin,
    )

    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # display
    if display == True:
        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots(figsize=(10 * centimeter, 7 * centimeter))
            librosa.display.specshow(
                spectrogram,
                sr=sampling_rate,
                x_axis="time",
                y_axis="linear",
                hop_length=hop_length,
                fmax=fmax,
                fmin=fmin,
            )
            ax.set_title(f"Log Mel Spectrogram")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            plt.colorbar(format="%+2.0f dB")
            plt.tight_layout()
            plt.show()
            if save == True:
                plt.savefig(save, dpi=300)
    return spectrogram

def cross_similarity(
    y_ref, y_comp, sampling_rate, hop_length, metric="cosine", display=False
):

    y_ref = standardize_array(y_ref)
    y_comp = standardize_array(y_comp)
    chroma_ref = librosa.feature.chroma_cqt(
        y=y_ref, sr=sampling_rate, hop_length=hop_length
    )
    chroma_comp = librosa.feature.chroma_cqt(
        y=y_comp, sr=sampling_rate, hop_length=hop_length
    )
    # Use time-delay embedding to get a cleaner recurrence matrix
    x_ref = librosa.feature.stack_memory(chroma_ref, n_steps=10, delay=3)
    x_comp = librosa.feature.stack_memory(chroma_comp, n_steps=10, delay=3)
    # xsim = librosa.segment.cross_similarity(x_comp, x_ref)
    xsim = librosa.segment.cross_similarity(x_comp, x_ref, metric=metric)
    xsim_aff = librosa.segment.cross_similarity(
        x_comp, x_ref, metric=metric, mode="affinity"
    )

    if display == True:

        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
            imgsim = librosa.display.specshow(
                xsim, x_axis="s", y_axis="s", hop_length=hop_length, ax=ax[0]
            )
            ax[0].set(title="Binary cross-similarity (symmetric)")
            imgaff = librosa.display.specshow(
                xsim_aff,
                x_axis="s",
                y_axis="s",
                cmap="magma_r",
                hop_length=hop_length,
                ax=ax[1],
            )
            ax[1].set(title="Cross-affinity")
            ax[1].label_outer()
            fig.colorbar(imgsim, ax=ax[0], orientation="horizontal", ticks=[0, 1])
            fig.colorbar(imgaff, ax=ax[1], orientation="horizontal")
            plt.show()

    return xsim, xsim_aff

def recurrence(y_ref, sampling_rate, hop_length, metric="cosine", display=False):

    y_ref = standardize_array(y_ref)
    chroma_ref = librosa.feature.chroma_cqt(
        y=y_ref, sr=sampling_rate, hop_length=hop_length
    )
    # Use time-delay embedding to get a cleaner recurrence matrix
    x_ref = librosa.feature.stack_memory(chroma_ref, n_steps=10, delay=3)
    # xsim = librosa.segment.cross_similarity(x_comp, x_ref)
    R_bi = librosa.segment.recurrence_matrix(x_ref, metric=metric, sym=True)
    R_aff = librosa.segment.recurrence_matrix(
        x_ref, metric=metric, mode="affinity", sym=True
    )

    if display == True:

        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots(figsize=(10 * centimeter, 7 * centimeter))
            imgsim = librosa.display.specshow(
                R_bi,
                x_axis="s",
                y_axis="s",
                hop_length=hop_length,
                ax=ax,
                sr=sampling_rate,
            )
            ax.set(title="Binary recurrence (symmetric)")
            fig.colorbar(imgsim, ax=ax, orientation="vertical", ticks=[0, 1])
            plt.show()

        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots(figsize=(10 * centimeter, 7 * centimeter))
            imgsim = librosa.display.specshow(
                R_aff,
                x_axis="s",
                y_axis="s",
                hop_length=hop_length,
                ax=ax,
                sr=sampling_rate,
            )
            ax.set(title="Affinity recurrence")
            fig.colorbar(imgsim, ax=ax, orientation="vertical")
            plt.show()

    return R_bi, R_aff

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

def normalize_array(array):
  """Normalizes a NumPy array to the range [0, 1].

  Args:
    array: The NumPy array to normalize.

  Returns:
    The normalized NumPy array.
  """

  min_val = np.min(array)
  max_val = np.max(array)
  return (array - min_val) / (max_val - min_val)

def get_first_and_last_index_above_value(arr, value):
    """
    Finds the first and last indices in a NumPy array that are above a given value.

    Args:
        arr: The NumPy array.
        value: The value to compare against.

    Returns:
        A tuple containing the first and last indices, or (None, None) if no indices are found.
    """

    indices = np.where(arr > value)[0]
    if indices.size == 0:
        return None, None
    return indices[0], indices[-1]

def standardize_dataframe_vertically(df):
    """
    Standardizes a Pandas DataFrame vertically using built-in Pandas and NumPy functions.

    Args:
        df: The Pandas DataFrame to standardize.

    Returns:
        A standardized copy of the DataFrame.
    """

    # Create a copy of the DataFrame to avoid modifying the original
    df_standardized = df.copy()

    # Calculate mean and standard deviation for each column
    column_means = df_standardized.mean()
    column_stds = df_standardized.std()

    # Standardize each column using built-in functions
    df_standardized = (df_standardized - column_means) / column_stds

    return df_standardized

def onehot_encode_dataframe_vertically(df):
    """
    One-hot encodes a Pandas DataFrame vertically using built-in Pandas functions.

    Args:
        df: The Pandas DataFrame to one-hot encode.

    Returns:
        A one-hot encoded copy of the DataFrame.
    """

    # Create a copy of the DataFrame to avoid modifying the original
    df_encoded = df.copy()

    # One-hot encode each column using get_dummies
    df_encoded = pd.get_dummies(df_encoded, columns=df_encoded.columns)

    return df_encoded

def onehot_encode_dataframe_vertically(df):
    """
    One-hot encodes a Pandas DataFrame vertically using built-in Pandas functions.

    Args:
        df: The Pandas DataFrame to one-hot encode.

    Returns:
        A one-hot encoded copy of the DataFrame.
    """

    # Create a copy of the DataFrame to avoid modifying the original
    df_encoded = df.copy()

    # One-hot encode each column using get_dummies
    df_encoded = pd.get_dummies(df_encoded, columns=df_encoded.columns)

    return df_encoded

def one_hot_encode_numpy(labels, num_classes):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

def find_first_and_last_indices(arr, number):
  """
  Finds the first and last indices of a given number in a NumPy array.

  Args:
    arr: The input NumPy array.
    number: The number to search for.

  Returns:
    A tuple containing the first and last indices of the number, or (-1, -1) if the number is not found.
  """

  indices = np.where(arr == number)[0]
  if len(indices) == 0:
    return -1, -1
  else:
    return indices[0], indices[-1]

def get_process_regime(layer_number,process_regime):
    """
    Finds the layer mapping category for a given layer number.
    Returns the category as a string.
    """
    for start, end, category, offset in process_regime:
        if start <= layer_number <= end:
            return category
    return None

def signal_interval_splitter(signal, labels,signal_fs, time_threshold:list=[0.003, 0.008],signalIndex=None):
    """
    split the signal by the interveal of labels
    Detect the durations greater than 3ms and less than 8ms 
    """
    min_laser_power = np.mean(labels[0:int(0.1*signal_fs)])
    laser_binary_mask = labels > 4*min_laser_power
    # Invert the binary mask to find the zero regions
    inverted_mask = 1 - laser_binary_mask
    labeled_array, num_features = scipy.ndimage.label(inverted_mask)

    min_gap_size = int(time_threshold[0]*signal_fs)  # Minimum length of a continuous region of 0s
    max_gap_size = int(time_threshold[1]*signal_fs)  # Maximum length of a continuous region of 0s
    # Find the sizes of each labeled region
    region_sizes = np.array([(labeled_array == i).sum() for i in range(1, num_features + 1)])
    # Get indices of regions that satisfy the gap size condition
    valid_region_indices = np.where((region_sizes >= min_gap_size) & (region_sizes <= max_gap_size))[0] + 1

    # Get the start and end positions of the valid regions
    valid_regions = [(np.where(labeled_array == idx)[0][0], np.where(labeled_array == idx)[0][-1]) for idx in valid_region_indices]
    first_index = np.argmax(laser_binary_mask)  # Finds the first occurrence of 1
    last_index = len(laser_binary_mask) - 1 - np.argmax(laser_binary_mask[::-1])  # Finds the last occurrence of 1

    # we have all the regions now, first get the indexes prepared for actual signal prot
    signal_portions = []
    curr_ind = first_index
    for start,end in valid_regions:
        signal_portions.append([curr_ind,start])
        # Make the end index the start index for the next run
        curr_ind = end
    # Add the last portion
    signal_portions.append([curr_ind,last_index])

    split_signal = []
    split_labels =[]
    split_laser_orig = []
    for start,end in signal_portions:
        split_signal.append(signal[start:end])
        split_labels.append(laser_binary_mask[start:end])
        split_laser_orig.append(labels[start:end])
    


    # If signal index is specified return the specific signal or return the whole list of signals
    if signalIndex != None:
        return split_signal[signalIndex],split_labels[signalIndex]
    else:
        return split_signal,split_labels


    """
    Debug Code just in case
    """
    # for sig in split_signal:
    #     plt.figure()
    #     t_axis = np.arange(len(sig)) / fs
    #     plt.plot(t_axis,sig)
    
    # Lets begin by visualising the data first
    # plt.figure()
    # t_axis = np.arange(len(laser_data)) / fs
    # plt.plot(t_axis,signal)
    # plt.plot(t_axis,laser_data)
    # plt.plot(t_axis,laser_binary_mask)
    # for start, end in valid_regions:
    #     plt.axvline(x = t_axis[start], color = 'green', label = 'axvline - full height')
    #     plt.axvline(x = t_axis[end], color = 'green', label = 'axvline - full height')
    # plt.axvline(x = t_axis[first_index], color = 'green', label = 'axvline - full height')
    # plt.axvline(x = t_axis[last_index], color = 'green', label = 'axvline - full height')
    # plt.show()

def pulse_signal_slicer_by_interval(signal, sampling_rate, pulse_on_thershold=0.2, min_interval_time=0.003, normalized=True,index_only=False,binary=True):
    if normalized == True:
        signal = normalize_array(signal) 
    else:
        signal = signal
    min_interval_samples = min_interval_time*sampling_rate
    is_on = signal> pulse_on_thershold

    on_indices = np.where(np.diff(is_on.astype(int)) == 1)[0] + 1
    off_indices = np.where(np.diff(is_on.astype(int)) == -1)[0] + 1
    # Handle the case where the signal is on at the start or off at the end
    if is_on[0]:
        on_indices = np.insert(on_indices, 0, 0) # If the array starts with True, prepend 0 to starts
    if is_on[-1]:
        off_indices = np.append(off_indices, len(signal)) # If the array ends with True, append the length of the array to ends
    segment_indices = list(zip(on_indices, off_indices))
    if binary ==True:
        return segment_indices
    intervals = (on_indices[1:] - off_indices[:-1])
    filtered_intervals = [interval for interval in intervals if interval > min_interval_samples]
    interval_segments = [segment_indices[i+1] for i, interval in enumerate(intervals) if interval >= min_interval_samples]
    signal_segments_indices = [(on_indices[0],interval_segments[0][0])]
    [signal_segments_indices.append((j[1],interval_segments[i+1][0])) for i,j in enumerate(interval_segments[:-1])]
    signal_segments_indices.append((interval_segments[-1][1],off_indices[-1]))
    return signal_segments_indices if index_only == True else np.asarray([signal[i[0]:i[1]] for i in signal_segments_indices])
    
class MaPS_LPBF_Construction:
    import os
    from nptdms import TdmsFile
    from natsort import natsorted
    """
    Data reader for the MaPS LPBF EXP1
    
    """
    def __init__(self,daq_dir,lmq_dir,sampling_rate_daq,sampling_rate_lmq,process_regime,laser_power_setting, scanning_speed_setting,defect_labels=None):
        self.daq_name_list = natsorted(
            [i for i in os.listdir(daq_dir) if i.split(".")[-1] == "tdms"]
        )
        self.lmq_name_list = natsorted([i for i in os.listdir(lmq_dir) if i.split(".")[-1] == "bin"])
        self.lmq_channel_name = [
            "Vector ID",
            "meltpooldiode",
            "X Coordinate",
            "Y Coordinate",
            "Laser power",
            "Spare",
            "Laser diode",
            "Varioscan(focal length)",
        ]
        self.daq_dir = daq_dir
        self.lmq_dir = lmq_dir

        self.sampling_rate_lmq = sampling_rate_lmq
        self.sampling_rate_daq = sampling_rate_daq
        self.process_regime = process_regime
        self.laser_power_setting = laser_power_setting
        self.scanning_speed_setting = scanning_speed_setting
        self.defect_labels = defect_labels

        ## Datalist with interface

        # Context info
        self.cube_position = []
        self.laser_power = []
        self.start_coord = []
        self.end_coord = []
        self.scanning_speed = []
        self.regime_info = []

        # defect_labels
        self.defect_labels = []

        # in_process_data
        self.microphone = []
        self.ae = []
        self.photodiode = []

    def _read_bin_file(self, file_path:str):
            """Reads a binary file and returns its contents as a bytes object.

            Args:
                file_path: The path to the binary file.

            Returns:
                The contents of the binary file as a bytes object.
            """
            # dtype = np.float32 
            dtype = "<i4"
            num_channels = 8 #  the number of channels
            samples_per_channel = int(np.fromfile(file_path, dtype=dtype).shape[0] / num_channels) # Calculate the number of samples per channel
            data = np.fromfile(file_path, dtype=dtype).reshape(-1,num_channels) # Load the data into a NumPy array
            return data
  
    def _construct_cube_wise_indices(self):

        pulse_on_thershold = 0.2
        sliced_indices_daq = np.asarray(pulse_signal_slicer_by_interval( self._photodiode, self.sampling_rate_daq, pulse_on_thershold=pulse_on_thershold, index_only=True,binary=False))
        sliced_indices_lmq = np.asarray(pulse_signal_slicer_by_interval( self._laserdiode, self.sampling_rate_lmq, pulse_on_thershold=pulse_on_thershold, index_only=True,binary=False))
        while len(sliced_indices_daq)!=len(sliced_indices_lmq)!=5:
            pulse_on_thershold -= 0.05
            sliced_indices_daq = np.asarray(pulse_signal_slicer_by_interval( self._photodiode, self.sampling_rate_daq, pulse_on_thershold=pulse_on_thershold, index_only=True))
            sliced_indices_lmq = np.asarray(pulse_signal_slicer_by_interval( self._laserdiode, self.sampling_rate_lmq, pulse_on_thershold=pulse_on_thershold, index_only=True))

        sliced_indices_lmq_0 = np.zeros([5,2],dtype=int)
        for i, vid in enumerate(np.unique(self.data_lmq[:,0])[1:-1]):
            sliced_indices_lmq_0[i,0], sliced_indices_lmq_0[i,1] = find_first_and_last_indices(self.data_lmq[:,0], vid)
   
        return sliced_indices_lmq, sliced_indices_daq
    
    def _construct_line_wise_indices(self):
        current_cube_lmq = self.sliced_indices_lmq[self.cube_i]
        current_cube_daq = self.sliced_indices_daq[self.cube_i]
        self.current_cube_lmq = current_cube_lmq
        self.current_cube_daq = current_cube_daq

        self.current_laserdiode = self._laserdiode[current_cube_lmq[0]:current_cube_lmq[1]]
        self.current_laserpower = self._laserpower[current_cube_lmq[0]:current_cube_lmq[1]]
        self.current_x = self._x[current_cube_lmq[0]:current_cube_lmq[1]]
        self.current_y = self._y[current_cube_lmq[0]:current_cube_lmq[1]]

        self.current_photodiode = self._photodiode[current_cube_daq[0]:current_cube_daq[1]]
        self.current_ae         = self._ae[current_cube_daq[0]:self.current_cube_daq[1]]
        self.current_mic        = self._mic[current_cube_daq[0]:self.current_cube_daq[1]]
        # print(f"current mic shape :{self.current_mic.shape}")
        
        pulse_on_thershold = 0.51
        line_indices_lmq = np.asarray(pulse_signal_slicer_by_interval(self.current_laserpower,sampling_rate=self.sampling_rate_lmq,min_interval_time=0.0001, index_only=True,pulse_on_thershold=pulse_on_thershold))
        line_indices_daq = np.asarray(pulse_signal_slicer_by_interval(self.current_photodiode,sampling_rate=self.sampling_rate_daq,min_interval_time=0.0001, index_only=True,pulse_on_thershold=pulse_on_thershold))
        # line_indices_lmq.shape,line_indices_daq.shape
        while (len(line_indices_lmq)!=len(line_indices_daq)):
            # print(f"T:{pulse_on_thershold:.3}|lmq:{line_indices_lmq.shape}|daq:{line_indices_daq.shape}")
            line_indices_lmq = np.asarray(pulse_signal_slicer_by_interval(self.current_laserpower,sampling_rate=self.sampling_rate_lmq,min_interval_time=0.0001, index_only=True,pulse_on_thershold=pulse_on_thershold))
            line_indices_daq = np.asarray(pulse_signal_slicer_by_interval(self.current_photodiode,sampling_rate=self.sampling_rate_daq,min_interval_time=0.0001, index_only=True,pulse_on_thershold=pulse_on_thershold))
            if pulse_on_thershold<0.11:
                break
            pulse_on_thershold -= 0.05
        min_len = min(len(line_indices_lmq),len(line_indices_daq))
        if min_len<10:
            print(f"Warning, only {min_len} lines in {self.layer_i} {self.cube_i}")
        return line_indices_lmq[:min_len-1], line_indices_daq[:min_len-1]

    def _construct_data_labels(self):
        for _lmq,_daq in zip(self.line_indices_lmq,self.line_indices_daq):
            lmq_i0, lmq_it = _lmq
            daq_i0, daq_it = _daq
            # print(f"cube:{self.cube_i}|lmq i{_lmq}|daq i{_daq}")
            
            self.cube_position.append(self.cube_i)
            self.laser_power.append(int(self._find_regime_setting(self.layer_i,self.laser_power_setting)))
            self.start_coord.append((self.current_x[lmq_i0],self.current_y[lmq_i0]))
            self.end_coord.append((self.current_x[lmq_it],self.current_y[lmq_it]))
            self.scanning_speed.append(int(self._find_regime_setting(self.layer_i,self.scanning_speed_setting)))
            self.regime_info.append(self._find_regime_setting(self.layer_i))


            self.microphone.append(self.current_mic[daq_i0:daq_it])
            self.ae.append(self.current_ae[daq_i0:daq_it])
            self.photodiode.append(self.current_photodiode[daq_i0:daq_it])
            self.defect_labels.append(0) # //TODO the defect labels
    
    def _construct_line_wise_labels_ram(self,layer_i=None,cube_i=None):
        """
        Construct line wise labels in RAM
        Return: 
            - Context info
                - Cube position (the n-th cube)
                - Laser power
                - Speed (remain constant in this case)
                - Scanning direction
                - Regime info (4 - class classifications)
                    - Base | Base plate, ignored
                    - GP   | Gas pore, or keyhole 
                    - NP   | No pore
                    - RLoF | Random lack of fusion
                    - LoF  | Lack of fusion

            - Strong labels ( TODO )
                    - GP   | Gas pore, or keyhole 
                    - NP   | No pore
                    - RLoF | Random lack of fusion
                    - LoF  | Lack of fusion
            
            - In-process data
                - microphone
                - AE
                - photodiode
        """
        def _run_layer(layer_i,cube_i):
            self.layer_i = layer_i
            # print(f"layer:{layer_i}")
            daq_file_path =os.path.join(self.daq_dir,self.daq_name_list[layer_i])
            lmq_file_path = os.path.join(self.lmq_dir,self.lmq_name_list[layer_i])
            self.data_daq = TdmsFile(daq_file_path)
            self.data_lmq = self._read_bin_file(lmq_file_path)
            _bin_name = os.path.split(lmq_file_path)[-1].split(".")[0]

            self._mic =        (self.data_daq.groups()[0]).channels()[0].data
            self._ae =         (self.data_daq.groups()[0]).channels()[1].data
            self._photodiode = (self.data_daq.groups()[0]).channels()[2].data

            self._x =          self.data_lmq[:,2]
            self._y =          self.data_lmq[:,3]
            self._laserpower = self.data_lmq[:,4]
            self._laserdiode = self.data_lmq[:,6]
            self._daq_t=np.linspace(0,len(self._mic)/self.sampling_rate_daq,len(self._mic))
            self._lmq_t=np.linspace(0,len(self._laserdiode)/self.sampling_rate_lmq,len(self._laserdiode))
            sliced_indices_lmq, sliced_indices_daq = self._construct_cube_wise_indices()
            assert len(sliced_indices_lmq) == len(sliced_indices_daq) ==5

            self.sliced_indices_lmq = sliced_indices_lmq
            self.sliced_indices_daq = sliced_indices_daq

            if cube_i != None:
                # print(f"cube{cube_i}")
                self.cube_i = cube_i
                line_indices_lmq, line_indices_daq = self._construct_line_wise_indices()
                self.line_indices_lmq = line_indices_lmq
                self.line_indices_daq = line_indices_daq
                self._construct_data_labels()

            else:
                for cube_i in range(5):
                    self.cube_i = cube_i
                    # print(f"cube{cube_i}")
                    line_indices_lmq, line_indices_daq = self._construct_line_wise_indices()
                    self.line_indices_lmq = line_indices_lmq
                    self.line_indices_daq = line_indices_daq
                    self._construct_data_labels()
        
        if layer_i:
                _run_layer(layer_i,cube_i)
        else:
            for layer_i in tqdm(range(len(self.daq_name_list))[:]):
                self.layer_i = layer_i
                if self._find_regime_setting(layer_i) =="Base":
                    continue
                _run_layer(layer_i,cube_i)

    def _xy_rescaler(self,x_offset= -17049,y_offset=16883,x_scaling=3409.76,y_scaling=3376):
        """
            This function has four magic numbers, 2 for offsets and 2 for scaling
            These numbers are provided by Shivam
            It convers the aribitary units of the X and Y coord from the LMQ data into microMeters (um)
            Magic Numbers:
                X-offset: -17049
                Y-offset: 16883

                X-scaling: 3.40976 bit/um
                Y-scaling: 3.376 bit/um
        """
        X =  self.current_x
        Y = self.current_y
        x_mod = (X+x_offset)*x_scaling
        y_mod = (Y+y_offset)*y_scaling
        self.current_x = x_mod
        self.current_y = y_mod
        return x_mod,y_mod

    def binaryReader(self,binaryFile):
        """
            This function is for reading the LMQ binary monitoring 
            The original function is provided by Shivam to build
            LMQ sampling rate is 100 kHZ 

            Column Descriptions:
            VectorID: Actual Speed and Laser power mapping to some Table of Shivam
            MPD: Melt pool laser diode actual sensor data
            Xcoord: X coordinate of the laser scanning area
            Ycoord: Y coordinate of the laser scanning area
            NLP: Nominal Laser Power 
            Spare: Spare column for some data in future
            LPD: Laser Photo diode <Logged alongside AE>
                 Measures laser power before falling into the power bed
            VPOS: Laser Lens focus area, usually 14000, 
        
        """
        with open(binaryFile, 'rb') as file:
            # Read the data as 32-bit signed integers ('int32') in little-endian format ('<')
            data_LMQ = np.fromfile(file, dtype=np.int32)

        # Reshape the data for each layer
        #for i in range(number_of_layers):
            # Reshape the data to have 8 columns and transpose it
        data_LMQ = data_LMQ.reshape(-1, 8)

        # Extract data components from reshaped data
        #for i in range(number_of_layers):
        VectorID = data_LMQ[:, 0]  # Assuming the first column represents VectorID
        MPD = data_LMQ[:, 1]       # Second column for MPD
        Xcoord = data_LMQ[:, 2]    # Third column for Xcoord
        Ycoord = data_LMQ[:, 3]    # Fourth column for Ycoord
        NLP = data_LMQ[:, 4]       # Fifth column for NLP
        Spare = data_LMQ[:, 5]     # Sixth column for Spare
        LPD = data_LMQ[:, 6]       # Seventh column for LPD
        VPOS = data_LMQ[:, 7]      # Eighth column for VPOS

        x,y = self._xy_rescaler(Xcoord,Ycoord)

        self.Xcoord = x
        self.Ycoord = y

        return

    def _find_regime_setting(self, layer_i, setting=None):
        setting = self.process_regime if setting == None else setting
        for start, end, category in setting:
            if start <= layer_i <= end:
                return category
        return None

class MaPS_LPBF_Point_Wise_Construction(MaPS_LPBF_Construction):
    def __init__(self, daq_dir,lmq_dir,sampling_rate_daq,sampling_rate_lmq,process_regime,laser_power_setting, scanning_speed_setting,label_dir,roi_radius):
        super().__init__(daq_dir,lmq_dir,sampling_rate_daq,sampling_rate_lmq,process_regime,laser_power_setting, scanning_speed_setting,defect_labels=None)
        self.label_dir = label_dir
        self.layer_indices = []
        self.line_indices = []
        self.line_labels = []
        self.point_labels = []
        self.roi_radius = roi_radius
        
    def _construct_point_wise_labels_ram(self,layer_i=None,cube_i=None):
        """
        Construct line wise labels in RAM
        Return: 
            - Context info
                - Cube position (the n-th cube)
                - Laser power
                - Speed (remain constant in this case)
                - Scanning direction
                - Regime info (4 - class classifications)
                    - Base | Base plate, ignored
                    - GP   | Gas pore, or keyhole 
                    - NP   | No pore
                    - RLoF | Random lack of fusion
                    - LoF  | Lack of fusion

            - Strong labels ( only binary at this moment)
                    - GP   | Gas pore, or keyhole 
                    - NP   | No pore
                    - RLoF | Random lack of fusion
                    - LoF  | Lack of fusion
            
            - In-process data
                - microphone
                - AE
                - photodiode
        """
        def _run_layer(layer_i,cube_i):
            self.layer_i = layer_i
            # print(f"layer:{layer_i}")
            daq_file_path =os.path.join(self.daq_dir,self.daq_name_list[layer_i])
            lmq_file_path = os.path.join(self.lmq_dir,self.lmq_name_list[layer_i])

            _roi_label = np.asarray(np.load(os.path.join(self.label_dir,f"cube{cube_i}",f"layer{layer_i}_roi_radius{self.roi_radius}.npy")),dtype=int)

            def _get_line_wise_defect_label(roi_label): 
                line_segs = []
                defect_labels_roi = []
                _shapes = [0]
                for i0,it in self.line_indices_lmq:
                    _shapes.append(self.current_x[i0:it].shape[0])
                    defect_labels_roi.append([
                        # lpbf.current_x[i0:it],
                        # lpbf.current_y[i0:it],
                        roi_label[np.sum(_shapes[:-1]):np.sum(_shapes)]])
                    self.point_labels.append(np.where(defect_labels_roi[-1][0]>0,1,0).tolist())
                line_wise_defect_label = np.asarray([np.sum(i) for i in defect_labels_roi])
                return np.where(line_wise_defect_label>0,1,0)

            
            self.data_daq = TdmsFile(daq_file_path)
            self.data_lmq = self._read_bin_file(lmq_file_path)
            _bin_name = os.path.split(lmq_file_path)[-1].split(".")[0]

            self._mic =        (self.data_daq.groups()[0]).channels()[0].data
            self._ae =         (self.data_daq.groups()[0]).channels()[1].data
            self._photodiode = (self.data_daq.groups()[0]).channels()[2].data

            self._x =          self.data_lmq[:,2]
            self._y =          self.data_lmq[:,3]
            self._laserpower = self.data_lmq[:,4]
            self._laserdiode = self.data_lmq[:,6]
            self._daq_t=np.linspace(0,len(self._mic)/self.sampling_rate_daq,len(self._mic))
            self._lmq_t=np.linspace(0,len(self._laserdiode)/self.sampling_rate_lmq,len(self._laserdiode))
            sliced_indices_lmq, sliced_indices_daq = self._construct_cube_wise_indices()
            assert len(sliced_indices_lmq) == len(sliced_indices_daq) ==5

            self.sliced_indices_lmq = sliced_indices_lmq
            self.sliced_indices_daq = sliced_indices_daq


            if cube_i != None:
                # print(f"cube{cube_i}")
                self.cube_i = cube_i
                line_indices_lmq, line_indices_daq = self._construct_line_wise_indices()
                self.line_indices_lmq = line_indices_lmq
                self.line_indices_daq = line_indices_daq
                self._current_defect_label = _get_line_wise_defect_label(_roi_label)
                self._construct_data_labels()

            else:
                for cube_i in range(5):
                    self.cube_i = cube_i
                    # print(f"cube{cube_i}")
                    line_indices_lmq, line_indices_daq = self._construct_line_wise_indices()
                    self.line_indices_lmq = line_indices_lmq
                    self.line_indices_daq = line_indices_daq
                    self._current_defect_label = _get_line_wise_defect_label(_roi_label)
                    self._construct_data_labels()
        
        if layer_i:
                _run_layer(layer_i,cube_i)
        else:
            for layer_i in tqdm(range(len(self.daq_name_list))[:]):
                self.layer_i = layer_i
                if self._find_regime_setting(layer_i) =="Base":
                    continue
                _run_layer(layer_i,cube_i)

    def _construct_data_labels(self):
        for i,(_lmq,_daq) in enumerate(zip(self.line_indices_lmq,self.line_indices_daq)):
            lmq_i0, lmq_it = _lmq
            daq_i0, daq_it = _daq
            # print(f"cube:{self.cube_i}|lmq i{_lmq}|daq i{_daq}")
            
            self.cube_position.append(self.cube_i)
            self.laser_power.append(int(self._find_regime_setting(self.layer_i,self.laser_power_setting)))
            self.start_coord.append((self.current_x[lmq_i0],self.current_y[lmq_i0]))
            self.end_coord.append((self.current_x[lmq_it],self.current_y[lmq_it]))
            self.scanning_speed.append(int(self._find_regime_setting(self.layer_i,self.scanning_speed_setting)))
            self.regime_info.append(self._find_regime_setting(self.layer_i))


            self.microphone.append(self.current_mic[daq_i0:daq_it])
            self.ae.append(self.current_ae[daq_i0:daq_it])
            self.photodiode.append(self.current_photodiode[daq_i0:daq_it]) 
            self.line_labels.append(self._current_defect_label[i])
            self.layer_indices.append(self.layer_i)
            self.line_indices.append(i)

def create_shared_memory_array(shape):
    shared_mem = multiprocessing.shared_memory.SharedMemory(create=True, size=int(np.prod(shape) * np.dtype(np.int32).itemsize))
    shared_array = np.ndarray(shape, dtype=np.int32, buffer=shared_mem.buf)
    return shared_mem, shared_array

@dask.delayed
def make_obj_mask_shared_memory(point, radius, mask_shape, shm_name, lock):
    # Reconnect to the shared memory segment by name
    existing_shm = multiprocessing.shared_memory.SharedMemory(name=shm_name)
    mask = np.ndarray(mask_shape, dtype=np.int32, buffer=existing_shm.buf)
    x, y = point
    with lock:
        cv2.circle(mask, (int(x), int(y)), radius, 1, thickness=-1)
    existing_shm.close()  # Close the shared memory when done

def make_obj_mask(point, radius, mask_shape):
    mask = np.zeros(mask_shape, dtype=np.uint8)
    x, y = point
    cv2.circle(mask, (int(x), int(y)), radius, 1, thickness=-1)
    # for i in range(int(max(0, x - radius)), int(min(mask_shape[0], x + radius + 1))):
    #     for j in range(int(max(0, y - radius)), int(min(mask_shape[1], y + radius + 1))):
    #         if (i - x)**2 + (j - y)**2 <= radius**2:
    #             mask[i, j] = 1
    # return mask

def make_obj_masks_multiprocessing(trajectory, radius, mask_shape):
    # import concurrent.futures
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     futures = [executor.submit(make_obj_mask, point, mask, radius) for point in trajectory]
    #     concurrent.futures.wait(futures)
    with multiprocessing.Pool() as pool:
        results = pool.starmap(make_obj_mask, [(point, radius, mask_shape) for point in trajectory])
    combined_mask = np.sum(results, axis=0)
    return combined_mask

def process_trajectory_with_dask_shared_memory(trajectory, radius, mask_shape):

    shared_mem, shared_mask = create_shared_memory_array(mask_shape)
    lock = multiprocessing.Lock()
    tasks = [make_obj_mask_shared_memory(point, radius, mask_shape,shared_mem.name,lock) for point in trajectory]
    masks = dask.compute(*tasks) 
    # Retrieve the final mask from shared memory
    result = np.copy(shared_mask)  # Copy the data to a NumPy array
    # Clean up shared memory
    shared_mem.close()
    shared_mem.unlink()  # Remove the shared memory block
    return result

def roi(data,center,radius):
    # Create an empty mask with the same shape as the data
    highlight_mask = np.zeros_like(data, dtype=bool)

    # Loop through the array and check the distance to the center
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Calculate the Euclidean distance from the center
            distance = np.sqrt((i - center[0])**2 + (j - center[1])**2)
            # Highlight points within the radius
            if distance <= radius:
                highlight_mask[i, j] = True

    # Create a masked array to highlight the points within the radius
    highlighted_data = np.ma.masked_where(~highlight_mask, data)
    return highlighted_data

def make_radius_mask(point, radius, mask_shape,points_value,point_i,comparision_data):
    mask = np.zeros(mask_shape, dtype=np.uint8)
    x, y = point
    cv2.circle(mask, (int(x), int(y)), radius, 1, thickness=-1)
    ma = np.ma.masked_where(mask!=1, comparision_data)
    ma_comp = ma.compressed()
    points_value[point_i] = int(np.sum(np.where(np.isnan(ma_comp), 0, ma_comp)))
    return mask

@dask.delayed
def make_radius_mask_shared_memory(point, radius, mask_shape, points_length, point_i, comparision_data, shm_name, lock):
    existing_shm = multiprocessing.shared_memory.SharedMemory(name=shm_name)
    points_value = np.ndarray(points_length, dtype=np.int32, buffer=existing_shm.buf)
    mask = np.zeros(mask_shape, dtype=np.uint8)
    x, y = point
    with lock:
        cv2.circle(mask, (int(x), int(y)), radius, 1, thickness=-1)
        ma = np.ma.masked_where(mask!=1, comparision_data)
        ma_comp = ma.compressed()
        points_value[point_i] = int(np.sum(np.where(np.isnan(ma_comp), 0, ma_comp)))
    existing_shm.close()
    return points_value

def process_roi_radius_with_mask_shared_memory(trajectory, comparision_data, radius, mask_shape):
    points_length = len(trajectory)
    shared_mem, shared_mask = create_shared_memory_array(points_length)
    lock = multiprocessing.Lock()
    tasks = [make_radius_mask_shared_memory(point, radius, mask_shape,points_length,i, comparision_data, shared_mem.name,lock) for i, point in enumerate(trajectory)]
    masks = dask.compute(*tasks) 
    # Retrieve the final mask from shared memory
    result = np.copy(shared_mask)  # Copy the data to a NumPy array
    # Clean up shared memory
    shared_mem.close()
    shared_mem.unlink()  # Remove the shared memory block
    return result

class Sender():
    def __init__(self,port=10086):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        # self.socket.bind(f"tcp://*:{self.port}")
        self.socket.connect(f"tcp://localhost:{self.port}")
        
    def _send_metadata(self, trajectory, ct_img_filter, mask_radius):
        # Metadata: trajectory length, ct_img_filter shape, ROI radius
        trajectory_length = trajectory.shape[0]
        ct_img_rows, ct_img_cols = ct_img_filter.shape
        
        # Pack metadata into a binary format: 4 integers
        metadata = struct.pack('llll', mask_radius, trajectory_length, ct_img_rows, ct_img_cols)
        
        # Send metadata
        # print(f"Sending..\n{trajectory_length,ct_img_rows,ct_img_cols} \n from {self.port}")
        self.socket.send(metadata)
        
        # Wait for acknowledgment
        ack = self.socket.recv()
        # print(f"Received ack for metadata: {ack.decode('latin-1')}")
        return ack.decode('latin-1')

    def _send_array(self, array):
        # Convert the array to bytes and send it
        # print(f"Sending..\n{array} \n from {self.port}")
        array_bytes = array.flatten().tobytes()
        self.socket.send(array_bytes)
        
        # Wait for acknowledgment
        response = self.socket.recv()
        # print(f"Received response: {response.decode('latin-1')}")
        return response

    def _send_array_in_chunk(self, array,chunk_size=1024):
        # Convert the array to bytes and send it
        print(f"Sending..\n{array} \n from {self.port}")
        rows, cols = array.shape
        # Send the metadata first: rows and cols of the array
        metadata = struct.pack('ll', rows, cols)  # Use 'll' for two long integers (rows and cols)
        self.socket.send(metadata, flags=zmq.SNDMORE)
        
        # Flatten the array and split it into chunks
        flat_array = array.ravel()
        total_elements = len(flat_array)
        
        # Send data in chunks
        for i in range(0, total_elements, chunk_size):
            chunk = flat_array[i:i + chunk_size]
            # Send the chunk
            self.socket.send(chunk.tobytes(), flags=zmq.SNDMORE if i + chunk_size < total_elements else 0)
        response = self.socket.recv()
        return response

    def send_in_chunk(self, trajectory, ct_img_filter, roi_radius,chunk_size=1024):
        self._send_metadata(trajectory,ct_img_filter,roi_radius)
        self._send_array_in_chunk(trajectory,chunk_size)
        res = self._send_array_in_chunk(ct_img_filter,chunk_size)
        counts = np.frombuffer(res, dtype=np.int32)

    def send(self, trajectory, ct_img, roi_radius, mask_radius):
        self._send_metadata(trajectory,ct_img,mask_radius)
        self._send_array(trajectory)
        img = self._send_array(ct_img) # expect to recieve a masked img
        if mask_radius==0:
            img = np.frombuffer(img,dtype=float)
            # img = img.reshape(trajectory.shape)
        else:
            img = np.frombuffer(img,dtype=int)
            img = img.reshape(ct_img.shape[1],ct_img.shape[0]).T
        # Pack metadata into a binary format: 4 integers
        # print(f"Sending roi_radius: [{roi_radius}] from {self.port}")
        self.socket.send(struct.pack('l', roi_radius))
        counts = self.socket.recv()
        counts = np.frombuffer(counts, dtype=int)


        return img, counts

    def __del__(self):
        self.socket.close()
        print(f"Port[{self.port}] closed")

def fill_nan_scipy(arr):
    x, y = np.meshgrid(np.arange(arr.shape[1]), np.arange(arr.shape[0]))
    points = np.vstack((x[~np.isnan(arr)].flatten(), y[~np.isnan(arr)].flatten())).T
    values = arr[~np.isnan(arr)].flatten()
    arr[np.isnan(arr)] = griddata(points, values, (x[np.isnan(arr)], y[np.isnan(arr)]), method='nearest')
    return arr

def fill_nan(arr):
    data = arr.copy()
    mask = np.isnan(data)
    data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
    return data
