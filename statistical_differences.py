import sys
sys.path.append("./utls")
sys.path.append("./preprocessing")
from preprocessing import *
import os
import string
import subprocess
from natsort import natsorted
from InterfaceDeclaration import LPBFData
from utls.preprocessing import MaPS_LPBF_Construction
from construct_roi_adjacent_labels import fourier_transform1d_interp
import pickle
import numpy as np
import matplotlib.pyplot as plt

alphabet = list(string.ascii_lowercase)

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

sampling_rate_daq: int = int(1.25 * 1e6)
sampling_rate_lmq: int = int(0.1 * 1e6)
tdms_daq_list = natsorted(
    [i for i in os.listdir(daq_dir) if i.split(".")[-1] == "tdms"]
)
bin_lmq_list = natsorted([i for i in os.listdir(lmq_dir) if i.split(".")[-1] == "bin"])
lmq_channel_name = [
    "Vector ID",
    "meltpooldiode",
    "X Coordinate",
    "Y Coordinate",
    "Laser power",
    "Spare",
    "Laser diode",
    "Varioscan(focal length)",
]
with open(os.path.join(os.path.dirname(daq_dir),'intermediate',f"lpbf_line_wise_data.pkl"), "rb") as fp:   
    lpbf_data = pickle.load(fp)

target_freq = 600000
target_length = 6000
common_freqs = np.linspace(0, target_freq, target_length)
scanning_vector = np.asarray(np.round(lpbf_data.print_vector[1]),dtype=int)
uni = list(np.unique(scanning_vector))
uni.pop(2)

for _c,_u in enumerate(uni):
    _AE_i = (np.where(scanning_vector==_u)[0])
    _AE = [lpbf_data.AE[i] for i in _AE_i]
    _ft = fourier_transform1d_interp(_AE,sampling_rate_daq,target_freq=target_freq,target_length=target_length, verbose=False)
    fig,ax = plt.subplots()
    for i in (_ft[0]):
        ax.plot(common_freqs[10:]/1e6,(i[10:]**2)/1e6,lw=0.005,alpha=0.5, c=cm_std[0])
    ax.set_ylabel(f"Amplitude")
    ax.set_xlabel(f"Frequency (MHz)")
    # ax.set_ylim(-0.1,2.3)
    ax.set_ylim(0.9e-13,np.log(2.5))
    ax.set_yscale('log')
    plt.savefig(f"./outputs/raw_PSD_DirectionDiff_with_{_u}")
    plt.close()