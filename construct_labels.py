import sys
import numpy as np
import scipy
import subprocess
from natsort import natsorted
sys.path.append("./utils")
import os
import string
from InterfaceDeclaration import LPBFData
from utils.preprocessing import MaPS_LPBF_Construction
import pickle

alphabet = list(string.ascii_lowercase)

if __name__ == "__main__":
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
    process_regime = [
        [0,60,     "Base"], # ignored
        [61, 130,  "GP"], # Gas Pore, Unstable keyhole, very hard to predict
        [131, 200, "NP"], # No Pore, Nominal Parameter
        [201, 270, "RLoF"], # Random lack of fusion
        [271, 340, "LoF"] # lack of Fusion, most easy to predict
    ]

    laser_power_setting = [
        [0,60,     "165"], # ignored
        [61, 130,  "110"], # Gas Pore, Unstable keyhole, very hard to predict
        [131, 200, "180"], # No Pore, Nominal Parameter
        [201, 270, "300"], # Random lack of fusion
        [271, 340, "165"] # lack of Fusion, most easy to predict
    ]

    scanning_speed_setting = [
        [0,60,     "900"], # ignored
        [61, 130,  "900"], # Gas Pore, Unstable keyhole, very hard to predict
        [131, 200, "900"], # No Pore, Nominal Parameter
        [201, 270, "900"], # Random lack of fusion
        [271, 340, "900"] # lack of Fusion, most easy to predict
    ]

    lpbf = MaPS_LPBF_Construction(daq_dir=daq_dir,lmq_dir=lmq_dir,sampling_rate_daq=sampling_rate_daq,sampling_rate_lmq=sampling_rate_lmq,process_regime=process_regime,laser_power_setting=laser_power_setting,scanning_speed_setting=scanning_speed_setting)

    lpbf._construct_line_wise_labels_ram()

    lpbf_data = LPBFData(
    context_info={
        "cube_position": lpbf.cube_position,
        "laser_power": lpbf.laser_power,
        "start_coord": lpbf.start_coord,
        "end_coord":   lpbf.end_coord,
        "scanning_speed": lpbf.scanning_speed,
        "regime_info": lpbf.regime_info,
    },
    defect_labels=lpbf.defect_labels,
    in_process_data={
        "microphone": lpbf.microphone,
        "AE": lpbf.ae,
        "photodiode": lpbf.photodiode,
        }
    )

    with open(os.path.join(os.path.dirname(daq_dir),'intermediate',f"lpbf_line_wise_data.pkl"), "wb") as fp:   
        pickle.dump(lpbf_data, fp)