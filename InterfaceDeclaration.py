from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Protocol, List, Tuple
import numpy as np
import math

@dataclass
class LPBFInterface(ABC):
    """
    - Context info: dict
                - Cube position (the n-th cube)
                - Laser power
                - start coord
                - end coord
                - Scaning Speed (remain constant in this case)
                - Scanning vector (will infer from coord)
                - Regime info (4 - class classifications)
                    - Base | Base plate, ignored
                    - GP   | Gas pore, or keyhole 
                    - NP   | No pore
                    - RLoF | Random lack of fusion
                    - LoF  | Lack of fusion

            - Defect labels: list[str]
                    - NP   | No pore
                    - GP   | Gas pore, or keyhole 
                    - RLoF | Random lack of fusion
                    - LoF  | Lack of fusion
            
            - In-process data: dict
                - microphone
                - AE
                - photodiode
    """
    context_info: dict
    defect_labels: list[int]
    in_process_data: dict

    @property
    def cube_position(self) -> list[int]:
        return self.context_info["cube_position"]

    @property
    def laser_power(self) -> list[float]:
        return self.context_info["laser_power"]

    @property
    def start_coord(self) -> list[float]:
        return self.context_info["start_coord"]

    @property
    def end_coord(self) -> list[float]:
        return self.context_info["end_coord"]

    @property
    def scanning_speed(self) -> list[float]:
        return self.context_info["scanning_speed"]

    @property
    def _calculate_vector(self) -> list[str]:
        return self._get_direction

    @property
    def regime_info(self) -> dict[str]:
        return self.context_info["regime_info"]

    @property
    def microphone(self) -> list[np.asarray]:
        return self.in_process_data["microphone"]

    @property
    def AE(self) -> list[np.asarray]:
        return self.in_process_data["AE"]

    @property
    def photodiode(self) -> list[np.asarray]:
        return self.in_process_data["photodiode"]

    @property
    def _get_direction(self) -> tuple[(float,float)]:
        import dask.array as da
        """
        Infer the direction and magnitude (the speed) from coordinate
        """
        start_coords_dask = da.from_array(self.context_info["start_coord"])
        end_coords_dask = da.from_array(self.context_info["end_coord"])
        delta_coords_dask = end_coords_dask - start_coords_dask
        distances_dask = np.linalg.norm(delta_coords_dask, axis=1)
        degrees_dask = np.degrees(np.arctan2(delta_coords_dask[:, 1], delta_coords_dask[:, 0]))

        distances, directions = distances_dask.compute(), degrees_dask.compute()

        return distances, directions
        # # if magnitude > 0:
        # #     direction /= magnitude
        # return degree, magnitude

    @property
    def _construct_dataset(self,context_info,defect_labels,in_process_data):
        self.context_info = context_info
        self.defect_labels = defect_labels
        self.in_process_data = in_process_data

class LPBFData(LPBFInterface):
    def __init__(self,context_info,defect_labels,in_process_data):
        self.context_info = context_info
        self.defect_labels = defect_labels
        self.in_process_data = in_process_data
        self.print_vector = self._calculate_vector

if __name__ =="__main__":
    lpbf_data = LPBFData(
    context_info={
        "cube_position": [1,2,3,4],
        "laser_power": [100,200,300,400],
        "start_coord": [(-200,100),( 200,100),( 100,-100),( 100,100)],
        "end_coord":   [( 200,100),(-200,100),(-100, 100),(-100,-100)],
        "scanning_speed": [10,20,30,40],
        "regime_info": ["GP","NP","RLoF","LoF"],
    },
    defect_labels=["GP", "NP","RLoF","LoF"],
    in_process_data={
        "microphone": [
            [0.1,0.2],
            [0.1,0.2,0.3],
            [0.1,0.2,0.3,0.4],
            [0.1,0.2,0.3,0.4,0.5],
            ],
        "AE": [
            [0.1,0.2],
            [0.1,0.2,0.3],
            [0.1,0.2,0.3,0.4],
            [0.1,0.2,0.3,0.4,0.5],
            ],
        "photodiode": [
            [0.1,0.2],
            [0.1,0.2,0.3],
            [0.1,0.2,0.3,0.4],
            [0.1,0.2,0.3,0.4,0.5],
            ],
        }
    )

    print(f"Position:\n{lpbf_data.cube_position}")  
    print(f"Regime info:\n{lpbf_data.regime_info}")    
    print(f"microphone:\n{lpbf_data.microphone}")     
    # print(f"scaning distances:\n{lpbf_data._calculate_vector[0]}") 
    # print(f"scaning directions:\n{lpbf_data._calculate_vector[1]}") 
    print(f"scaning vector:\n{lpbf_data.print_vector}") 