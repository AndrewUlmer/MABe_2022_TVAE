from .core import TrajectoryDataset
from .mouse_v1 import MouseV1Dataset
from .fly_v1 import FlyV1Dataset

dataset_dict = {
   	'mouse_v1' : MouseV1Dataset,
    'fly_v1' : FlyV1Dataset,
    'trajectory_v1' : TrajectoryDataset
}


def load_dataset(data_config):
    dataset_name = data_config['name'].lower()

    if dataset_name in dataset_dict:
        return dataset_dict[dataset_name](data_config)
    else:
        raise NotImplementedError
