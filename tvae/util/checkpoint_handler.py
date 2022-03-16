import os
import numpy as np
from glob import glob

def sort_checkpoints(checkpoints):
	epox_idx = np.argsort(np.array([strip_epoch(pt) for pt in checkpoints]))
	return checkpoints[epox_idx[-1]]

def checkpoint_handler(train_config):
    if not os.path.exists('./checkpoints'):
        print('-=-= Creating directory to store tvae project checkpoints =-=-')
        os.mkdir('./checkpoints')

    proj_dir = os.path.join('./checkpoints/', train_config['project_name'])
    # Check if the project directory exists withinn checkpoints
    if os.path.exists(proj_dir):
        # Find most recent checkpoint
        pts = glob("{}/*.pt".format(proj_dir))
		# Sort paths
        train_config['start_model_path'] = sort_checkpoints(pts)
        train_config['start_epoch'] = strip_epoch(train_config['start_model_path'])
    else:
        # Make project directory, and start training from 0
        os.mkdir('./checkpoints/{}'.format(train_config['project_name']))
        train_config['start_epoch'] = 0
        train_config['start_model_path'] = None
        
    return train_config 

def strip_epoch(model_path):
    return int(model_path.split('/')[-1].split('.')[0])

