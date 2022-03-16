import random
import torch
from torch.utils.data import Dataset

TRAIN = 1
EVAL = 2

class TrajectoryDataset(Dataset):

    # Default parameters
    mode = TRAIN
    subsample = 1

    _state_dim = 0
    _action_dim = 0
    _seq_len = 0

    def __init__(self, data_config):
        assert hasattr(self, 'name')

        # Check if trajectories will be subsampled
        if 'subsample' in data_config:
            assert isinstance(data_config['subsample'], int) and data_config['subsample'] > 0
            self.subsample = data_config['subsample']

        self.config = data_config
        self.summary = {'name' : self.name}

        # Load data (and true labels, if any)
        self._load_data()

        # Assertions for train data
        assert hasattr(self,'train_states') 
        assert hasattr(self,'train_actions')
        assert self.train_states.shape[0] == self.train_actions.shape[0]
        assert self.train_states.shape[1]-1 == self.train_actions.shape[1]

        # Assertions for in_test data
        assert hasattr(self, 'test_states') 
        assert hasattr(self, 'test_actions')
        assert self.test_states.shape[0] == self.test_actions.shape[0]
        assert self.test_states.shape[1]-1 == self.test_actions.shape[1]#  == self.seq_len

        self.states = { TRAIN : self.train_states, EVAL : self.test_states }
        self.actions = { TRAIN : self.train_actions, EVAL : self.test_actions }

    def __len__(self):
        return self.states[self.mode].shape[0]

    def __getitem__(self, index):
        states = self.states[self.mode][index]
        actions = self.actions[self.mode][index]

        return states, actions 

    @property
    def seq_len(self):
        assert self._seq_len > 0
        return self._seq_len

    @property
    def state_dim(self):
        assert self._state_dim > 0
        return self._state_dim

    @property
    def action_dim(self):
        assert self._action_dim > 0
        return self._action_dim
        
    def _load_data(self):
        raise NotImplementedError

    def train(self):
        self.mode = TRAIN

    def eval(self):
        self.mode = EVAL
