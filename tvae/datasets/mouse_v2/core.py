import os
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch

from datasets import TrajectoryDataset
from tqdm import tqdm

from util.logging import LogEntry
from .preprocess import *
import pickle

class MouseV1Dataset(TrajectoryDataset):

    name = 'mouse_v1'

    # Default config
    _state_dim = 33 
    _action_dim = 33

    normalize_data = True

    def __init__(self, data_config):
        super().__init__(data_config)

    def _load_data(self):
        self.log = LogEntry()
        self._load_data_wrapper()

    def _load_data_wrapper(self):
        # Load in entire dataset
        self.train_states, self.train_actions, self.test_states, self.test_actions = self._load_and_preprocess()

    def _load_and_preprocess(self):
        data_train = self.config['data_train']
        data_test = self.config['data_test']

		# Compute states and actions
        train_states = data_train
        train_actions = train_states[:, 1:] - train_states[:, :-1]
        test_states = data_test
        test_actions = test_states[:, 1:] - test_states[:, :-1]

        # Update dimensions
        self._len_train = train_actions.shape[1]
        self._len_test = test_actions.shape[1]
        self._state_dim_train = train_states.shape[-1]
        self._action_dim_train = train_actions.shape[-1]
        self._state_dim_test = test_states.shape[-1]
        self._action_dim_test = test_actions.shape[-1]

        return train_states, train_actions, test_states, test_actions
