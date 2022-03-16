from comet_ml import Experiment

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from lib.models.core import TVAE
from util.logging import LogEntry 
from datasets import load_dataset

import os
import numpy as np
import random
from tqdm import tqdm

def train(model, data_loader, train_config, device):
	if train_config['start_model_path']:
		model.load_state_dict(torch.load(train_config['start_model_path']))
	else:
		pass
 
	# Training
	train_losses, val_losses = [], []
	for epoch in range(train_config['start_epoch'], train_config['num_epochs']):
		print('-=-=-= EPOCH {} OF {} =-=-=-'.format(epoch, train_config['num_epochs']))
		if epoch % train_config['num_epochs_til_val'] == 0:

			model = model.eval()
			data_loader.dataset.eval()

			torch.save(
				model.state_dict(),
				'./checkpoints/{}/{}.pt'.format(train_config['project_name'], epoch)
			)

		else:
			model = model.train()
			data_loader.dataset.train()
		
		log = LogEntry()
		for batch_idx, (states, actions) in enumerate(tqdm(data_loader)):
			
			states = states.to(device).float()
			actions = actions.to(device).float()

			batch_log = model(states, actions, no_teacher_force = True)

			if epoch % train_config['num_epochs_til_val'] == 0:	
				pass
			else:
				model.optimize(batch_log.losses)

			batch_log.itemize()
			log.absorb(batch_log)

		# Add metrics to experiment
		log.average(N=len(data_loader.dataset))
		if epoch % train_config['num_epochs_til_val'] == 0:
			with open(f'./checkpoints/{train_config["project_name"]}/val_losses.txt', 'a') as myFile:
				myFile.write(f'-=-= Validation losses @ epoch {epoch}\n')
				myFile.write(f'NLL: {log.to_dict()["losses"]["nll"]}\n')
				myFile.write(f'KLD: {log.to_dict()["losses"]["kl_div"]}\n')
		
		with open(f'./checkpoints/{train_config["project_name"]}/train_losses.txt','a') as myFile:
			myFile.write(f'-=-= Training losses @ epoch {epoch}\n')
			myFile.write(f'NLL: {log.to_dict()["losses"]["nll"]}\n')
			myFile.write(f'KLD: {log.to_dict()["losses"]["kl_div"]}\n')

	return train_losses, val_losses
