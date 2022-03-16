import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from lib.models.core import TVAE
from util.logging import LogEntry 
from datasets import load_dataset
from datasets.mouse_v1.preprocess import unnormalize

import os
import random
import numpy as np
from tqdm import tqdm

def reconstruct(model, data_loader, device):
	"""
	This method will generate reconstructions for the data set contained in the
	passed data_loader. This function will generate num_reconstructions samples
	per input sample, and then select the one with the lowest NLL as the reconstruction
	to return. It will also return the mean of the predicted posterior from which the
	samples are generated, and the original sequence.

	Args:
		model (TVAE): trained TVAE model --> must be loaded in before calling reconstruct
		data_loader (PyTorch DataLoader) --> iterator containing the dataset we want to reconstruct
		device (PyTorch device) --> cpu or gpu to perform forward pass of the model on
		num_reconstructions (int) --> how many reconstructions per input sequence to generate

	"""
	# Don't track gradients 
	model = model.eval()
	log = LogEntry()
	
	reconstructions, embeddings, originals = [], [], []
	for batch_idx, (states, actions) in enumerate(tqdm(data_loader)):
		states = states.to(device).float()
		actions = actions.to(device).float()

		batch_log, reconstruction_set, embedding = model(
			states, 
			actions, 
			reconstruct = True,
			no_teacher_force = True
		)

		reconstructions.append(reconstruction_set.permute(1,0,2).detach().cpu().numpy())
		originals.append(states.detach().cpu().numpy())
		embeddings.append(embedding.squeeze().detach().cpu().numpy())

	# Remove batch dimension 
	reconstructions = np.concatenate(reconstructions, axis=0)
	originals = np.concatenate(originals, axis=0)
	embeddings = np.concatenate(embeddings, axis=0)

	return reconstructions, originals, embeddings
