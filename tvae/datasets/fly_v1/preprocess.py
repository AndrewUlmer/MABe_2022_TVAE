import numpy as np
import pickle
import torch
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

########### MOUSE DATASET FRAME WIDTH
FRAME_WIDTH_TOP = 1024 
FRAME_HEIGHT_TOP = 1024 

def normalize(data):
    """Scale by dimensions of image and mean-shift to center of image."""
    state_dim = data.shape[1] // 2
    shift = [int(FRAME_WIDTH_TOP / 2), int(FRAME_HEIGHT_TOP / 2)] * state_dim
    scale = [int(FRAME_WIDTH_TOP / 2), int(FRAME_HEIGHT_TOP / 2)] * state_dim
    
    return np.divide(data - shift, scale)

def unnormalize(data):
    """Undo normalize.
	expects input data to be [sequence length, coordinates alternating between x and y]
	"""
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    
    state_dim = data.shape[1] // 2
    
    x_shift = int(FRAME_WIDTH_TOP / 2)
    y_shift = int(FRAME_HEIGHT_TOP / 2)
    x_scale = int(FRAME_WIDTH_TOP / 2)
    y_scale = int(FRAME_HEIGHT_TOP / 2)

    data[:,::2] = data[:,::2]*x_scale + x_shift
    data[:,1::2] = data[:,1::2]*y_scale + y_shift

    return data.reshape(-1, state_dim*2)

def rotate(data, center_index):
    # data shape is num_seq x 11 x 19 x 2
    data = data.reshape(data.shape[0], 11, 19, 2)
    flies = [data[:,i,:,:] for i in range(11)]
    data = np.concatenate(flies, axis=0)

    del flies
    gc.collect()
    
    fly_center = data[:, center_index, :]
    centered_data = data - fly_center[:, np.newaxis, :]

	# Rotate such that keypoints 3 and 6 are parallel with the y axis
    fly_rotation = np.arctan2(
        data[:, 7, 0] - data[:, 8, 0], data[:, 7, 1] - data[:, 8, 1])

    R = (np.array([[np.cos(fly_rotation), -np.sin(fly_rotation)],
            [np.sin(fly_rotation),  np.cos(fly_rotation)]]).transpose((2, 0, 1)))

	# Encode mouse rotation as sine and cosine
    fly_rotation = np.concatenate([np.sin(fly_rotation)[:, np.newaxis], np.cos(
        fly_rotation)[:, np.newaxis]], axis=-1)

    centered_data = np.matmul(R, centered_data.transpose(0, 2, 1))
    centered_data = centered_data.transpose((0, 2, 1))
    centered_data = centered_data.reshape((-1, 38))

    return centered_data, fly_center, fly_rotation

def transform_to_svd_components(data,
                                center_index=7,
                                n_components=10,
                                svd_computer=None,
                                mean=None,
                                stack_agents = False,
                                stack_axis = 1,
                                save_svd = True,
                                partial_svd = 0.25):

    centered_data, fly_center, fly_rotation = rotate(data, center_index)

    if mean is None:
        mean = np.mean(centered_data, axis=0)
    centered_data = centered_data - mean

    # Compute SVD components
    if svd_computer is None:
        svd_computer = TruncatedSVD(n_components=n_components)
        if partial_svd:
            svd_idxs = np.random.choice(
                range(centered_data.shape[0]),
                int(partial_svd * centered_data.shape[0]),
                replace=False
            )
        else:
            svd_indxs = range(centered_data.shape[0])
        svd_computer.fit(centered_data[svd_idxs])
        svd_data = svd_computer.transform(centered_data)
    else:
        svd_data = svd_computer.transform(centered_data)
        explained_variances = np.var(svd_data, axis=0) / np.var(centered_data, axis=0).sum()
    
    # Concatenate state as mouse center, mouse rotation and svd components
    data = np.concatenate([fly_center, fly_rotation, svd_data], axis=1)
    
    data = np.array_split(data, 11, axis=0)
    data = np.stack(data, axis=stack_axis)
    
    if save_svd:
        with open('./svd.pkl', 'wb') as f:
            pickle.dump(svd_computer, f)
        with open('./mean.pkl', 'wb') as f:
            pickle.dump(mean, f)

    return data, svd_computer, mean

def unnormalize_keypoint_center_rotation(keypoints, center, rotation):

    keypoints = keypoints.reshape((-1, 7, 2))

    # Apply inverse rotation
    rotation = -1 * rotation
    R = np.array([[np.cos(rotation), -np.sin(rotation)],
                  [np.sin(rotation),  np.cos(rotation)]]).transpose((2, 0, 1))
    centered_data = np.matmul(R, keypoints.transpose(0, 2, 1))

    keypoints = centered_data + center[:, :, np.newaxis]
    keypoints = keypoints.transpose(0, 2, 1)

    return keypoints.reshape(-1, 14)

def unrotate(fly_kps, fly_rotation, fly_center):
    # Compute rotation angle from sine and cosine representation
    fly_rotation = np.arctan2(
        fly_rotation[:, 0], fly_rotation[:, 1])

    fly_kps = fly_kps.reshape((-1, 19, 2))

    fly_rotation = -1 * fly_rotation

    R1 = np.array([[np.cos(fly_rotation), -np.sin(fly_rotation)],
              [np.sin(fly_rotation),  np.cos(fly_rotation)]]).transpose((2, 0, 1))

    centered_fly_data = np.matmul(R1, fly_kps.transpose(0, 2, 1))

    fly_keypoints = centered_fly_data + fly_center[:, :, np.newaxis]
    fly_keypoints = fly_keypoints.transpose(0, 2, 1)

    return fly_keypoints.reshape(-1, 38)

def transform_svd_to_keypoints(data, svd_computer, mean, stack_agents = False,
                            stack_axis = 1, num_components = 7, single=False):

    if single:
        svd_components = [data[:,i,4:] for i in range(1)]
        fly_centers = [data[:,i,:2] for i in range(1)]
        fly_rotations = [data[:,i,2:4] for i in range(1)]
    else:
        svd_components = [data[:,i,4:] for i in range(11)]
        fly_centers = [data[:,i,:2] for i in range(11)]
        fly_rotations = [data[:,i,2:4] for i in range(11)]
    
    fly_kps = [svd_computer.inverse_transform(comps) + mean for comps in svd_components]
    fly_kps = [unrotate(fly_kp, fly_rotation, fly_center)
                  for fly_kp, fly_rotation, fly_center in 
                  zip(fly_kps, fly_rotations, fly_centers)]

    fly_kps = np.concatenate(fly_kps, axis=1)

    return fly_kps 
