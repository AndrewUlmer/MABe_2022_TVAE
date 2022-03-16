import numpy as np
import pickle
import torch
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

########### MOUSE DATASET FRAME WIDTH
FRAME_WIDTH_TOP = 570 
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
    # data shape is num_seq x 3 x 10 x 2
    
    data = data.reshape(data.shape[0], 2, 7 ,2)
    mice = [data[:,i,:,:] for i in range(2)]
    data = np.concatenate(mice, axis=0)

    del mice
    gc.collect()

    mouse_center = data[:, center_index, :]
    centered_data = data - mouse_center[:, np.newaxis, :]

	# Rotate such that keypoints 3 and 6 are parallel with the y axis
    mouse_rotation = np.arctan2(
		data[:, 3, 0] - data[:, 6, 0], data[:, 3, 1] - data[:, 6, 1])

    R = (np.array([[np.cos(mouse_rotation), -np.sin(mouse_rotation)],
				   [np.sin(mouse_rotation),  np.cos(mouse_rotation)]]).transpose((2, 0, 1)))

	# Encode mouse rotation as sine and cosine
    mouse_rotation = np.concatenate([np.sin(mouse_rotation)[:, np.newaxis], np.cos(
		mouse_rotation)[:, np.newaxis]], axis=-1)

    centered_data = np.matmul(R, centered_data.transpose(0, 2, 1))
    centered_data = centered_data.transpose((0, 2, 1))
    centered_data = centered_data.reshape((-1, 14))

    return centered_data, mouse_center, mouse_rotation

def transform_to_svd_components(data,
                                center_index=3,
                                n_components=5,
                                svd_computer=None,
                                mean=None,
                                stack_agents = False,
                                stack_axis = 1,
                                save_svd = True,
                                partial_svd=0.25):

    centered_data, mouse_center, mouse_rotation = rotate(data, center_index)

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
                replace = False
            )
        else:
            svd_idxs = range(centered_data.shape[0])
        svd_computer.fit(centered_data[svd_idxs])
        svd_data = svd_computer.transform(centered_data)
    else:
        svd_data = svd_computer.transform(centered_data)
        explained_variances = np.var(svd_data, axis=0) / np.var(centered_data, axis=0).sum()
    
    # Concatenate state as mouse center, mouse rotation and svd components
    data = np.concatenate([mouse_center, mouse_rotation, svd_data], axis=1)

    data = np.array_split(data, 2, axis=0)
    data = np.stack(data, axis=stack_axis)

    if save_svd:
        with open('./svd.pkl', 'wb') as f:
            pickle.dump(svd_computer, f)
        with open('./mean.pkl', 'wb') as f:
            pickle.dump(mean, f)

    return data, svd_computer, mean

def unrotate(mouse_kps, mouse_rotation, mouse_center):
    # Compute rotation angle from sine and cosine representation
    mouse_rotation = np.arctan2(
        mouse_rotation[:, 0], mouse_rotation[:, 1])

    mouse_kps = mouse_kps.reshape((-1, 7, 2))

    mouse_rotation = -1 * mouse_rotation

    R1 = np.array([[np.cos(mouse_rotation), -np.sin(mouse_rotation)],
              [np.sin(mouse_rotation),  np.cos(mouse_rotation)]]).transpose((2, 0, 1))

    centered_mouse_data = np.matmul(R1, mouse_kps.transpose(0, 2, 1))

    mouse_keypoints = centered_mouse_data + mouse_center[:, :, np.newaxis]
    mouse_keypoints = mouse_keypoints.transpose(0, 2, 1)

    return mouse_keypoints.reshape(-1, 14)

def transform_svd_to_keypoints(data, svd_computer, mean, stack_agents = False,
                            stack_axis = 1, num_components = 7):

    svd_components = [data[:,i,4:] for i in range(2)]
    mouse_centers = [data[:,i,:2] for i in range(2)]
    mouse_rotations = [data[:,i,2:4] for i in range(2)]

    mouse_kps = [svd_computer.inverse_transform(comps) + mean for comps in svd_components]
    mouse_kps = [unrotate(mouse_kp, mouse_rotation, mouse_center)
                  for mouse_kp, mouse_rotation, mouse_center in 
                  zip(mouse_kps, mouse_rotations, mouse_centers)]

    mouse_kps = np.concatenate(mouse_kps, axis=1)

    return mouse_kps

#def transform_svd_to_keypoints(data, svd_computer, mean, stack_agents = False,
#                            stack_axis = 1, num_components = 7):
#
#    data = data.reshape(-1, data.shape[-1])
#
#    mouse1_center = data[:,:2]
#    mouse2_center = data[:,11:13]
#    mouse3_center = data[:,22:24]
#
#    mouse1_rotation = data[:,2:4]
#    mouse2_rotation = data[:,13:15]
#    mouse3_rotation = data[:,24:26]
#
#    mouse1_components = data[:,4:11]
#    mouse2_components = data[:,15:22]
#    mouse3_components = data[:,26:]
#
#    mouse1_kps = svd_computer.inverse_transform(mouse1_components)
#    mouse2_kps = svd_computer.inverse_transform(mouse2_components)
#    mouse3_kps = svd_computer.inverse_transform(mouse3_components)
#
#    if mean is not None:
#        mouse1_kps = mouse1_kps + mean
#        mouse2_kps = mouse2_kps + mean
#        mouse3_kps = mouse3_kps + mean
#
#    # data = np.stack([mouse1_kps, mouse2_kps, mouse3_kps], axis=stack_axis)
#    # return data
#
#    # Compute rotation angle from sine and cosine representation
#    mouse1_rotation = np.arctan2(
#        mouse1_rotation[:, 0], mouse2_rotation[:, 1])
#
#    mouse2_rotation = np.arctan2(
#        mouse2_rotation[:, 0], mouse2_rotation[:, 1])
#
#    mouse3_rotation = np.arctan2(
#        mouse3_rotation[:, 0], mouse3_rotation[:, 1])
#
#    mouse1_kps = mouse1_kps.reshape((-1, 10, 2))
#    mouse2_kps = mouse2_kps.reshape((-1, 10, 2))
#    mouse3_kps = mouse3_kps.reshape((-1, 10, 2))
#
#    mouse1_rotation = -1 * mouse1_rotation
#    mouse2_rotation = -1 * mouse2_rotation
#    mouse3_rotation = -1 * mouse3_rotation
#
#    R1 = np.array([[np.cos(mouse1_rotation), -np.sin(mouse1_rotation)],
#              [np.sin(mouse1_rotation),  np.cos(mouse1_rotation)]]).transpose((2, 0, 1))
#
#    R2 = np.array([[np.cos(mouse2_rotation), -np.sin(mouse2_rotation)],
#              [np.sin(mouse2_rotation),  np.cos(mouse2_rotation)]]).transpose((2, 0, 1))
#
#    R3 = np.array([[np.cos(mouse3_rotation), -np.sin(mouse3_rotation)],
#              [np.sin(mouse3_rotation),  np.cos(mouse3_rotation)]]).transpose((2, 0, 1))
#
#    centered_mouse1_data = np.matmul(R1, mouse1_kps.transpose(0, 2, 1))
#    centered_mouse2_data = np.matmul(R2, mouse2_kps.transpose(0, 2, 1))
#    centered_mouse3_data = np.matmul(R3, mouse3_kps.transpose(0, 2, 1))
#
#    mouse1_keypoints = centered_mouse1_data + mouse1_center[:, :, np.newaxis]
#    mouse1_keypoints = mouse1_keypoints.transpose(0, 2, 1)
#
#    mouse2_keypoints = centered_mouse2_data + mouse2_center[:, :, np.newaxis]
#    mouse2_keypoints = mouse2_keypoints.transpose(0, 2, 1)
#
#    mouse3_keypoints = centered_mouse3_data + mouse3_center[:, :, np.newaxis]
#    mouse3_keypoints = mouse3_keypoints.transpose(0, 2, 1)
#
#    data = np.stack([mouse1_keypoints, mouse2_keypoints, mouse3_keypoints], axis=stack_axis)
#
#    return data
