from celluloid import Camera
import numpy as np
import matplotlib.pyplot as plt

def plot_reconstruction(orig_seq, recon_seq, intr_seq=None, path='test_seq.gif'):
	# Configure plot
	fig = plt.figure()
	camera = Camera(fig)
	plt.xlim([0,1024])
	plt.ylim([0, 570])

	# Just reconstructing one mouse for this series of experiments
	for idx in range(orig_seq.shape[0]):
		# Plot the faded original
		origx, origy = (orig_seq[idx,::2], orig_seq[idx,1::2])
	
		# Include legend in plot	
		if idx == 0:
			plt.scatter(x=origx, y=origy, c='#788eff', label='Original')
		else:
			plt.scatter(x=origx, y=origy, c='#788eff')
	
		# Connect lines between keypoints on original	
		origx = np.insert(origx,3,origx[0]).reshape(-1,1)
		origx = np.insert(origx,4,origx[2]).reshape(-1,1)
		origx = np.insert(origx,6,origx[1]).reshape(-1,1)
		origx = np.insert(origx,7,origx[5]).reshape(-1,1)
		origx = np.insert(origx,10,origx[5]).reshape(-1,1)
		origx = np.insert(origx,11,origx[8]).reshape(-1,1)
		origx = np.insert(origx,13,origx[9]).reshape(-1,1)

		origy = np.insert(origy,3,origy[0]).reshape(-1,1)
		origy = np.insert(origy,4,origy[2]).reshape(-1,1)
		origy = np.insert(origy,6,origy[1]).reshape(-1,1)
		origy = np.insert(origy,7,origy[5]).reshape(-1,1)
		origy = np.insert(origy,10,origy[5]).reshape(-1,1)
		origy = np.insert(origy,11,origy[8]).reshape(-1,1)
		origy = np.insert(origy,13,origy[9]).reshape(-1,1)

		plt.plot(origx, origy, c='#788eff')

		# Now do the same for the original, but with darker color
		reconx, recony = (recon_seq[idx,::2], recon_seq[idx,1::2])
		if idx == 0:
			plt.scatter(x=reconx, y=recony, c='b', label='Reconstruction')
		else:
			plt.scatter(x=reconx, y=recony, c='b')
	
		# Connect lines between keypoints on reconinal	
		reconx = np.insert(reconx,3,reconx[0]).reshape(-1,1)
		reconx = np.insert(reconx,4,reconx[2]).reshape(-1,1)
		reconx = np.insert(reconx,6,reconx[1]).reshape(-1,1)
		reconx = np.insert(reconx,7,reconx[5]).reshape(-1,1)
		reconx = np.insert(reconx,10,reconx[5]).reshape(-1,1)
		reconx = np.insert(reconx,11,reconx[8]).reshape(-1,1)
		reconx = np.insert(reconx,13,reconx[9]).reshape(-1,1)

		recony = np.insert(recony,3,recony[0]).reshape(-1,1)
		recony = np.insert(recony,4,recony[2]).reshape(-1,1)
		recony = np.insert(recony,6,recony[1]).reshape(-1,1)
		recony = np.insert(recony,7,recony[5]).reshape(-1,1)
		recony = np.insert(recony,10,recony[5]).reshape(-1,1)
		recony = np.insert(recony,11,recony[8]).reshape(-1,1)
		recony = np.insert(recony,13,recony[9]).reshape(-1,1)

		plt.plot(reconx, recony, c='b')

		# Now do the same for the original, but with darker color
		if intr_seq is not None:
			intrx, intry = (intr_seq[idx,:7], intr_seq[idx,7:])
			if idx == 0:
				plt.scatter(x=intrx, y=intry, c='r', label='Intruder')
			else:
				plt.scatter(x=intrx, y=intry, c='r')
		
			# Connect lines retween keypoints on intrinal	
			intrx = np.insert(intrx,3,intrx[0]).reshape(-1,1)
			intrx = np.insert(intrx,4,intrx[2]).reshape(-1,1)
			intrx = np.insert(intrx,6,intrx[1]).reshape(-1,1)
			intrx = np.insert(intrx,7,intrx[5]).reshape(-1,1)
			intrx = np.insert(intrx,10,intrx[5]).reshape(-1,1)
			intrx = np.insert(intrx,11,intrx[8]).reshape(-1,1)
			intrx = np.insert(intrx,13,intrx[9]).reshape(-1,1)

			intry = np.insert(intry,3,intry[0]).reshape(-1,1)
			intry = np.insert(intry,4,intry[2]).reshape(-1,1)
			intry = np.insert(intry,6,intry[1]).reshape(-1,1)
			intry = np.insert(intry,7,intry[5]).reshape(-1,1)
			intry = np.insert(intry,10,intry[5]).reshape(-1,1)
			intry = np.insert(intry,11,intry[8]).reshape(-1,1)
			intry = np.insert(intry,13,intry[9]).reshape(-1,1)

			plt.plot(intrx, intry, c='r')
		
		camera.snap()

	plt.legend()
	animation = camera.animate()
	animation.save('{}'.format(path), writer='imagemagick')

def plot_sequence(seq, fn='test'):
	# Assuming there are seven keypoints
	num_mice = int(seq.shape[-1] / 14)

	# Create figure settings	
	fig = plt.figure()
	camera = Camera(fig)
	plt.xlim([0,1024])
	plt.ylim([0, 570])

	for i, frame in enumerate(seq):
		# Separate coordinates
		m1x = frame[::2][:7]
		m1y = frame[1::2][:7]
			
		if num_mice == 2:
			m2x = frame[::2][7:]
			m2y = frame[1::2][7:]

		if i == 0:
			plt.scatter(x=m1x, y=m1y, c='b', label='Resident')
		
			if num_mice == 2:
				plt.scatter(x=m2x, y=m2y, c='r', label='Intruder')
		else:	
			plt.scatter(x=m1x, y=m1y, c='b')
			if num_mice == 2:
				plt.scatter(x=m2x, y=m2y, c='r')

		# Connect lines
		m1x = np.insert(m1x,3,m1x[0]).reshape(-1,1)
		m1x = np.insert(m1x,4,m1x[2]).reshape(-1,1)
		m1x = np.insert(m1x,6,m1x[1]).reshape(-1,1)
		m1x = np.insert(m1x,7,m1x[5]).reshape(-1,1)
		m1x = np.insert(m1x,10,m1x[5]).reshape(-1,1)
		m1x = np.insert(m1x,11,m1x[8]).reshape(-1,1)
		m1x = np.insert(m1x,13,m1x[9]).reshape(-1,1)

		m1y = np.insert(m1y,3,m1y[0]).reshape(-1,1)
		m1y = np.insert(m1y,4,m1y[2]).reshape(-1,1)
		m1y = np.insert(m1y,6,m1y[1]).reshape(-1,1)
		m1y = np.insert(m1y,7,m1y[5]).reshape(-1,1)
		m1y = np.insert(m1y,10,m1y[5]).reshape(-1,1)
		m1y = np.insert(m1y,11,m1y[8]).reshape(-1,1)
		m1y = np.insert(m1y,13,m1y[9]).reshape(-1,1)

		if num_mice == 2:
			m2x = np.insert(m2x,3,m2x[0]).reshape(-1,1)
			m2x = np.insert(m2x,4,m2x[2]).reshape(-1,1)
			m2x = np.insert(m2x,6,m2x[1]).reshape(-1,1)
			m2x = np.insert(m2x,7,m2x[5]).reshape(-1,1)
			m2x = np.insert(m2x,10,m2x[5]).reshape(-1,1)
			m2x = np.insert(m2x,11,m2x[8]).reshape(-1,1)
			m2x = np.insert(m2x,13,m2x[9]).reshape(-1,1)

			m2y = np.insert(m2y,3,m2y[0]).reshape(-1,1)
			m2y = np.insert(m2y,4,m2y[2]).reshape(-1,1)
			m2y = np.insert(m2y,6,m2y[1]).reshape(-1,1)
			m2y = np.insert(m2y,7,m2y[5]).reshape(-1,1)
			m2y = np.insert(m2y,10,m2y[5]).reshape(-1,1)
			m2y = np.insert(m2y,11,m2y[8]).reshape(-1,1)
			m2y = np.insert(m2y,13,m2y[9]).reshape(-1,1)

			plt.plot(m2x, m2y, c='r')

		plt.plot(m1x, m1y, c='b')

		camera.snap()

	plt.legend()
	animation = camera.animate()
	animation.save('{}.gif'.format(fn), writer='imagemagick')

if __name__ == "__main__":
	print('something')

