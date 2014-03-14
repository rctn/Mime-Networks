import random
import array
import numpy as np
import glob

class vanHatData:
	def __init__(self, patch_width=8, ntrain=1e6, nholdout = 10000, img_path=None, base_fname=''):
		"""
		Load vanHateren dataset with number of training points and holdout points outlined above.

		img_path -- Path to vanHateren images.
		base_fname -- Base filename to save extracted patches.
		"""

		self.patch_width = patch_width
		self.patch_size = self.patch_width**2
		self.ntrain = int(ntrain)
		self.nholdout = int(nholdout)

		try:
			print "trying to load data ..."
			self.X = self.open_datafiles(base_fname + 'data_', self.ntrain)
			print "...data loaded"
		except:
			print "...could not load.  generating new dataset..."
			self.X = self.open_datafiles(base_fname + 'data_', self.ntrain, mode='w+')
			self.generate_X(ntrain, self.X, img_path=img_path)
			print "...data generated"

		try:
			print "trying to load holdout data ..."
			self.X_holdout = self.open_datafiles(base_fname + 'holdout_', self.nholdout)
			print "...holdout data loaded"
		except:
			print "...could not load holdout data.  generating new dataset..."
			self.X_holdout = self.open_datafiles(base_fname + 'holdout_', self.nholdout, mode='w+')
			self.generate_X(nholdout, self.X_holdout, img_path=img_path)
			print "...holdout generated"

	def get_data_range(self, i0, i1, holdout=False):
		"""
		Return sample points between i0 and i1.
		"""

		if not holdout:
			X = self.X
		else:
			X = self.X_holdout
		i0 = min(i0, X.shape[0])
		i1 = min(i1, X.shape[0])
		return X[i0:i1].copy()

	def open_datafiles(self, base_fname, ntrain, mode='r'):
		"""
		Open a memmap file for the target data
		"""

		fname = base_fname + "patchwidth%d_ntrain%d.memmap"%(
			self.patch_width, ntrain)
		shape = (ntrain, self.patch_width**2,)
		X = np.memmap(fname, dtype='float32', mode=mode, shape=shape)
		return X

	def generate_X(self, ntrain, X, ntrain_per_batch = 1e6, images_per_batch=100, img_path=None):
		if img_path is None:
			img_list= glob.glob('/Users/jascha/data/images/vanHateren/iml00001-04212/*.iml')
			if len(img_list) == 0:
				img_list= glob.glob('/data/images/vanHateren/iml00001-04212/*.iml')
		else:
			img_list= glob.glob(img_path + '*.iml')
		if len(img_list) == 0:
			throw("no image files, using random data")

		images_per_batch = int(images_per_batch)
		ntrain_per_batch = int(ntrain_per_batch)

		i0 = 0
		i1 = min(ntrain_per_batch, ntrain)
		for ni in range(int(np.ceil(ntrain/ntrain_per_batch))):
			print "%d / %d"%(i0, ntrain)
			fnames = random.sample(img_list, images_per_batch)
			X[i0:i1,:] = self.load_from_images(fnames, i1-i0)
			i0 = i1
			i1 = i0 + ntrain_per_batch
			if i1 > ntrain:
				i1 = ntrain
			if i1 <= i0:
				break

	def load_from_images(self, fnames, n, downsample_stride=5):
		im_shape = (1024, 1536)
		nimgs = len(fnames)
		imgs = np.zeros((im_shape[0], im_shape[1], nimgs), dtype='uint16')

		for ii, fname in enumerate(fnames):
			with open(fname, 'rb') as handle:
			   s = handle.read()
			arr = array.array('H', s)
			arr.byteswap()
			imgs[:,:,ii] = np.array(arr, dtype='uint16').reshape(im_shape)

		# downsample to avoid any high frequency rolloff
		# which would make it hard for independent sampling
		# of each pixel to mix
		# (note, we explicitly DO NOT want to smooth first,
		# because that would reintroduce the rolloff)
		# TODO -- For some (non-dpm) uses, may want to smooth first.
		imgs = imgs[::downsample_stride,::downsample_stride,:]

		# preprocess
		imgs = np.log(imgs+1)
		imgs -= np.mean(np.mean(imgs, axis=0), axis=0).reshape((1,1,nimgs))
		imgs /= np.sqrt(np.mean(np.mean(imgs**2, axis=0), axis=0)).reshape((1,1,nimgs))
		# square it
		min_dim = np.min(imgs.shape[:2])
		imgs = imgs[:min_dim, :min_dim, :]

		# fourier whiten
		Fimgs = np.fft.fft2(imgs)
		freqs = np.fft.fftfreq(imgs.shape[0])
		freq_pow = freqs.reshape((-1,1))**2 + freqs.reshape((1,-1))**2
		scl = np.sqrt(freq_pow)
		#scl[~np.isfinite(scl)] = 0.
		Fimgs *= scl.reshape((scl.shape[0], scl.shape[1], 1))
		imgs_white = np.real(np.fft.ifft2(Fimgs))

		# try:
			# import matplotlib
			# import matplotlib.pyplot as plt
		# 	img_idx = np.random.randint(nimgs)
		# 	plt.figure(100)
		# 	plt.clf()
		# 	plt.imshow(imgs[:,:,img_idx].copy(), cmap=cm.Greys_r)
		# 	plt.colorbar()
		# 	plt.figure(101)
		# 	plt.clf()
		# 	plt.imshow(imgs_white[:,:,img_idx].copy(), cmap=cm.Greys_r)
		# 	plt.colorbar()
		# 	plt.draw()
		# except:
		# 	print "plotting failed"
		# 	pass
		# 1./0

		imgs = imgs_white
		imgs -= np.mean(np.mean(imgs, axis=0), axis=0).reshape((1,1,nimgs))
		imgs /= np.sqrt(np.mean(np.mean(imgs**2, axis=0), axis=0)).reshape((1,1,nimgs))

		# make it too long to make sure we have enough space
		X = np.zeros((self.patch_width, self.patch_width, n))

		num_pixels = np.prod(im_shape)

		for patch_i in xrange(n):
			ii = np.random.randint(imgs.shape[0]-self.patch_width)
			jj = np.random.randint(imgs.shape[1]-self.patch_width)
			kk = np.random.randint(imgs.shape[2])
			X[:,:,patch_i] = imgs[ii:ii+self.patch_width,jj:jj+self.patch_width,kk]

		return X.reshape(self.patch_size,-1).T
