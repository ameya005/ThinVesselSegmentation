'''
Module : Patch Image
'''

from skimage.util import pad
from sklearn.feature_extraction import image as imext

def imgpad(img,patchsize):
	'''
	Pad the image around the edges with constant values.
	'''

	impad = pad(img, patchsize,'constant',constant_values=(0,0))

	return impad


def patchify(img,patch_size=(10,10)):
	'''
	Take an input image and create overlapping patches.

	Input : img
	Output : patch

	'''
	size = img.shape

	patch = imext.extract_patches_2d(img, patch_size)

	return patch,size

def unpatchify(patch,image_size):
	'''
	Construct image from patches.

	'''

	im = imext.reconstruct_from_patches_2d(patch,image_size)

	return im

def patch_image(img,patch_size=(10,10)):
	'''
	Take an input image and create overlapping patches.
	The patches are stored in an array.

	Input : img
	Output : patch

	'''
	#iw,ih = patch_size
	#img = imgpad(im,(iw/2,ih/2))
	size = img.shape

	patch = imext.extract_patches(img, patch_size)

	return patch,size

def reimg(patches,imgsize):
	'''
	Combine Image patches stored in array

	'''
	pw,ph = patches.shape[2:]
	iw,ih = imgsize

	img = np.zeros((iw,ih))

	nw = iw-pw+1
	nh = ih-ph+1

	for i in range(nw):
		for j in range(nh):
			img[i:i+pw,j:j+ph] += patches[i,j]

	return img

