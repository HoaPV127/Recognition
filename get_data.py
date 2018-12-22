import numpy as np
from PIL import Image

def get_data(list_data):
	path = '../../datasets/msceleb/extracted_images/'
	X_train = []

	for i in range(len(list_data)):
		img = Image.open(path + list_data[i])
		img = img.resize([256,256])
		img = img.convert("RGB")
		img_arr = np.asarray(img)
		img.close()
		X_train.append(img_arr)

	X_train = np.asarray(X_train)

	return X_train


