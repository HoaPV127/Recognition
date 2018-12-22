import os
import numpy as np
from PIL import Image

def duplicate(value, n):
	lst = []
	for j in range(n):
		lst.append(value)
	
	return lst

def get_list_data(PATH):

	list_dir = os.listdir(PATH)
	list_dir = list_dir[1:len(list_dir)]
	list_image = []

	X_train_path = []
	y_train = []
	X_val_path = []
	y_val = []
	
	for d in range(0,len(list_dir)):
		path = PATH + list_dir[d] + '/'
		ls_img = os.listdir(path)
		for i in range(len(ls_img)):
			if(i<4):
				X_val_path.append(list_dir[d] + '/' + ls_img[i])
				y_val.append(d)
			else:
				X_train_path.append(list_dir[d] + '/' + ls_img[i])
				y_train.append(d)

	y_train = np.asarray(y_train)
	y_val = np.asarray(y_val)
	X_train_path = np.asarray(X_train_path)
	X_val_path = np.asarray(X_val_path)
	list_dir = np.asarray(list_dir)	
	return X_train_path, X_val_path, y_train, y_val, list_dir

"""
link = '../../datasets/msceleb/extracted_images/'

X_train_path, X_val_path, y_train, y_val, listdir= get_list_data(link)

print("X_train: ", X_train_path.shape[0])
print("y_train: ", y_train.shape[0])
print("X_validate: ", X_val_path[0])
print("y_validate: ", y_val.shape[0])

test = np.random.randint(0,5000,size=(10,))
print(test)

print(listdir[y_train[test]], X_train_path[test])
"""





