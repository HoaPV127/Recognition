import tensorflow as tf
import numpy as np
import math
from model import *
from get_list_data import *

X_train_path,X_val_path,y_train,y_val, listdir = get_list_data('../../datasets/msceleb/extracted_images/')



saver = tf.train.Saver(max_to_keep=1)

with tf.Session() as sess:
	with tf.device('/gpu:0'):
                sess.run(initial_operator)
                
                saver.restore(sess, './save/our_model.ckpt')                

                print("Training")
                run_model(sess,y_out,X_train_path,y_train,listdir,10,128,optimizer)
		
                print("Validation")
                run_model(sess,y_out,X_val_path,y_val,listdir,1,128)

                print("saving variable")
                savePath = saver.save(sess, './save/our_model.ckpt')

