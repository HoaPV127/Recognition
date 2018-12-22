import tensorflow as tf
import numpy as np
import math
import timeit
from get_data import *
from get_list_data import *

# clear old variables
tf.reset_default_graph()

# input 
X = tf.placeholder(tf.float32, [None, 256, 256, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)
C = 5030	# number of class


Wc1 = tf.get_variable("Wc1", shape=[7, 7, 3, 16], initializer=tf.contrib.layers.xavier_initializer())
bc1 = tf.get_variable("bc1", shape=[16], initializer=tf.contrib.layers.xavier_initializer())

gamma1 = tf.get_variable("gamma1", shape=[16], initializer=tf.contrib.layers.xavier_initializer())
beta1 = tf.get_variable("beta1", shape=[16], initializer=tf.contrib.layers.xavier_initializer())

Wc2a = tf.get_variable("Wc2a", shape=[1,1,16,16], initializer=tf.contrib.layers.xavier_initializer())
bc2a = tf.get_variable("bc2a", shape=[16], initializer=tf.contrib.layers.xavier_initializer())

Wc2 = tf.get_variable("Wc2", shape=[3,3,16,32], initializer=tf.contrib.layers.xavier_initializer())
bc2 = tf.get_variable("bc2", shape=[32], initializer=tf.contrib.layers.xavier_initializer())

gamma2 = tf.get_variable("gamma2", shape=[32], initializer=tf.contrib.layers.xavier_initializer())
beta2 = tf.get_variable("beta2", shape=[32], initializer=tf.contrib.layers.xavier_initializer())

Wc3a = tf.get_variable("Wc3a", shape=[1,1,32,32], initializer=tf.contrib.layers.xavier_initializer())
bc3a = tf.get_variable("bc3a", shape=[32], initializer=tf.contrib.layers.xavier_initializer())

Wc3 = tf.get_variable("Wc3", shape=[3,3,32,64], initializer=tf.contrib.layers.xavier_initializer())
bc3 = tf.get_variable("bc3", shape=[64], initializer=tf.contrib.layers.xavier_initializer())

Wc4a = tf.get_variable("Wc4a", shape=[1,1,64,64], initializer=tf.contrib.layers.xavier_initializer())
bc4a = tf.get_variable("bc4a", shape=[64], initializer=tf.contrib.layers.xavier_initializer())

Wc4 = tf.get_variable("Wc4", shape=[3,3,64,128], initializer=tf.contrib.layers.xavier_initializer())
bc4 = tf.get_variable("bc4", shape=[128], initializer=tf.contrib.layers.xavier_initializer())

Wc5a = tf.get_variable("Wc5a", shape=[1,1,128,128], initializer=tf.contrib.layers.xavier_initializer())
bc5a = tf.get_variable("bc5a", shape=[128], initializer=tf.contrib.layers.xavier_initializer())

Wc5 = tf.get_variable("Wc5", shape=[3,3,128,256], initializer=tf.contrib.layers.xavier_initializer())
bc5 = tf.get_variable("bc5", shape=[256], initializer=tf.contrib.layers.xavier_initializer())

Wc6a = tf.get_variable("Wc6a", shape=[1,1,256,256], initializer=tf.contrib.layers.xavier_initializer())
bc6a = tf.get_variable("bc6a", shape=[256], initializer=tf.contrib.layers.xavier_initializer())

Wc6 = tf.get_variable("Wc6", shape=[3,3,256,512], initializer=tf.contrib.layers.xavier_initializer())
bc6 = tf.get_variable("bc6", shape=[512], initializer=tf.contrib.layers.xavier_initializer())

W1 = tf.get_variable("W1", shape=[4608,2048], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable("b1", shape=[2048], initializer=tf.contrib.layers.xavier_initializer())

W2 = tf.get_variable("W2", shape=[2048, 5030], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable("b2", shape=[5030], initializer=tf.contrib.layers.xavier_initializer())

def my_model(X):

    #------------------ 1st layer ------------------#
    # Convolution + ReLu
    a1 = tf.nn.conv2d(X, Wc1, strides=[1,2,2,1], padding='VALID') + bc1
    h1 = tf.nn.relu(a1)
    print(h1) 
    # Max-Pooling
    pool1 = tf.nn.max_pool(h1, ksize=[1,3,3,1], strides=[1,2,2,1],padding='VALID')
    
    # Batch Norm
    mean1, var1 = tf.nn.moments(pool1, [0])
    esp = 1e-5
    bn1 = tf.nn.batch_normalization(pool1, mean1, var1, beta1, gamma1, esp)
    
    #------------------ 2nd layer ------------------#
    # Conv + Conv + ReLu
    a2a = tf.nn.conv2d(bn1, Wc2a, strides=[1,1,1,1], padding='VALID') +bc2a
    h2a = tf.nn.relu(a2a)
    
    a2 = tf.nn.conv2d(h2a, Wc2, strides=[1,1,1,1], padding='VALID') + bc2
    h2 = tf.nn.relu(a2)
    print(h2)

    # Batch Norm
    mean2, var2 = tf.nn.moments(h2, [0])
    bn2 = tf.nn.batch_normalization(h2, mean2, var2, beta2, gamma2, esp)
    
    # Max Pooling
    pool2 = tf.nn.max_pool(bn2, ksize=[1,3,3,1], strides=[1,2,2,1],padding='VALID')
    
    #------------------ 3rd layer ------------------#
    # Conv + Conv + ReLu
    a3a = tf.nn.conv2d(pool2, Wc3a, strides=[1,1,1,1], padding='VALID') + bc3a
    h3a = tf.nn.relu(a3a)
    
    a3 = tf.nn.conv2d(h3a, Wc3, strides=[1,1,1,1], padding='VALID') + bc3
    h3 = tf.nn.relu(a3)
    print(h3)
        
    # Max Pooling
    pool3 = tf.nn.max_pool(h3, ksize=[1,3,3,1], strides=[1,2,2,1],padding='VALID')
    
    #------------------ 4th layer ------------------#
    # Conv + Conv + ReLu
    a4a = tf.nn.conv2d(pool3, Wc4a, strides=[1,1,1,1], padding='VALID') + bc4a
    h4a = tf.nn.relu(a4a)
    
    a4 = tf.nn.conv2d(h4a, Wc4, strides=[1,1,1,1], padding='VALID') + bc4
    h4 = tf.nn.relu(a4)
    print(h4) 
    
    #------------------ 5th layer ------------------#
    # Conv + Conv + ReLu
    a5a = tf.nn.conv2d(h4, Wc5a, strides=[1,1,1,1], padding='VALID') + bc5a
    h5a = tf.nn.relu(a5a)

    a5 = tf.nn.conv2d(h5a, Wc5, strides=[1,1,1,1], padding='VALID') + bc5
    h5 = tf.nn.relu(a5)
    print(h5)

    #------------------ 6th layer ------------------#
    # Conv + Conv + ReLu
    a6a = tf.nn.conv2d(h5, Wc6a, strides=[1,1,1,1], padding='VALID') + bc6a
    h6a = tf.nn.relu(a6a)
    
    a6 = tf.nn.conv2d(h6a, Wc6, strides=[1,1,1,1], padding='VALID') + bc6
    h6 = tf.nn.relu(a6)
    print(h6)                          
    # Pooling 
    pool6 = tf.nn.max_pool(h6, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    print(pool6)                      
    # flat
    flat = tf.reshape(pool6, [-1,4608])
    print(flat)    

    # hidden1 
    hid1 = tf.matmul(flat,W1) + b1
    relu1 = tf.nn.relu(hid1)
    print(relu1)

    # hidden2
    hid2 = tf.matmul(relu1,W2) + b2
    print(hid2)
    # softmax
    soft_max = tf.nn.softmax(hid2)
                          
    y = soft_max
    return y


# compute prediction
y_out = my_model(X)		

print("*******************************************")

# losses
onehot = tf.one_hot(y,5030)
total_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=onehot, logits = y_out)
mean_loss = tf.reduce_mean(total_loss)

#optimixer
optimizer = tf.train.AdamOptimizer(0.3).minimize(mean_loss) # AdamOptimizer with learning rate = 5e-4

initial_operator = tf.global_variables_initializer()



def run_model(session, predict, X_path, y_,listdir, epochs=1, batch_size=64, training=None):
    # compute Accuracy
    corr_pred = tf.equal(tf.argmax(predict,1), y)
    accuracy = tf.reduce_mean(tf.cast(corr_pred, tf.float32))
    predict = tf.argmax(predict,1) 
    #shuffle indicies
    train_indicies = np.arange(X_path.shape[0])
    np.random.shuffle(train_indicies)
    
    training_now = training is not None
    
    # setting up variables to train
    variables = [mean_loss, corr_pred, accuracy]
    if training_now:
        variables[-1] = training

        
    for e in range(epochs):
        # count
        train_sample = 0    
        
        correct = 0
        losses = []
        
        for i in range(int(math.ceil(X_path.shape[0]/batch_size))):
            start_idx = (i*batch_size)%X_path.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # get data train
            X_ = get_data(X_path[idx])
            
            # create a feed_dict
            feed_dict = {X: X_, y: y_[idx], is_training: training_now}
            
            # get actual batch size
            actual_batch_size = y_[idx].shape[0]
            
            # compute losses and correct predictions
            loss, corr, _ = session.run(variables, feed_dict=feed_dict)
            pred = session.run(predict, feed_dict=feed_dict)
            #print(pred)
            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)
            
            train_sample += 128 ;
            # print process
            if training_now and (train_sample % 6400)==0:
                print("Epoch({3}/{4}) - Process: {0}% - training loss = {1:.3g} and accuracy of {2:.2g}"\
                      .format(round(100*train_sample/y_.shape[0]),loss,np.sum(corr)/actual_batch_size,e+1,epochs))

                #print(pred)
                #print(y_[idx])	
                #print("**************************************************************")	
        total_correct = correct/y_.shape[0]
        total_loss = np.sum(losses)/y_.shape[0]
            
        # print Epoch
        print("Epoch {0} - Overall: Loss = {1}, Accuracy of {2}"\
              .format(e+1, total_loss, total_correct))
        
    return total_loss, total_correct 
