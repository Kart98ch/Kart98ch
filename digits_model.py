import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn import model_selection

#Load csv file to train model with pixel intensities for each entry.
dig = pd.read_csv('train.csv')

clas = dig['label']

train_data = dig.drop(['label'],axis=1)

#Create a training and validation set.
x_train, x_valid, y_train, y_valid = model_selection.train_test_split(train_data, clas,
                                                                    test_size=0.3,stratify=clas,random_state=0)

x_train = tf.cast(tf.Variable(x_train.to_numpy()/255), tf.float32)
#y_train = tf.reshape(tf.Variable(y_train.to_numpy()),[29400,1])

x_valid = tf.cast(tf.Variable(x_valid.to_numpy()/255), tf.float32)
#y_valid = tf.reshape(tf.Variable(y_valid.to_numpy()),[12600,1])

train_data = tf.cast(tf.Variable(train_data.to_numpy()/255), tf.float32)

#Initialise random matrices for each layer of the neural network.
W1 = tf.Variable(tf.random.normal([784,300],stddev=0.03), name = 'W1')
b1 = tf.Variable(tf.random.normal([300],stddev=0.03), name = 'b1')
W2 = tf.Variable(tf.random.normal([300,10],stddev=0.03), name = 'W2')
b2 = tf.Variable(tf.random.normal([10],stddev=0.03), name = 'b2')

#Defining functions to be used later when running multiple passes.
def nn_model(X, W1, b1, W2, b2):
    Z1 = tf.add(tf.matmul(X,W1),b1)
    h1 = tf.nn.relu(Z1)
    logits = tf.add(tf.matmul(h1,W2),b2)
    return logits

def loss_fn(logits, labels):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                                              logits=logits))
    return cross_entropy


optimizer = tf.keras.optimizers.Adam()


epochs = 500
y_train = tf.one_hot(y_train, 10)                       #y_train to clas
clas = tf.one_hot(clas, 10)

#Training.
for i in range(epochs):
    with tf.GradientTape() as tape:
        logits = nn_model(x_train, W1, b1, W2, b2)      #change x_train to train_data
        loss = loss_fn(logits, y_train)                 #change y_train to clas
    gradients = tape.gradient(loss,[W1, b1, W2, b2])
    optimizer.apply_gradients(zip(gradients, [W1, b1, W2, b2]))
    test_logits = nn_model(x_valid, W1, b1, W2, b2)
    max_idxs = tf.argmax(test_logits, axis=1)
    test_acc = np.sum(max_idxs.numpy() == y_valid) / len(y_valid)
    print('Epoch '+str(i+1)+': Test Acc '+str(test_acc))
   