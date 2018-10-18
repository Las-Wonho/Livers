import tensorflow as tf
import numpy as np
import os
import scipy.misc
from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver

from keras.preprocessing.image import load_img, img_to_array
images = []
for i in range(70):
    image = load_img('./DATA/'+str(i)+'.jpeg',target_size=(64,64))
    image = img_to_array(image)
    image.shape
    image = image / 256
    images.append(image)
    images.append(image)
    images.append(image)

def get_noise(batch_size, n_noise):
    return np.random.uniform(-1., 1., size = [batch_size, n_noise])

X = tf.placeholder(tf.float32, [None, 64,64,3])
Y = tf.placeholder(tf.float32, [None, 1])
Z = tf.placeholder(tf.float32, [None, 100])

def generate(Z):
    w1 = tf.Variable(tf.random_normal([100, 16384]))
    h1 = tf.matmul(Z,w1)
    re = tf.reshape(h1,[-1,4,4,1024])
    re = tf.layers.batch_normalization(re)

    conv_h1 = tf.layers.conv2d_transpose(re ,512, 5, 2, padding='same')
    
    conv_h1 = tf.nn.relu(conv_h1)

    conv_h2 = tf.layers.conv2d_transpose(conv_h1 ,256, 4, 2, padding='same')
    conv_h2 = tf.nn.relu(conv_h2)

    conv_h3 = tf.layers.conv2d_transpose(conv_h2 ,128, 4, 2, padding='same')
    conv_h3 = tf.nn.relu(conv_h3)

    conv_h5 = tf.layers.conv2d_transpose(conv_h3 ,3, 4, 2, padding='same')
    return conv_h5

def discriminaster(inputs):
    W1 = tf.Variable(tf.random_normal([3,3,3,32],stddev=0.02))
    H1 = tf.nn.conv2d(inputs,W1, strides=[1,2,2,1], padding="SAME")
    H1 = tf.layers.batch_normalization(H1)
    H1 = tf.nn.relu(H1)

    W2 = tf.Variable(tf.random_normal([3,3,32,64],stddev=0.02))
    H2 = tf.nn.conv2d(H1,W2, strides=[1,2,2,1], padding="SAME")
    H2 = tf.nn.relu(H2)

    W3 = tf.Variable(tf.random_normal([3,3,64,256],stddev=0.02))
    H3 = tf.nn.conv2d(H2,W3, strides=[1,2,2,1], padding="SAME")
    H3 = tf.nn.relu(H3)
    
    W4 = tf.Variable(tf.random_normal([3,3,256,1024],stddev=0.02))
    H4 = tf.nn.conv2d(H3,W4, strides=[1,2,2,1], padding="SAME")
    H4 = tf.nn.relu(H4)
    
    out = tf.reshape(H4,[-1,16384])
    out = tf.layers.dense(out,256)
    out = tf.layers.dense(out,1)
    return out

G = generate(Z)

D_real = discriminaster(X)
D_gene = discriminaster(G)

loss_D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real,labels=tf.ones_like(D_real)))
loss_D_gene = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gene,labels=tf.zeros_like(D_gene)))

loss_D = loss_D_gene + loss_D_real
loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gene, labels=tf.ones_like(D_gene)))

train_D = tf.train.AdamOptimizer(0.001).minimize(loss_D)
train_G = tf.train.AdamOptimizer(0.001).minimize(loss_G)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10):
    noise = get_noise(1,100)
    _, varsD=sess.run([train_D, loss_D], 
                  feed_dict={
                      X:np.array(images),
                      Z: noise})
    _, varsG=sess.run([train_G, loss_G], 
                  feed_dict={
                      X:np.array(images),
                      Z: noise})
    if(i%100==0):
        nn = get_noise(3,100)
        im = sess.run(G,feed_dict={Z:nn})
        scipy.misc.imsave('./'+str(i)+'-0.jpg',im[0])
        scipy.misc.imsave('./'+str(i)+'-1.jpg',im[1])
        scipy.misc.imsave('./'+str(i)+'-2.jpg',im[2])

    print(str(i)+"   D : "+str(varsD)+"\t G : "+str(varsG))