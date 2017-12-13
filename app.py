import flask
from flask import Flask
app = Flask(__name__)

g_sess = 0
img = 0
input_node = 0
net = 0

@app.route("/")
def hello():
	global g_sess
	return "hey"

@app.route("/img")
def hello():
	global g_sess
	global img
	global input_node
	global net

	# Evalute the network for the given image
	pred = g_sess.run(net.get_output(), feed_dict={input_node: img})

	# Plot result
	formatted = ((pred[0,:,:,0]) * 255 / np.max(pred[0,:,:,0])).astype('uint8')
	img2 = Image.fromarray(formatted)
	img2.save('my.png')

	return flask.send_file('/home/tomnomnom12/depthDemo1/my.png',  mimetype='image/gif')


import argparse
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import models


def predict(model_data_path, image_path):
	global g_sess
	global img
	global input_node
	global net
	# Default input size
	height = 228
	width = 304
	channels = 3
	batch_size = 1

	# Read image
	img = Image.open(image_path)
	img = img.resize([width,height], Image.ANTIALIAS)
	img = np.array(img).astype('float32')
	img = np.expand_dims(np.asarray(img), axis = 0)

	# Create a placeholder for the input image
	input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

	# Construct the network
	net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)
	g_sess = tf.Session()      

	# Load the converted parameters
	print('Loading the model')

	# Use to load from ckpt file
	saver = tf.train.Saver()     
	saver.restore(g_sess, model_data_path)

	# Use to load from npy file
	#net.load(model_data_path, sess) 

        
                
pred = predict('./NYU_FCRN.ckpt', './meanwhile-2453-cc63cdb89c527209296a3ec7ffd9ee59@1x.jpg')
    

        



