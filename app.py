from __future__ import print_function
import flask
from flask import Flask, request
import base64
from io import StringIO
import base64
import cStringIO
import sys


app = Flask(__name__)

g_sess = 0
img = 0
input_node = 0
net = 0

@app.route("/", methods=['GET', 'POST'])
def hell2o():
	global g_sess
	return flask.send_file('/home/tomnomnom12/depthDemo1/index.html')

@app.route("/img", methods=['GET', 'POST'])
def hello():
	global g_sess
	global img
	global input_node
	global net
	height = 228
	width = 304
	channels = 3
	batch_size = 1
	file = request.files['file']
	file.save('./my.png')

	png = Image.open('./my.png').convert('RGBA')
	background = Image.new('RGBA', png.size, (255,255,255))

	img = Image.alpha_composite(background, png)
	img = img.convert('RGB')
	img = img.resize([width,height], Image.ANTIALIAS)
	img = np.array(img).astype('float32')
	img = np.expand_dims(np.asarray(img), axis = 0)
	# Evalute the network for the given image
	pred = g_sess.run(net.get_output(), feed_dict={input_node: img})

	# Plot result
	formatted = ((pred[0,:,:,0]) * 255 / np.max(pred[0,:,:,0])).astype('uint8')
	img2 = Image.fromarray(formatted)
	buffer = cStringIO.StringIO()
	img2.save(buffer, format="JPEG")
	img_str = base64.b64encode(buffer.getvalue())
	return img_str


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
    

        



