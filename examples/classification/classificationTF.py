#!/usr/bin/env python3
# coding: utf-8

# In[41]:


from PIL import Image
import sys
import os
import urllib
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tf_trt_models.classification import download_classification_checkpoint, build_classification_graph
import timeit


MODEL = 'inception_v3'
CHECKPOINT_PATH = 'inception_v3.ckpt'
NUM_CLASSES = 1001
LABELS_PATH = './data/imagenet_labels_%d.txt' % NUM_CLASSES
IMAGE_PATH = './data/dog-yawning.jpg'
NUM_TIME = 1000

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def run():
    output = tf_sess.run(tf_output, feed_dict={
        tf_input: image[None, ...]
    })
    return output

# Download the checkpoint and sample image
checkpoint_path = download_classification_checkpoint(MODEL, 'data')
# Build the frozen graph
frozen_graph, input_names, output_names = build_classification_graph(
    model=MODEL,
    checkpoint=checkpoint_path,
    num_classes=NUM_CLASSES
)

# Create session

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(frozen_graph, name='')
tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
tf_output = tf_sess.graph.get_tensor_by_name(output_names[0] + ':0')

# Preprocess image
image = Image.open(IMAGE_PATH)
plt.imshow(image)
width = int(tf_input.shape.as_list()[1])
height = int(tf_input.shape.as_list()[2])
image = np.array(image.resize((width, height)))

# Execute model and time it
setup = "from __main__ import run"
# trt_time = timeit.timeit("run()", setup=setup, number = NUM_TIME)
trt_time = timeit.repeat("run()", setup=setup, repeat=NUM_TIME, number = 1)
# print('TensorFlow total time: {time}'.format(time=trt_time))
print('The average time to complete: {avg_time:.5f} seconds'.format(avg_time=sum(trt_time[1:])/(len(trt_time)-1)))
# Close the session to release resources
tf_sess.close()
