#!/usr/bin/env python
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

# Load TensorFlow graph
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
# tf_time = timeit.timeit("run()", setup=setup, number = NUM_TIME)
tf_time = timeit.repeat("run()", setup=setup, repeat=10, number = 1)

# Close session to release resources
tf_sess.close()


# Optimize the graph with TensorRT
#trt_graph = trt.create_inference_graph(
#    input_graph_def=frozen_graph,
#    outputs=output_names,
#    max_batch_size=1,
#    max_workspace_size_bytes=1 << 25,
#    precision_mode='FP16',
#    minimum_segment_size=50
#)

# Create session
#tf_config = tf.ConfigProto()
#tf_config.gpu_options.allow_growth = True
#tf_sess = tf.Session(config=tf_config)

# Load TensorRT optimized graph
#tf.import_graph_def(trt_graph, name='')
#tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
#tf_output = tf_sess.graph.get_tensor_by_name(output_names[0] + ':0')

# Preprocess image
#image = Image.open(IMAGE_PATH)
#plt.imshow(image)
#width = int(tf_input.shape.as_list()[1])
#height = int(tf_input.shape.as_list()[2])
#image = np.array(image.resize((width, height)))

# Execute model and time it
#setup = "from __main__ import run"
# trt_time = timeit.timeit("run()", setup=setup, number = NUM_TIME)
#trt_time = timeit.repeat("run()", setup=setup, repeat=10, number = 1)

# Close the session to release resources
#tf_sess.close()



#print('TensorRT total time: {time}'.format(time=trt_time))
#print('TensorRT time per classification: {time}\n'.format(time=trt_time/NUM_TIME))
print('TensorFlow time: {time}'.format(time=tf_time))
#print('TensorFlow time per classification: {time}\n'.format(time=tf_time/NUM_TIME))
#print('Time differences: {differences}'.format(differences=trt_time - tf_time))
#print('Time difference per classification: {differences}'.format(differences=(trt_time - tf_time)/NUM_TIME))

#print('Time differences: {differences}'.format(differences=[trt-tf for (trt, tf) in zip(trt_time, tf_time)]))
