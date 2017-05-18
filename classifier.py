import tensorflow as tf 
import numpy as np 

#Global variables
tf.flags.DEFINE_string("image_path", "./images/image1.jpg", "Image to be classified")
label_path = './retrained_labels.txt'
model_path = './retrained_graph.pb'

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

image = tf.gfile.FastGFile(FLAGS.image_path, 'rb').read()
labels = [line.rstrip() for line in tf.gfile.GFile("retrained_labels.txt")]

with tf.gfile.FastGFile(model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    #predict
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image})
    
    #Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    for node_id in top_k:
        human_string = labels[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))