'''
Prints all tensors in a frozen graph
'''
import tensorflow as tf
import argparse
from recognition import facenet

parser=argparse.ArgumentParser()

parser.add_argument("--frozen_graph",default="recognition/model/20180402-114759.pb",
help="Frozen graph to use.")

args=parser.parse_args()

with tf.Graph().as_default() as graph:
    facenet.load_model(args.frozen_graph)

nodes=[n.name for n in graph.as_graph_def().node if not (n.name.startswith("InceptionResnetV1") )]#or n.name.startswith("MobilenetV2"))]

print("\n".join(nodes))

print(tf.contrib.graph_editor.get_tensors(tf.get_default_graph()))