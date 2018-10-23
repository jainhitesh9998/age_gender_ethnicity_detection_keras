#import tensorrt as trt
#datatype = trt.float32
#model_file ='my_model.uff'

'''with builder = trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
    parser.register_input("Placeholder", (3, 64, 64))
    parser.register_output("gender_detection/Softmax")
    parser.register_output("age_detection/Softmax")
    parser.register_output("race/Softmax")
parser.parse(model_file, network) '''


import time
import numpy as np
import tensorflow as tf
#import tensorrt as trt
from tensorflow.contrib import tensorrt as trt
import cv2
batch_size = 1 # change to 128 when you use batch
workspace_size_bytes = 1 << 30
precision_mode = 'FP16' # use 'FP32' for K80
trt_gpu_ops = tf.GPUOptions(per_process_gpu_memory_fraction = 0.50)



classifier_model_file = 'weights/my_model.pb'
classifier_graph_def = tf.GraphDef()
with tf.gfile.Open(classifier_model_file, 'rb') as f:
    data = f.read()
    classifier_graph_def.ParseFromString(data)
print('Loaded classifier graph def')

trt_graph_def = trt.create_inference_graph(
    input_graph_def=classifier_graph_def,
    outputs=['gender_detection/Softmax', 'age_detection/Softmax', 'race/Softmax'],
    max_batch_size=batch_size,
    max_workspace_size_bytes=workspace_size_bytes,
    precision_mode=precision_mode)
#trt_graph_def=trt.calib_graph_to_infer_graph(trt_graph_def) # For only 'INT8'
print('Generated TensorRT graph def')



tf.reset_default_graph()
g2 = tf.Graph()
with g2.as_default():
    trt_x, trt_y1, trt_y2, trt_y3 = tf.import_graph_def(
        trt_graph_def,
        return_elements=['input:0','gender_detection/Softmax:0', 'age_detection/Softmax:0', 'race/Softmax:0'])
    print('Generated tensor by TensorRT graph')



with tf.Session(graph=g2, config=tf.ConfigProto(gpu_options=trt_gpu_ops)) as sess:
    image = cv2.imread('/workspace/age_gender_race_detection_keras/dataset/crop_part1/42_1_0_20170104235631908.jpg.chip.jpg')
    image = cv2.resize(image, (64, 64))
    import numpy as np
    image = np.expand_dims(image, axis=0)
    feed_dict = {
        trt_x: image
    }
    start_time = time.process_time()
    result = sess.run([trt_y1, trt_y2, trt_y3], feed_dict=feed_dict)
    stop_time = time.process_time()
    print(result)
