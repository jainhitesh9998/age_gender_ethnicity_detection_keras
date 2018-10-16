import tensorrt as trt
import sys
import uff
import tensorflow as tf


def getModel(filepath):
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        tf.train.Saver().restore(session, filepath)
        graphdef = tf.get_default_graph().as_graph_def()
        frozen_graph = tf.graph_util.convert_variables_to_constants(session, graphdef, ['import/gender_detection/Softmax', 'import/age_detection/Softmax', 'import/race/Softmax'])
        return tf.graph_util.remove_training_nodes(frozen_graph)

#if len(sys.argv) < 3:
#    print ('Usage: python tf_to_trt.py [TF Model] [TRT Model]')
#    sys.exit(0)
#else:
#    print (' Convert '+sys.argv[1]+' to '+sys.argv[2])

#tf_model = model.getChatBotModel(sys.argv[1])
#uff_model = uff.from_tensorflow(getModel('weights/my_model.pb') , ['import/gender_detection/Softmax', 'import/age_detection/Softmax', 'import/race/Softmax'], output_filename='my_model.uff', text=False)
uff_model = uff.from_tensorflow_frozen_model('weights/my_model.pb' , ['gender_detection/Softmax', 'age_detection/Softmax', 'race/Softmax'], output_filename='my_model.uff', text=False)
print ( 'Successfully transfer to UFF model')
