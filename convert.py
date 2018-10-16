import keras
import keras.backend as K
import tensorflow as tf
import uff
from wide_resnet import wide_resnet


model = wide_resnet.WideResNet(image_size = 64,race = True, train_branch=False)()
model.load_weights('weights/model_new.h5')
model.summary()

output_names = ['Dense/gender_detection', 'Dense/age_detection', 'Dense/race']
frozen_graph_filename = 'weights/model_new.h5'
sess = K.get_session()

# freeze graph and remove training nodes
graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_names)
graph_def = tf.graph_util.remove_training_nodes(graph_def)

# write frozen graph to file
with open(frozen_graph_filename, 'wb') as f:
    f.write(graph_def.SerializeToString())
f.close()

# convert frozen graph to uff
uff_model = uff.from_tensorflow_frozen_model(frozen_graph_filename, output_names)
