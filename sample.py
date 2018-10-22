# This sample uses a UFF MNIST model to create a TensorRT Inference Engine
from random import randint
from PIL import Image
import numpy as np
import cv2
import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit

import tensorrt as trt

import sys, os
sys.path.insert(1, '../tensorrt/samples/python')
import common

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class ModelData(object):
    MODEL_FILE = os.path.join(os.path.dirname(__file__), "models/my_model.uff")
    INPUT_NAME ="input_1"
    INPUT_SHAPE = (3, 64, 64)
    OUTPUT_NAME  = ['import/gender_detection/Softmax', 'import/age_detection/Softmax', 'import/race/Softmax']

def build_engine(model_file):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = common.GiB(1)
        # Parse the Uff Network
        parser.register_input(ModelData.INPUT_NAME, ModelData.INPUT_SHAPE)
        parser.register_output(ModelData.OUTPUT_NAME)
        parser.parse(model_file, network)
        # Build and return an engine.
        return builder.build_cuda_engine(network)

# Loads a test case into the provided pagelocked_buffer.
def load_normalized_test_case(data_path, pagelocked_buffer, case_num=randint(0, 9)):
    test_case_path = os.path.join(data_path, str(case_num) + ".pgm")
    # Flatten the image into a 1D array, normalize, and copy to pagelocked memory.
    img = np.array(Image.open(test_case_path)).ravel()
    np.copyto(pagelocked_buffer, 1.0 - img / 255.0)
    return case_num


def load_data():
    samples = utils.read_csv('extras/data.csv')
    image_path = sample[0][0]
    return cv2.imread(image_path)

def main():
    #data_path = common.find_sample_data(description="Runs an MNIST network using a UFF model file", subfolder="mnist")
    model_file = ModelData.MODEL_FILE

    with build_engine(model_file) as engine:
        # Build an engine, allocate buffers and create a stream.
        # For more information on buffer allocation, refer to the introductory samples.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            #case_num = load_normalized_test_case(data_path, pagelocked_buffer=inputs[0].host)
            # For more information on performing inference, refer to the introductory samples.
            # The common.do_inference function will return a list of outputs - we only have one in this case.
            [output] = common.do_inference(load_data(), context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            #pred = np.argmax(output)
            print(output)
            #print("Test Case: " + str(case_num))
            #print("Prediction: " + str(pred))

if __name__ == '__main__':
    main()
