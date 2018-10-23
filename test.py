from wide_resnet import wide_resnet
import cv2
import time
import numpy as np

model = wide_resnet.WideResNet(image_size = 64,race = True, train_branch=True)()
#model.summary()
image = cv2.imread('/workspace/age_gender_race_detection_keras/dataset/crop_part1/42_1_0_20170104235631908.jpg.chip.jpg')
image = cv2.resize(image, (64, 64))
image_list = [image for i in range(2)]
image  = np.array(image_list)
for i in range(10):
	start_time = time.process_time()
	#result = sess.run([trt_y1, trt_y2, trt_y3], feed_dict=feed_dict)
	result = model.predict(image)
	stop_time = time.process_time()
	print(stop_time - start_time)
#print(result)


