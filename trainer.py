from wide_resnet import wide_resnet
from utils import utils
import sklearn
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import os
import numpy as np

def generator(samples, batch_size = 32):
    num_samples = len(samples)
    print(num_samples)
    while 1: #to run the generator indefinitely, pumping the X, Y sets for the neural network
        #shuffle the samples on each EPOCH
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset + batch_size]
            labels = []
            images = []
            ages = []
            genders = []
            races = []
            age = []
            gender = []
            race = []
            for batch_sample in batch_samples:
                path, a, g, r = batch_sample
                name =  os.path.split(batch_sample[0])[-1]
                image = utils.read_image(path)
                #print(image.shape)
                if(a>100):
                  continue
                images.append(image)
                age.append(a)
                gender.append(abs(1 - g))
                race.append(r)
            ages = to_categorical(age, num_classes=101)
            genders = to_categorical(gender, num_classes=2)
            races = to_categorical(race, num_classes=5)
            labels = [np.array(genders), np.array(ages), np.array(races)]
            X_train = np.array(images)
            yield X_train,labels

def main():
    samples = utils.read_csv('extras/data.csv')
    model = wide_resnet.WideResNet(image_size=64, race=True, train_branch=True)()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mae', 'acc'])
    sklearn.utils.shuffle(samples)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    # Hyperparameters
    train_generator = generator(train_samples, 4)
    validation_generator = generator(validation_samples, 4)
    epoch = 3
    # end of hyperparameters

    history = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                                  validation_data=validation_generator, \
                                  nb_val_samples=len(validation_samples), nb_epoch=epoch)
    model.save(os.path.join('weights','model_new.h5'))

if '__name__' == '__main__':
    main()