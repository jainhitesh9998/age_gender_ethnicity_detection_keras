import glob
import os
import pandas as pd
import csv
import numpy as np
import cv2

'''
return list of all files with a particular extension in a list
'''
def list_files(paths = [], extension = "*.jpg"):
    file_list = []
    for path in paths:
        print(os.path.join(path, extension))
        file_list += glob.glob(os.path.join(path, extension))
    return file_list


'''
creates a csv file with the age, race, gender dataset
'''
def create_csv(file_list=[], save_path='.', file_name='data.csv'):
    files = []
    age = []
    gender = []
    race = []
    for file in file_list:
        name = os.path.split(file)[-1]
        split = name.split('_')
        if len(split) != 4 or int(split[0]) > 100:
            continue
        a, g, r = split[:3]
        files.append(file)
        age.append(a)
        gender.append(g)
        race.append(r)
    print(len(files))
    df = pd.DataFrame(np.column_stack([files, age, gender, race]),
                      columns=['files', 'age', 'gender', 'race'])
    save = file_name
    if save_path != '.':
        save = os.path.join(save_path, file_name)
    df.to_csv(save, index=False)


def convert_int(x):
    try:
        return int(x)
    except ValueError:
        return x


def read_csv(file_name):
    output_list = []
    with open(file_name, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            _ = [convert_int(x) for x in row]
            output_list.append(_)
    return output_list

def read_image(path):
    #print (path)
    image = cv2.imread(path)
    return cv2.resize(image, (64,64))