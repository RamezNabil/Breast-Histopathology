import os
import cv2
import random
import matplotlib.pyplot  as plt

rel_directory = './Dataset/'
directory = r'C:\Users\Ramez\PycharmProjects\Breast-Histopathology\Dataset\\'

# def resize_data():
#     data = []
#     for folder in os.listdir(current_directory):  # looping on the folders (from 0 to 9)
#         for img in os.listdir(current_directory + folder):
#             label = int(img.split('_')[2].split('.')[0]) - int(img.split('_')[1].split('-')[0])
#             path = os.path.join(current_directory, folder, img)
#             path_processed = r'' + str(path)
#             img_data = cv2.imread(path_processed)
#             img_data = cv2.resize(img_data, (100, 100))
#             os.chdir(directory)
#
#             cv2.imwrite(str(label) + '_' + str(img.split('_')[0]) + '.jpg', img_data)

def load_data():
    features = []
    label = []
    for folder in os.listdir(rel_directory):
        for labelfolder in os.listdir(rel_directory + folder):
            for img in os.listdir(rel_directory + folder + '/' + labelfolder):
                label.append(labelfolder)
                img_data = cv2.imread(os.path.join(rel_directory, folder, labelfolder, img))
                img_data = cv2.resize(img_data, (224, 224))
                features.append(img_data)

    # shuffle both lists
    both = list(zip(features, label))  # join both arrays together
    random.shuffle(both)  # shuffle
    b: object
    a, b = zip(*both)  # disconnect them from each other
    return a, b