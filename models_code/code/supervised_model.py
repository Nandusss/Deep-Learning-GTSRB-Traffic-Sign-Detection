import os
import time
import zipfile
import sys
from tqdm import tqdm

import numpy as np

# Get the current working directory
cwd = os.getcwd()

# Path to the archive file
archive_path = os.path.join(cwd, '..', 'dataset', 'archive.zip')

# Path to extract the dataset
input_path = os.path.join(cwd, '..', 'dataset', 'input')

# Extract the archive if it hasn't been extracted yet
if not os.path.exists(input_path):
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(input_path)
        sys.stdout.write('Extracting dataset...')
        sys.stdout.flush()
        for i in tqdm(range(10)):
            time.sleep(0.5)
            sys.stdout.write('.')
            sys.stdout.flush()
        sys.stdout.write(' Done!\n')

# Now you can access the extracted dataset and build your CNN
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
import keras

# define the path to the images
data_dir = '../dataset/input/'
train_path = '../dataset/input/Train'
test_path = '../dataset/input/Test'

# Resizing the images to 30x30x3, channels 3 means RGB
IMG_HEIGHT = 30
IMG_WIDTH = 30
channels = 3

# Get the number of categories
NUM_CATEGORIES = len(os.listdir(train_path))
print(f'Number of categories: {NUM_CATEGORIES}')

# Define the labels
classes = { 
            0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles',
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory',
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons'
        }

# Plotting the number of images in each class
folders = os.listdir(train_path)

train_number = []
class_num = []

# Count the number of images in each class
for folder in tqdm(folders):
    train_files = os.listdir(train_path + '/' + folder)
    train_number.append(len(train_files))
    class_num.append(classes[int(folder)])
    
# Sorting the dataset on the basis of number of images in each class
zipped_lists = zip(train_number, class_num)
sorted_pairs = sorted(zipped_lists)
tuples = zip(*sorted_pairs)
train_number, class_num = [ list(tuple) for tuple in  tuples]

# Plotting the number of images in each class
plt.figure(figsize=(21,10))
plt.bar(class_num, train_number)
plt.xticks(class_num, rotation='vertical')
plt.show()

# Load the images and labels
image_data = []
image_labels = []
for i in tqdm(range(NUM_CATEGORIES)):
    path = data_dir + '/Train/' + str(i)
    images = os.listdir(path)
    for img in images:
        try:
            image = cv2.imread(path + '/' + img)
            image_fromarray = Image.fromarray(image, 'RGB')
            resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
            image_data.append(np.array(resize_image))
            image_labels.append(i)
        except:
            print("Error in " + img)
# Changing the list to numpy array
image_data = np.array(image_data)
image_labels = np.array(image_labels)
print(f'\nimage data shape: {image_data.shape}, image label shape:{image_labels.shape}')

# Shuffle the data
shuffle_indexes = np.arange(image_data.shape[0])
np.random.shuffle(shuffle_indexes)
image_data = image_data[shuffle_indexes]
image_labels = image_labels[shuffle_indexes]

# Split the data into training and validation
X_train, X_val, y_train, y_val = train_test_split(image_data, image_labels, test_size=0.2, random_state=42, shuffle=True)

# Normalize the data from 0 to 255 to 0 to 1
X_train = X_train/255
X_val = X_val/255

print("X_train.shape", X_train.shape)
print("X_valid.shape", X_val.shape)
print("y_train.shape", y_train.shape)
print("y_valid.shape", y_val.shape)

# One hot encoding the labels
y_train = keras.utils.to_categorical(y_train, NUM_CATEGORIES)
y_val = keras.utils.to_categorical(y_val, NUM_CATEGORIES)

print('y_train.shape', y_train.shape)
print('y_valid.shape', y_val.shape)

# Building the model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(NUM_CATEGORIES, activation='softmax'))

# Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Training the model
history = model.fit(X_train, y_train, batch_size=32, epochs=30, validation_data=(X_val, y_val))

# Save the model
model.save('../dataset/output/traffic_sign_classifier.h5')

# Plotting the training and validation accuracy
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# Plotting the training and validation loss
plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# Load the model
from keras.models import load_model
import pandas as pd
import cv2
from PIL import Image
import numpy as np

model = load_model('../dataset/output/traffic_sign_classifier.h5')

IMG_HEIGHT = 30
IMG_WIDTH = 30
data_dir = '../dataset/input/'
test_path = '../dataset/input/Test'

# Load the test data
test = pd.read_csv(data_dir + '/Test.csv')

labels = test["ClassId"].values
imgs = test["Path"].values

# Load the images
test_image = []
for img_name in tqdm(imgs):
    try:
        image = cv2.imread(test_path + '/' + img_name.replace('Test/', ''))
        image_fromarray = Image.fromarray(image, 'RGB')
        resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
        test_image.append(np.array(resize_image))
    except:
        print("Error in " + img_name)

# Normalize the data
test_data = np.array(test_image)
test_data = test_data/255

# Make predictions
predictions = model.predict(test_data)
predictions = np.argmax(predictions, axis=1)

# Accuracy with the test data
from sklearn.metrics import accuracy_score
print(f'Test Accuracy: {accuracy_score(labels, predictions)}')

# print the classification report
from sklearn.metrics import classification_report
print(classification_report(labels, predictions))