import zipfile
import sys
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

class TrafficSignClassifier:
    def __init__(self):
        self.cwd = os.getcwd()
        self.archive_path = os.path.join(self.cwd, '..', 'dataset', 'archive.zip')
        self.input_path = os.path.join(self.cwd, '..', 'dataset', 'input')
        self.data_dir = '../dataset/input/'
        self.train_path = os.path.join(self.data_dir, 'Train')
        self.test_path = os.path.join(self.data_dir, 'Test')
        self.IMG_HEIGHT = 30
        self.IMG_WIDTH = 30
        self.channels = 3
        self.NUM_CATEGORIES = len(os.listdir(self.train_path))
        self.classes = { 
            0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)', 
            9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection', 
            12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles', 16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 18:'General caution', 19:'Dangerous curve left', 20:'Dangerous curve right', 
            21:'Double curve', 22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right', 
            25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing', 
            29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing', 
            32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead', 
            35:'Ahead only', 36:'Go straight or right', 37:'Go straight or left', 38:'Keep right', 
            39:'Keep left', 40:'Roundabout mandatory', 41:'End of no passing', 42:'End no passing veh > 3.5 tons'
        }
        self.model = None

    def extract_dataset(self):
        if not os.path.exists(self.input_path):
            with zipfile.ZipFile(self.archive_path, 'r') as zip_ref:
                zip_ref.extractall(self.input_path)
                sys.stdout.write('Extracting dataset...\n')
                sys.stdout.flush()

    def augment_data(self):
        folders = os.listdir(self.train_path)
        train_number = [len(os.listdir(os.path.join(self.train_path, folder))) for folder in folders]
        max_images = max(train_number)
        datagen = ImageDataGenerator(
            rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, 
            shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
        )
        for folder, num_images in tqdm(zip(folders, train_number)):
            if num_images < max_images:
                folder_path = os.path.join(self.train_path, folder)
                images = os.listdir(folder_path)
                for i in range(max_images - num_images):
                    img_path = os.path.join(folder_path, images[i % num_images])
                    img = load_img(img_path)
                    x = img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    it = datagen.flow(x, batch_size=1, save_to_dir=folder_path, save_prefix='aug', save_format='jpeg')
                    it.next()

    def load_data(self):
        image_data = []
        image_labels = []
        for i in tqdm(range(self.NUM_CATEGORIES)):
            path = os.path.join(self.train_path, str(i))
            images = os.listdir(path)
            for img in images:
                try:
                    image = cv2.imread(os.path.join(path, img))
                    image_fromarray = Image.fromarray(image, 'RGB')
                    resize_image = image_fromarray.resize((self.IMG_HEIGHT, self.IMG_WIDTH))
                    image_data.append(np.array(resize_image))
                    image_labels.append(i)
                except:
                    print("Error in " + img)
        image_data = np.array(image_data)
        image_labels = np.array(image_labels)
        return image_data, image_labels

    def augment_and_concatenate_data(self, image_data, image_labels):
        datagen = ImageDataGenerator(
            rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1
        )
        augmented_data = []
        augmented_labels = []
        for i in tqdm(range(self.NUM_CATEGORIES)):
            path = os.path.join(self.train_path, str(i))
            images = os.listdir(path)
            for img in images:
                try:
                    image = cv2.imread(os.path.join(path, img))
                    image_fromarray = Image.fromarray(image, 'RGB')
                    resize_image = image_fromarray.resize((self.IMG_HEIGHT, self.IMG_WIDTH))
                    data = np.array(resize_image)
                    data = data.reshape((1,) + data.shape)
                    j = 0
                    for batch in datagen.flow(data, batch_size=1):
                        augmented_data.append(batch[0])
                        augmented_labels.append(i)
                        j += 1
                        if j == 5:
                            break
                except:
                    print("Error in " + img)
        augmented_data = np.array(augmented_data)
        augmented_labels = np.array(augmented_labels)
        image_data = np.concatenate((image_data, augmented_data))
        image_labels = np.concatenate((image_labels, augmented_labels))
        return image_data, image_labels

    def preprocess_data(self, image_data, image_labels):
        shuffle_indexes = np.arange(image_data.shape[0])
        np.random.shuffle(shuffle_indexes)
        image_data = image_data[shuffle_indexes]
        image_labels = image_labels[shuffle_indexes]
        X_train, X_val, y_train, y_val = train_test_split(image_data, image_labels, test_size=0.2, random_state=42, shuffle=True)
        X_train = X_train / 255
        X_val = X_val / 255
        y_train = tf.keras.utils.to_categorical(y_train, self.NUM_CATEGORIES)
        y_val = tf.keras.utils.to_categorical(y_val, self.NUM_CATEGORIES)
        return X_train, X_val, y_train, y_val

    def build_model(self, input_shape):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=input_shape))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(self.NUM_CATEGORIES, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        self.model = model

    def train_model(self, X_train, y_train, X_val, y_val):
        history = self.model.fit(X_train, y_train, batch_size=16, epochs=20, validation_data=(X_val, y_val))
        self.model.save('../dataset/output/traffic_sign_classifier.h5')
        return history

    def plot_history(self, history):
        plt.figure(0)
        plt.plot(history.history['accuracy'], label='training accuracy')
        plt.plot(history.history['val_accuracy'], label='val accuracy')
        plt.title('Accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show()
        plt.figure(1)
        plt.plot(history.history['loss'], label='training loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.title('Loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

    def evaluate_model(self):
        model = load_model('../dataset/output/traffic_sign_classifier.h5')
        test = pd.read_csv(os.path.join(self.data_dir, 'Test.csv'))
        labels = test["ClassId"].values
        imgs = test["Path"].values
        test_image = []
        for img_name in tqdm(imgs):
            try:
                image = cv2.imread(os.path.join(self.test_path, img_name.replace('Test/', '')))
                image_fromarray = Image.fromarray(image, 'RGB')
                resize_image = image_fromarray.resize((self.IMG_HEIGHT, self.IMG_WIDTH))
                test_image.append(np.array(resize_image))
            except:
                print("Error in " + img_name)
        test_data = np.array(test_image)
        test_data = test_data / 255
        predictions = model.predict(test_data)
        predictions = np.argmax(predictions, axis=1)
        print(f'Test Accuracy: {accuracy_score(labels, predictions)}')
        print(classification_report(labels, predictions))
        cm = confusion_matrix(labels, predictions)
        plt.figure(figsize=(25, 25))
        sns.heatmap(cm, annot=True)
        plt.show()

if __name__ == "__main__":
    classifier = TrafficSignClassifier()
    classifier.extract_dataset()
    classifier.augment_data()
    image_data, image_labels = classifier.load_data()
    image_data, image_labels = classifier.augment_and_concatenate_data(image_data, image_labels)
    X_train, X_val, y_train, y_val = classifier.preprocess_data(image_data, image_labels)
    classifier.build_model(X_train.shape[1:])
    history = classifier.train_model(X_train, y_train, X_val, y_val)
    classifier.plot_history(history)
    classifier.evaluate_model()
