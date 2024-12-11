import json
import os
import time
import zipfile
import sys
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import smart_resize
from tensorflow.keras.applications.efficientnet import preprocess_input

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
        self.classes = self.get_classes()
        self.model = None

    def get_classes(self):
        return {
            0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)', 
            3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)', 
            6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)', 
            9: 'No passing', 10: 'No passing veh over 3.5 tons', 11: 'Right-of-way at intersection', 
            12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles', 16: 'Veh > 3.5 tons prohibited', 
            17: 'No entry', 18: 'General caution', 19: 'Dangerous curve left', 20: 'Dangerous curve right', 
            21: 'Double curve', 22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right', 
            25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing', 
            29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing', 
            32: 'End speed + passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead', 
            35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left', 38: 'Keep right', 
            39: 'Keep left', 40: 'Roundabout mandatory', 41: 'End of no passing', 42: 'End no passing veh > 3.5 tons'
        }

    def extract_dataset(self):
        if not os.path.exists(self.input_path):
            with zipfile.ZipFile(self.archive_path, 'r') as zip_ref:
                zip_ref.extractall(self.input_path)
                sys.stdout.write('Extracting dataset...')
                sys.stdout.flush()
                for i in tqdm(range(10)):
                    time.sleep(0.5)
                    sys.stdout.write('.')
                    sys.stdout.flush()
                sys.stdout.write(' Done!\n')

    def plot_class_distribution(self):
        folders = os.listdir(self.train_path)
        train_number = []
        class_num = []

        for folder in tqdm(folders):
            train_files = os.listdir(os.path.join(self.train_path, folder))
            train_number.append(len(train_files))
            class_num.append(self.classes[int(folder)])

        zipped_lists = zip(train_number, class_num)
        sorted_pairs = sorted(zipped_lists)
        train_number, class_num = [list(tuple) for tuple in zip(*sorted_pairs)]

        plt.figure(figsize=(21, 10))
        plt.bar(class_num, train_number)
        plt.xticks(class_num, rotation='vertical')
        plt.show()

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

    def preprocess_data(self, X, y=None, X_val=None, y_val=None, batch_size=1000):
        """
        Preprocess data with flexible input handling
        """
    
        # Handle splitting if validation set not provided
        if X_val is None or y_val is None:
            print("Splitting data into train and validation sets...")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            X_train, y_train = X, y

        def process_batch(X, desc):
            processed = []
            total_batches = (len(X) + batch_size - 1) // batch_size
            
            for i in tqdm(range(0, len(X), batch_size), 
                         desc=desc, 
                         total=total_batches):
                batch = X[i:i + batch_size]
                resized = np.array([smart_resize(img, (224, 224)) for img in batch])
                processed.append(preprocess_input(resized))
                
            return np.concatenate(processed, axis=0)

        print("\nProcessing training data...")
        X_train_processed = process_batch(X_train, "Training batches")
        
        print("\nProcessing validation data...")
        X_val_processed = process_batch(X_val, "Validation batches")


        # Convert labels to categorical
        print("\nConverting labels to categorical...")
        y_train = to_categorical(y_train, num_classes=self.NUM_CATEGORIES)
        y_val = to_categorical(y_val, num_classes=self.NUM_CATEGORIES)
        
        return X_train_processed, X_val_processed, y_train, y_val

    def build_model(self):
        """
        Build model architecture
        Args:
            model_type: EfficientNetB0
        """
        # EfficientNet implementation
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        base_model.trainable = False
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.NUM_CATEGORIES, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
            
        # Common compilation settings
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        model.summary()
        self.model = model

    def train_model(self, X_train, y_train, X_val, y_val):
        """
        Train model with preprocessed data
        """
        import tensorflow as tf
        import os
        
        if self.model is None:
            self.build_model()
        
        # Set save path based on model type
        save_dir = '../dataset/output'
        os.makedirs(save_dir, exist_ok=True)
        
        model_name = 'traffic_sign_classifier_efficient'
        save_path = os.path.join(save_dir, f'{model_name}.h5')

        # Create ModelCheckpoint callback
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )

        # Train model
        history = self.model.fit(
            X_train, 
            y_train,
            validation_data=(X_val, y_val),
            epochs=1,
            batch_size=32,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=save_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]
        )

        # Convert history to regular Python types before saving
        history_dict = {}
        for key in history.history:
            # Convert TensorFlow tensors to Python floats
            history_dict[key] = [float(val) for val in history.history[key]]

        try:
            # Save the history with error handling
            with open('training_history.json', 'w') as f:
                json.dump(history_dict, f)
        except Exception as e:
            print(f"Warning: Could not save training history: {str(e)}")

        return history

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
    print("Initializing TrafficSignClassifier...")
    classifier = TrafficSignClassifier()
    
    print("Extracting dataset...")
    classifier.extract_dataset()
    
    print("Plotting class distribution...")
    classifier.plot_class_distribution()
    
    print("Loading data...")
    image_data, image_labels = classifier.load_data()
    
    print("Preprocessing data...")
    X_train, X_val, y_train, y_val = classifier.preprocess_data(image_data, image_labels)
    
    print("Training model...")
    classifier.train_model(X_train, y_train, X_val, y_val)
    
    print("Evaluating model...")
    classifier.evaluate_model()