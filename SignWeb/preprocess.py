import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Directory containing the dataset
DATASET_DIR = r'C:\Users\Bhuvana R\Documents\datasets_zip\datasets'

labels = os.listdir(DATASET_DIR)
X, y = [], []

for label in labels:
    class_num = labels.index(label)
    for image_path in os.listdir(os.path.join(DATASET_DIR, label)):
        img = cv2.imread(os.path.join(DATASET_DIR, label, image_path))
        img = cv2.resize(img, (64, 64))
        X.append(img)
        y.append(class_num)

X = np.array(X).astype('float32') / 255.0
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(labels), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

model.save('sign_language_model.h5')
