import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical

data = []
labels = []

categories = ["with_mask", "without_mask"]

for category in categories:
    path = os.path.join("data", category)
    label = categories.index(category)

    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)

        img = cv2.resize(img, (224, 224))
        img = img / 255.0

        data.append(img)
        labels.append(label)

data = np.array(data)
labels = to_categorical(labels, 2)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

import tensorflow as tf

base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
output = tf.keras.layers.Dense(2, activation="softmax")(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)


model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

model.save("model/mask_model.h5")

loss, acc = model.evaluate(X_test, y_test)
print("Accuracy:", acc)