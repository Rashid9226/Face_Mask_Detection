import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input #newly added
from sklearn.model_selection import train_test_split


data = []
labels = []

# categories = ["with_mask", "without_mask"]
categories = ["with_mask", "without_mask", "incorrect"] #newly added

for category in categories:
    path = os.path.join("data", category)
    label = categories.index(category)

    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        img = cv2.resize(img, (224, 224))
        # img = img / 255.0
        img = preprocess_input(img) #newly added

        data.append(img)
        labels.append(label)

data = np.array(data)
labels = to_categorical(labels, 3) #2 -> 3

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

#Data Augmentation #newly added
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

datagen.fit(X_train)


#Model Building
base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

#freezing layers of the base model
base_model.trainable = False

# Fine-tune last 20 layers #newly added
for layer in base_model.layers[-20:]:
    layer.trainable = True

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x) #newly added
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x) #newly added
output = tf.keras.layers.Dense(3, activation="softmax")(x) #2->3

model = tf.keras.Model(inputs=base_model.input, outputs=output)


model.compile(
    # optimizer="adam",
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


#Training the model 
EPOCHS = 10
BATCH_SIZE = 32

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_test, y_test),
    epochs=EPOCHS
)

# model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

model.save("model/mask_model.h5")

loss, acc = model.evaluate(X_test, y_test)
print("Accuracy:", acc)
