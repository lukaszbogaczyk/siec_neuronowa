import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, RandomFlip, RandomRotation, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import cv2



# Wczytanie danych
# 0 - daisy, 1 - dandelion, 2 - rose, 3 - sunflower, 4 - tulip
dane = tf.keras.utils.image_dataset_from_directory(
    'data', shuffle=True, image_size=(256, 256), batch_size=32
)
dane = dane.shuffle(1000, seed=6, reshuffle_each_iteration=True)


# Rotacja
data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.05)
])
dane = dane.map(lambda x, y: (data_augmentation(x, training=True), y))


# Normalizacja
znormalizowane_dane = dane.map(lambda x,y: (x/255, y))


# Podział na zestawy
train_size = 94 # 70%
val_size = 27 # 20%
test_size = 14 # 10%


grupa_treningowa = znormalizowane_dane.take(train_size)
grupa_walidacyjna = znormalizowane_dane.skip(train_size).take(val_size)
grupa_testowa = znormalizowane_dane.skip(train_size + val_size).take(test_size)


# Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(5, activation='softmax')
])

# Kompilacja modelu
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.summary()

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

# Trening modelu
hist = model.fit(grupa_treningowa, epochs=30, validation_data=grupa_walidacyjna, callbacks=[tensorboard_callback, early_stopping])

# Wykresy
fig = plt.figure()
plt.plot(hist.history['loss'], color='blue', label='krzywa straty uczenia')
plt.plot(hist.history['val_loss'], color='orange', label='krzywa straty walidacji')
fig.suptitle('krzywe straty', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='krzywa dokładności')
plt.plot(hist.history['val_accuracy'], color='orange', label='krzywa dokładności walidacji')
fig.suptitle('krzywe dokładności', fontsize=20)
plt.legend(loc="upper left")
plt.show()

# Testowanie modelu
test_iterator = grupa_testowa.take(test_size).as_numpy_iterator()
acc = SparseCategoricalAccuracy()

for batch in test_iterator:
    X, y = batch
    yhat = model.predict(X)
    yhat_classes = np.argmax(yhat, axis=1)  # Najwyższe prawdopodobieństwo
    acc.update_state(y, yhat_classes)

print(f'Accuracy: {acc.result().numpy()}')

# Zapisywanie modelu do pliku
model.save(os.path.join('modele','model_rozpoznawania_kwiatow.h5'))