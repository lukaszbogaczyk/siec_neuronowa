import cv2
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential, load_model

nazwa_obrazu = 'roza.jpg' # dmuchawiec.jpg, roza.jpg, tulipan.jpg


def sprawdzenie_predykcji(prediction):
    labels = ['daisy(stokrotka)', 'dandelion(dmuchawiec)', 'rose(róża)', 'sunflower(słonecznik)', 'tulip(tulipan)']
    max_prob_index = np.argmax(prediction)
    flower_type = labels[max_prob_index]
    confidence = prediction[0][max_prob_index] * 100
    return f'Model przewiduje: {flower_type} z {confidence:.2f}% pewnością.'


model = load_model(os.path.join('modele','model_rozpoznawania_kwiatow.h5'))

# Obraz testowy
img = cv2.imread(nazwa_obrazu)
resize = tf.image.resize(img, (256, 256))
pre = model.predict(np.expand_dims(resize / 255, 0))

wynik = sprawdzenie_predykcji(pre)
print(wynik)