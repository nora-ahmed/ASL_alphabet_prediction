import streamlit as st
import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('my.keras')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
               12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
               23: 'X', 24: 'Y', 25: 'Z', 26: 'DEL', 27: 'nothing', 28: ' '}
s = []
c = 0
while True:

    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    k = cv2.waitKey(1)
    if k % 256 == 27:
        break

    elif k % 256 == 32:
        face = frame
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (200, 200))

        img_name = "frame_.png"
        cv2.imwrite(img_name, face)
        pre = cv2.imread(img_name)
        img = tf.keras.preprocessing.image.img_to_array(pre)

        img_array = np.expand_dims(pre, axis=0)

        # Predict the class
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_character = labels_dict[predicted_class]

        if predicted_character == 'DEL':
            s.pop()
        elif predicted_character == 'nothing':
            print(predicted_character)
            continue
        else:
            s.append(predicted_character)
            c = c + 1

        print(predicted_character)

    cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()

with open('example.txt', 'w') as file:
    file.write(''.join(s))
