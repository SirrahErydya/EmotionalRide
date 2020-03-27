import cv2
import os
from random import randint

anger = cv2.imread(os.path.join('ferg_data', 'aia', 'aia_anger', 'aia_anger_1.png'))
happy = cv2.imread(os.path.join('ferg_data', 'bonnie', 'bonnie_joy', 'bonnie_joy_1.png'))
sad = cv2.imread(os.path.join('ferg_data', 'malcolm', 'malcolm_sadness', 'malcolm_sadness_1.png'))
surprise = cv2.imread(os.path.join('ferg_data', 'ray', 'ray_surprise', 'ray_surprise_1.png'))

emotions = [anger, happy, sad, surprise]


def generate_random_emotion():
    emo = randint(0, 3)
    image = cv2.resize(emotions[emo], (480, 480))
    return image, emo


def evaluate(predicted, ground_truth):
    return predicted == ground_truth
