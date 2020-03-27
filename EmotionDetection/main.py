import sys
import torch
import cv2
from network import EmotionClassifier
import perception as p
import emo_generator as e
import numpy as np

if __name__ == '__main__':
    classifier = EmotionClassifier(out_layers=4)
    classifier.load_state_dict(torch.load('models/model-light.pt', map_location=torch.device('cpu')))
    video_capture = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    key_frame = 20
    frame_counter = 0
    prediction_text = []
    predicted_emo = -1
    gen_img, gen_emo = e.generate_random_emotion()
    correct_prediction = False
    face_display = np.zeros((480, 480, 3))
    while True:
        if frame_counter == key_frame:
            frame_counter = 0
        ret, frame = video_capture.read()
        face = p.extract_face(face_cascade, frame)
        if correct_prediction:
            gen_img, gen_emo = e.generate_random_emotion()
        if frame_counter == 0 and face is not None:
            prediction_text, predicted_emo = p.predict(classifier, face)
            face_display = cv2.cvtColor(cv2.resize(face, (480,480)), cv2.COLOR_GRAY2BGR)
        correct_prediction = e.evaluate(predicted_emo, gen_emo)
        text_vert = 30
        for text in prediction_text:
            cv2.putText(frame, text, (50, text_vert), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0, 255), 3)
            text_vert += 30
        stacked_image = np.hstack((gen_img, frame))
        image = np.zeros(stacked_image.shape)
        cv2.imshow('Emotion_Detection', stacked_image)
        frame_counter += 1
        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()