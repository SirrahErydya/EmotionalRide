import cv2
import torch

LABELS = ['Angry/Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def track_webcam(classifier, key_frame=20):
    video_capture = cv2.VideoCapture(0)
    frame_counter = 0
    while True:
        if frame_counter == key_frame:
            frame_counter = 0
        ret, frame = video_capture.read()
        height, width, _ = frame.shape
        obsolete_pxls = width - height
        crop = frame[:, int(obsolete_pxls / 2): width - int(obsolete_pxls / 2)]
        big_picture = cv2.resize(crop, (1500, 1500))
        if frame_counter == 0:
            grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(grayscale, (48, 48))
            tensor = torch.Tensor(resized)
            prediction = classifier(tensor).flatten()
        display_prediction(prediction, big_picture)
        cv2.imshow('Video', big_picture)
        cv2.imshow("Control Image", resized)
        frame_counter += 1
        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


def display_prediction(emo_prediction, frame):
    emos = torch.argsort(emo_prediction, descending=True)
    text_vert = 30
    for emo in emos:
        percentage = (emo_prediction[emo].item())
        text = "{0}: {1}".format(LABELS[emo.item()], percentage)
        cv2.putText(frame, text, (50, text_vert), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0, 255), 3)
        text_vert += 30
