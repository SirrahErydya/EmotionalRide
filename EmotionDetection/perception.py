import cv2
import torch

LABELS = ['Angry', 'Happy', 'Sad', 'Surprise']


def track_webcam(classifier, key_frame=20):
    video_capture = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    frame_counter = 0
    while True:
        if frame_counter == key_frame:
            frame_counter = 0
        ret, frame = video_capture.read()
        face = extract_face(face_cascade, frame)
        if frame_counter == 0 and face is not None:
            resized = cv2.resize(face, (48, 48))
            tensor = torch.Tensor(resized)
            prediction = classifier(tensor).flatten()
            cv2.imshow("Face", face)
        display_prediction(prediction, frame)
        cv2.imshow('Video', frame)
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
        percentage = torch.exp(emo_prediction[emo]).item() * 100
        text = "{0}: {1:.2f}".format(LABELS[emo.item()], percentage, '.2f')
        cv2.putText(frame, text, (50, text_vert), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0, 255), 3)
        text_vert += 30


def extract_face(face_cascade, frame):
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        grayscale,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(48, 48)
    )
    if len(faces) > 0:
        x, y, w, h = faces[0]
        if w > 0 and h > 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face = frame[y:y+h, x:x+w]
            return cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    return None
