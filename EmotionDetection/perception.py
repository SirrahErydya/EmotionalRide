import cv2
import torch

LABELS = ['Angry', 'Happy', 'Sad', 'Surprise']


def predict(classifier, face):
    resized = cv2.resize(face, (48, 48))
    tensor = torch.Tensor(resized)
    prediction = classifier(tensor).flatten()
    emos = torch.argsort(prediction, descending=True)
    prediction_text = []
    for emo in emos:
        percentage = torch.exp(prediction[emo]).item() * 100
        text = "{0}: {1:.2f}".format(LABELS[emo.item()], percentage, '.2f')
        prediction_text.append(text)
    return prediction_text, emos[0]


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
