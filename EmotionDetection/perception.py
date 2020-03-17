import cv2
import torch

LABELS = ['Angry/Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def track_webcam(classifier):
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        emo_prediction = predict(classifier, frame)
        # Display the resulting frame
        cv2.putText(frame, LABELS[emo_prediction], (50, 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255,255), 3)
        cv2.imshow('Video', frame)

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


def predict(classifier, frame):
    height, width, _ = frame.shape
    obsolete_pxls = width - height
    crop = frame[:, obsolete_pxls:width-obsolete_pxls]
    grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grayscale, (48,48))
    tensor = torch.Tensor(resized)
    prediction = classifier(tensor)
    return torch.argmax(prediction).item()
