import sys
import torch
from network import EmotionClassifier
from perception import track_webcam

if __name__ == '__main__':
    emo = EmotionClassifier()
    emo.load_state_dict(torch.load('model-light.pt'))
    track_webcam(emo)