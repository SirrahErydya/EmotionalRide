import sys
import torch
from network import EmotionClassifier
from perception import track_webcam

if __name__ == '__main__':
    emo = EmotionClassifier(out_layers=6)
    emo.load_state_dict(torch.load('models/model-light.pt', map_location=torch.device('cpu')))
    track_webcam(emo)