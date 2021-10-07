from facenet_pytorch import MTCNN
import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw
from matplotlib import cm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(keep_all=True, device=device)

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, _ = mtcnn.detect(frame)

    # Draw faces
    frame_draw = frame.copy()

    if not boxes is None:
        for box in boxes:
            box = [int(coordinate) for coordinate in box]
            print(box)
            cv2.rectangle(frame_draw, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 6)
            cv2.imwrite()
        print('\nDone')

    cv2.imshow("Processing", frame_draw)

    if cv2.waitKey(1) & 0XFF == ord("q"):
        break
