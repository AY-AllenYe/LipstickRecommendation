import cv2
import dlib
import numpy as np
from collections import OrderedDict

# Loading models
detector = dlib.get_frontal_face_detector()
criticPoints = dlib.shape_predictor("models\pretrained\shape_predictor_68_face_landmarks.dat")

shape_predictor_68_face_landmark=OrderedDict([
    ('mouth',(48,68)),
    ('right_eyebrow',(17,22)),
    ('left_eye_brow',(22,27)),
    ('right_eye',(36,42)),
    ('left_eye',(42,48)),
    ('nose',(27,36)),
    ('jaw',(0,17))
])

# Draw rectangle box to the whole face
def drawRectangle(detected,frame):
    margin = 0.2
    img_h,img_w,_=np.shape(frame)
    if len(detected) > 0:
        for i, locate in enumerate(detected):
            x1, y1, x2, y2, w, h = locate.left(), locate.top(), locate.right() + 1, locate.bottom() + 1, locate.width(), locate.height()

            xw1 = max(int(x1 - margin * w), 0)
            yw1 = max(int(y1 - margin * h), 0)
            xw2 = min(int(x2 + margin * w), img_w - 1)
            yw2 = min(int(y2 + margin * h), img_h - 1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            face = frame[yw1:yw2 + 1, xw1:xw2 + 1, :]
            cv2.putText(frame, 'User', (locate.left(), locate.top() - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    return frame

# Transform the detected facial attibutes to 2D coordinates 
def predict2Np(predict):
    # Initialize the 68 pairs of 2D coordinates
    # [(x1,y1),(x2,y2)……]
    dims=np.zeros(shape=(predict.num_parts,2),dtype=np.int32)
    # Traverse each key point of the face to obtain the 2D coordinates
    length=predict.num_parts
    for i in range(0,length):
        dims[i]=(predict.part(i).x,predict.part(i).y)
    return dims

# Research from the rectangle box and draw facial landmarks
def drawCriticPoints(detected,frame,organ_range=None):
    for (step,locate) in enumerate(detected):
        # 68 2D coordinates
        dims=criticPoints(frame,locate)
        # Transfer into 2D
        dims=predict2Np(dims)

        # (i,j) means range, which can refer to the dictory. The mouth goes to (48,68)
        for (name,(i,j)) in shape_predictor_68_face_landmark.items():
            # Search for the parts of face and point out, otherwise not to display
            if organ_range is not None and (i,j) != organ_range:
                break
            # Point out by dots
            for (x,y) in dims[i:j]:
                cv2.circle(img=frame,center=(x,y),
                           radius=2,color=(0,255,0),thickness=-1)
    return frame


# Single image
def signal_detect(img_path='images/face1.jpg'):
    img=cv2.imread(img_path)
    detected=detector(img)
    frame=drawRectangle(detected,img)
    frame = drawCriticPoints(detected, img)
    cv2.imshow('frame',frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Real-time video capture
def detect_time(organ_range=None):
    cap=cv2.VideoCapture(0)
    while cap.isOpened():
        ret,frame=cap.read()
        detected = detector(frame)
        frame = drawRectangle(detected, frame)
        frame=drawCriticPoints(detected,frame, organ_range)
        cv2.imshow('frame', frame)
        key=cv2.waitKey(1)
        if key==27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    mouth_range = shape_predictor_68_face_landmark['mouth']
    # signal_detect()
    detect_time(mouth_range)