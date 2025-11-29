import cv2
import dlib
# 打开摄像头
capture = cv2.VideoCapture(0)
 
# 获取人脸检测器
detector = dlib.get_frontal_face_detector()
 
# 获取人脸关键点检测模型
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor("models\pretrained\shape_predictor_68_face_landmarks.dat")

while True:
    # 读取视频流
    ret, frame = capture.read()
    # 灰度转换
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 人脸检测
    faces = detector(gray, 1)
    # 绘制每张人脸的矩形框和关键点
    for face in faces:
        # 绘制矩形框
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0,255,0), 3)
        # 检测到关键点
        shape = predictor(gray, face)  #68个关键点
        # 获取关键点的坐标
        for pt in shape.parts():
            # 每个点的坐标
            pt_position = (pt.x, pt.y)
            # 绘制关键点
            cv2.circle(frame, pt_position, 3, (255,255,255), -1)
    cv2.imshow("face detection landmark", frame)
            #设置退出按钮
    key_pressed = cv2.waitKey(100)
    if key_pressed == 27:
        break
capture.release()
cv2.destroyAllWindows()