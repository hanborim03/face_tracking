from retinaface import RetinaFace

def detect_faces(frame):
    # frame: numpy array (BGR)
    # 반환값: 얼굴 bounding box 리스트 [x1, y1, x2, y2]
    results = RetinaFace.detect_faces(frame)
    boxes = []
    for face in results.values():
        x1, y1, x2, y2 = face['facial_area']
        boxes.append([x1, y1, x2, y2])
    return boxes
