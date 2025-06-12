import insightface
import cv2

# GPU 우선, CPU 백업 실행 프로바이더 명시
app = insightface.app.FaceAnalysis(
    name='buffalo_l',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
app.prepare(ctx_id=0, det_size=(640, 640))

def get_embedding(face_img):
    """
    얼굴 crop(BGR 이미지)에서 512차원 얼굴 임베딩 벡터를 추출합니다.
    얼굴이 검출되지 않으면 None을 반환합니다.
    """
    faces = app.get(face_img)
    if faces:
        return faces[0].embedding  # 512차원 임베딩 벡터
    else:
        return None
