import cv2
from utils.detector import detect_faces
from utils.embedder import get_embedding
from utils.tracker import update_tracks
import os

print(os.getcwd())
print(os.path.isfile('utils/data/mixkit-group-of-friends-partying-happily-4640-hd-ready.mp4'))

VIDEO_PATH = 'utils/data/mixkit-group-of-friends-partying-happily-4640-hd-ready.mp4'

if not os.path.isfile(VIDEO_PATH):
    print(f"[에러] 영상 파일이 존재하지 않습니다: {VIDEO_PATH}")
    exit()

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"[에러] 영상 파일을 열 수 없습니다: {VIDEO_PATH}")
    exit()

print("[정보] 영상이 정상적으로 열렸습니다.")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"[정보] 총 프레임 수: {frame_count}")

ASPECT_W = 4
ASPECT_H = 5

frame_saved = False  # 한 번만 이미지 저장

while True:
    ret, frame = cap.read()
    if not ret:
        print("[정보] 더 이상 읽을 프레임이 없습니다. 종료합니다.")
        break

    h, w, _ = frame.shape

    # 1. 얼굴 검출
    boxes = detect_faces(frame)
    print(f"[디버그] 검출된 얼굴 개수: {len(boxes)}")
    detections = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        orig_w = x2 - x1
        orig_h = y2 - y1

        new_h = orig_h
        new_w = int(new_h * ASPECT_W / ASPECT_H)
        if new_w < orig_w:
            new_w = orig_w
            new_h = int(new_w * ASPECT_H / ASPECT_W)

        nx1 = max(min(cx - new_w // 2, w - 2), 0)
        ny1 = max(min(cy - new_h // 2, h - 2), 0)
        nx2 = max(min(cx + new_w // 2, w - 1), nx1 + 1)
        ny2 = max(min(cy + new_h // 2, h - 1), ny1 + 1)

        print(f"[디버그] 박스 좌표: ({nx1},{ny1})-({nx2},{ny2}), 크기: {nx2-nx1}x{ny2-ny1}")

        # 얼굴 검출 결과에 바로 박스 그리기 (빨간색)
        cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), (0, 0, 255), 2)

        face_crop = frame[ny1:ny2, nx1:nx2]
        emb = get_embedding(face_crop)
        if emb is not None:
            detections.append([nx1, ny1, nx2 - nx1, ny2 - ny1, 0.99, emb])

    # 2. 얼굴 추적 및 박스 그리기 (트래킹 결과도 표시하고 싶으면 아래 코드 유지)
    tracks = update_tracks(detections, frame)
    for track in tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        track_w = x2 - x1
        track_h = y2 - y1
        new_h = track_h
        new_w = int(new_h * ASPECT_W / ASPECT_H)
        if new_w < track_w:
            new_w = track_w
            new_h = int(new_w * ASPECT_H / ASPECT_W)
        nx1 = max(min(cx - new_w // 2, w - 2), 0)
        ny1 = max(min(cy - new_h // 2, h - 2), 0)
        nx2 = max(min(cx + new_w // 2, w - 1), nx1 + 1)
        ny2 = max(min(cy + new_h // 2, h - 1), ny1 + 1)

        if nx2 > nx1 and ny2 > ny1:
            cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), (0,255,0), 2)
            track_id = track.track_id
            cv2.putText(frame, f"ID {track_id}", (nx1, max(ny1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            print(f"[경고] 잘못된 박스: ({nx1},{ny1})-({nx2},{ny2})")

    # 3. 영상 출력 (박스가 그려진 frame을 반드시 출력)
    cv2.imshow('Face Tracking', frame)

    # 4. 첫 프레임에 박스가 그려진 이미지를 파일로 저장
    if not frame_saved:
        cv2.imwrite("debug_output.jpg", frame)
        print("[디버그] debug_output.jpg 파일로 저장 완료 (박스가 보이는지 확인)")
        frame_saved = True

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[정보] 'q' 키 입력으로 종료합니다.")
        break

cap.release()
cv2.destroyAllWindows()
print("[정보] 프로그램이 정상적으로 종료되었습니다.")
