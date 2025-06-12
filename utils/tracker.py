from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort(max_age=30)

def update_tracks(detections, frame):
    # detections: [[x, y, w, h, conf, embedding], ...]
    tracks = tracker.update_tracks(detections, frame=frame)
    return tracks
