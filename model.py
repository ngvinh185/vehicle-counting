from ultralytics import YOLO
import supervision as sv
from tqdm import tqdm
import numpy as np
# from supervision.draw.color import ColorPalette
# from supervision.geometry.dataclasses import Point
# from supervision.video.dataclasses import VideoInfo
# from supervision.video.source import get_video_frames_generator
# from supervision.video.sink import VideoSink
# from supervision.notebook.utils import show_frame_in_notebook
# from supervision.tools.detections import Detections, BoxAnnotator
# from supervision.tools.line_counter import LineCounter, LineCounterAnnotator

model = YOLO('yolov8s.pt')
model.fuse()
video_path = 'vehicle-counting.mp4'
video_info = sv.VideoInfo.from_video_path(video_path) #VideoInfo(width=1280, height=720, fps=30, total_frames=329)
# print(video_info.width)
label_annotator = sv.LabelAnnotator()
line_counter = sv.LineZone(start = sv.Point(0, video_info.height // 3 * 2), end = sv.Point(video_info.width, video_info.height // 3 * 2))
box_annotator = sv.BoxAnnotator()
line_annotator = sv.LineZoneAnnotator()
track = sv.ByteTrack()
video_path_target = 'video1.mp4'
vid_gen = sv.get_video_frames_generator(video_path)
cls_name = model.names
cls_ids = [2, 3, 5, 7]
with sv.VideoSink(video_path_target, video_info) as sink:
  for frame in tqdm(vid_gen, total = video_info.total_frames):
    results = model(source = frame)
    result = results[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = track.update_with_detections(detections)
    detections = detections[np.isin(detections.class_id, cls_ids)] # add np condition
    labels = [f'{track_id}, {cls_name[cls_id]}, {conf:.2f}' for cls_id, track_id, conf
    in zip(
        detections.class_id,
        detections.tracker_id,
        detections.confidence
    )]
    #ko can frame = cx dc vi no da viet len frame r
    frame = box_annotator.annotate(frame, detections = detections)
    frame = label_annotator.annotate(frame, detections=detections, labels=labels)
    line_counter.trigger(detections = detections)
    frame = line_annotator.annotate(frame, line_counter = line_counter)
    sink.write_frame(frame)
