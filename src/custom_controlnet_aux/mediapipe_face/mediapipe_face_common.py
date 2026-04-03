import os
from typing import Mapping

import mediapipe as mp
import numpy

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    FaceLandmarksConnections,
    RunningMode,
    drawing_utils,
)
from mediapipe.tasks.python.vision.drawing_utils import DrawingSpec

_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
    "ckpts", "mediapipe", "face_landmarker.task",
)

min_face_size_pixels: int = 64
f_thick = 2
f_rad = 1
right_iris_draw = DrawingSpec(color=(10, 200, 250), thickness=f_thick, circle_radius=f_rad)
right_eye_draw = DrawingSpec(color=(10, 200, 180), thickness=f_thick, circle_radius=f_rad)
right_eyebrow_draw = DrawingSpec(color=(10, 220, 180), thickness=f_thick, circle_radius=f_rad)
left_iris_draw = DrawingSpec(color=(250, 200, 10), thickness=f_thick, circle_radius=f_rad)
left_eye_draw = DrawingSpec(color=(180, 200, 10), thickness=f_thick, circle_radius=f_rad)
left_eyebrow_draw = DrawingSpec(color=(180, 220, 10), thickness=f_thick, circle_radius=f_rad)
mouth_draw = DrawingSpec(color=(10, 180, 10), thickness=f_thick, circle_radius=f_rad)
head_draw = DrawingSpec(color=(10, 200, 10), thickness=f_thick, circle_radius=f_rad)

face_connection_spec: dict[tuple[int, int], DrawingSpec] = {}
_face_connections: list = []
for _conn in FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL:
    face_connection_spec[(_conn.start, _conn.end)] = head_draw
    _face_connections.append(_conn)
for _conn in FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE:
    face_connection_spec[(_conn.start, _conn.end)] = left_eye_draw
    _face_connections.append(_conn)
for _conn in FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW:
    face_connection_spec[(_conn.start, _conn.end)] = left_eyebrow_draw
    _face_connections.append(_conn)
for _conn in FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE:
    face_connection_spec[(_conn.start, _conn.end)] = right_eye_draw
    _face_connections.append(_conn)
for _conn in FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW:
    face_connection_spec[(_conn.start, _conn.end)] = right_eyebrow_draw
    _face_connections.append(_conn)
for _conn in FaceLandmarksConnections.FACE_LANDMARKS_LIPS:
    face_connection_spec[(_conn.start, _conn.end)] = mouth_draw
    _face_connections.append(_conn)

iris_landmark_spec = {468: right_iris_draw, 473: left_iris_draw}


def draw_pupils(image, landmarks, drawing_spec, halfwidth: int = 2):
    """Custom pupil drawing — the standard draw_landmarks doesn't support per-landmark specs
    for individual iris points."""
    if len(image.shape) != 3:
        raise ValueError("Input image must be H,W,C.")
    image_rows, image_cols, image_channels = image.shape
    if image_channels != 3:  # BGR channels
        raise ValueError('Input image must contain three channel bgr data.')
    for idx, landmark in enumerate(landmarks):
        if landmark.visibility is not None and landmark.visibility < 0.9:
            continue
        if landmark.presence is not None and landmark.presence < 0.5:
            continue
        if landmark.x >= 1.0 or landmark.x < 0 or landmark.y >= 1.0 or landmark.y < 0:
            continue
        image_x = int(image_cols * landmark.x)
        image_y = int(image_rows * landmark.y)
        draw_color = None
        if isinstance(drawing_spec, Mapping):
            if drawing_spec.get(idx) is None:
                continue
            else:
                draw_color = drawing_spec[idx].color
        elif isinstance(drawing_spec, DrawingSpec):
            draw_color = drawing_spec.color
        image[image_y-halfwidth:image_y+halfwidth, image_x-halfwidth:image_x+halfwidth, :] = draw_color


def reverse_channels(image):
    """Given a numpy array in RGB form, convert to BGR.  Will also convert from BGR to RGB."""
    return image[:, :, ::-1]


def generate_annotation(
        img_rgb,
        max_faces: int,
        min_confidence: float
):
    """
    Find up to 'max_faces' inside the provided input image.
    If min_face_size_pixels is provided and nonzero it will be used to filter faces that occupy less than this many
    pixels in the image.
    """
    model_path = os.path.abspath(_MODEL_PATH)
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_faces=max_faces,
        min_face_detection_confidence=min_confidence,
        min_face_presence_confidence=min_confidence,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        img_height, img_width, img_channels = img_rgb.shape
        assert img_channels == 3

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_image)

        if not result.face_landmarks:
            print("No faces detected in controlnet image for Mediapipe face annotator.")
            return numpy.zeros_like(img_rgb)

        filtered_landmarks = []
        for face_lms in result.face_landmarks:
            face_rect = [face_lms[0].x, face_lms[0].y, face_lms[0].x, face_lms[0].y]
            for lm in face_lms:
                face_rect[0] = min(face_rect[0], lm.x)
                face_rect[1] = min(face_rect[1], lm.y)
                face_rect[2] = max(face_rect[2], lm.x)
                face_rect[3] = max(face_rect[3], lm.y)
            if min_face_size_pixels > 0:
                face_width = abs(face_rect[2] - face_rect[0])
                face_height = abs(face_rect[3] - face_rect[1])
                face_width_pixels = face_width * img_width
                face_height_pixels = face_height * img_height
                face_size = min(face_width_pixels, face_height_pixels)
                if face_size >= min_face_size_pixels:
                    filtered_landmarks.append(face_lms)
            else:
                filtered_landmarks.append(face_lms)

        empty = numpy.zeros_like(img_rgb)

        for face_lms in filtered_landmarks:
            drawing_utils.draw_landmarks(
                empty,
                face_lms,
                connections=_face_connections,
                landmark_drawing_spec=None,
                connection_drawing_spec=face_connection_spec,
                is_drawing_landmarks=False,
            )
            draw_pupils(empty, face_lms, iris_landmark_spec, 2)

        empty = reverse_channels(empty).copy()

        return empty
