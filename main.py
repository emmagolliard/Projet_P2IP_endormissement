from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import numpy as np
import math

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# Indices des points du visage
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]
NOSE = 1
CHIN = 152

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def get_point(landmarks, index, w, h):
    lm = landmarks[index]
    return (int(lm.x * w), int(lm.y * h))

def eye_ratio(landmarks, eye, w, h):
    p1 = get_point(landmarks, eye[0], w, h)
    p2 = get_point(landmarks, eye[1], w, h)
    p3 = get_point(landmarks, eye[2], w, h)
    p4 = get_point(landmarks, eye[3], w, h)
    p5 = get_point(landmarks, eye[4], w, h)
    p6 = get_point(landmarks, eye[5], w, h)

    vertical = distance(p2, p6) + distance(p3, p5)
    horizontal = 2 * distance(p1, p4)

    return vertical / horizontal if horizontal != 0 else 0

def mouth_ratio(landmarks, w, h):
    top = get_point(landmarks, MOUTH[0], w, h)
    bottom = get_point(landmarks, MOUTH[1], w, h)
    left = get_point(landmarks, MOUTH[2], w, h)
    right = get_point(landmarks, MOUTH[3], w, h)

    vertical = distance(top, bottom)
    horizontal = distance(left, right)

    return vertical / horizontal if horizontal != 0 else 0

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "image invalide"}

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return {
            "face_detected": False,
            "drowsy": False
        }

    landmarks = result.multi_face_landmarks[0].landmark
    h, w = frame.shape[:2]

    ear_left = eye_ratio(landmarks, LEFT_EYE, w, h)
    ear_right = eye_ratio(landmarks, RIGHT_EYE, w, h)
    ear = (ear_left + ear_right) / 2

    mar = mouth_ratio(landmarks, w, h)

    # seuils simples
    eyes_closed = ear < 0.22
    yawning = mar > 0.6

    score = 0
    if eyes_closed:
        score += 0.6
    if yawning:
        score += 0.4

    return {
        "face_detected": True,
        "drowsy": score > 0.5,
        "score": round(score, 2),
        "eyes_closed": eyes_closed,
        "yawning": yawning
    }