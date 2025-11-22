import cv2
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import entropy

# Eye landmark indices
LEFT_EYE = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]

# EAR FUNCTION
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def extract_aperture_from_video(video_path):
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    landmark_model = cv2.face.createFacemarkLBF()
    landmark_model.loadModel(cv2.data.haarcascades + "lbfmodel.yaml")

    cap = cv2.VideoCapture(video_path)
    aperture = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            aperture.append(np.nan)
            continue

        _, landmarks = landmark_model.fit(gray, faces)

        if len(landmarks) == 0:
            aperture.append(np.nan)
            continue

        shape = landmarks[0][0]

        left_eye = shape[LEFT_EYE]
        right_eye = shape[RIGHT_EYE]

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        aperture.append(ear)

    cap.release()
    return np.array(aperture), fps


def extract_blink_features(ap, fps):
    ap = np.nan_to_num(ap, nan=np.nanmean(ap))
    smooth = savgol_filter(ap, 11, 3)
    inv = -smooth

    peaks, _ = find_peaks(inv, height=np.std(inv)*0.5, distance=int(fps*0.1))

    blink_times = peaks / fps
    blink_durations = []
    blink_amplitudes = []

    for p in peaks:
        blink_amplitudes.append(inv[p])
        blink_durations.append(0.06)

    if len(blink_times) < 2:
        ibi = [0]
    else:
        ibi = np.diff(blink_times)

    feats = {
        'blink_rate_per_min': len(peaks) / (len(ap)/fps) * 60,
        'ibi_mean': float(np.mean(ibi)),
        'ibi_std': float(np.std(ibi)),
        'ibi_entropy': float(entropy(np.histogram(ibi, bins=5)[0] + 1e-6)),
        'blink_duration_mean': float(np.mean(blink_durations)),
        'blink_duration_std': float(np.std(blink_durations)),
        'blink_amplitude_mean': float(np.mean(blink_amplitudes)),
        'blink_amplitude_std': float(np.std(blink_amplitudes)),
        'blink_regularity': float(1/(np.std(ibi)+1e-5))
    }

    return feats
