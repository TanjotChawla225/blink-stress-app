import cv2
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import entropy

# Load Haar cascades (built into OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def extract_aperture_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    aperture = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            aperture.append(np.nan)
            continue

        (x, y, w, h) = faces[0]
        roi = gray[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi)
        if len(eyes) < 2:   # need two eyes
            aperture.append(np.nan)
            continue

        # Pick two largest eyes
        eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]

        eye_heights = []
        for (ex, ey, ew, eh) in eyes:
            eye_heights.append(eh / ew)

        ear = np.mean(eye_heights)
        aperture.append(ear)

    cap.release()
    return np.array(aperture), fps


def extract_blink_features(ap, fps):
    ap = np.nan_to_num(ap, nan=np.nanmean(ap))

    smooth = savgol_filter(ap, 11, 3)
    inv = -smooth

    peaks, _ = find_peaks(inv, height=np.std(inv)*0.6, distance=int(fps*0.1))

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
