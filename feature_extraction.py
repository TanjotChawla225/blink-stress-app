import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import entropy

mp_face = mp.solutions.face_mesh

def extract_aperture_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    aperture = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            if not res.multi_face_landmarks:
                aperture.append(np.nan)
                continue

            lm = res.multi_face_landmarks[0]
            h, w = frame.shape[:2]

            upper = [386, 387, 388]
            lower = [159, 145, 153]

            uy = np.mean([lm.landmark[i].y for i in upper]) * h
            ly = np.mean([lm.landmark[i].y for i in lower]) * h

            aperture.append(ly - uy)

    cap.release()
    return np.array(aperture), fps


def extract_blink_features(ap, fps):

    ap = np.nan_to_num(ap, nan=np.nanmean(ap))

    smooth = savgol_filter(ap, 11, 3)
    inv = -smooth

    peaks, _ = find_peaks(inv, height=np.std(inv)*0.5, distance=int(0.1*fps))

    blink_times = peaks / fps
    blink_durations = []
    blink_amplitudes = []

    for p in peaks:
        blink_amplitudes.append(inv[p])
        blink_durations.append(0.06)  # approx 60ms

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
