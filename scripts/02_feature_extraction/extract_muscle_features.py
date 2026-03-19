# NOTE:
# The features extracted here are intensity-based proxies, not clinically accurate
# measurements of skeletal muscle. This approach uses fixed grayscale thresholds
# on PNG images, which do not preserve true CT Hounsfield Unit (HU) values.
# As a result, the “muscle” regions may be imprecise, and the computed metrics
# (muscle_ratio, muscle_density) should be interpreted as approximate signals
# rather than validated clinical indicators of sarcopenia.

import os
import cv2
import numpy as np
import pandas as pd

data_dir = "CT-Scan"

rows = []

for patient in os.listdir(data_dir):

    patient_path = os.path.join(data_dir, patient)

    if not os.path.isdir(patient_path):
        continue

    muscle_pixels = 0
    total_pixels = 0
    muscle_intensity = []

    for file in os.listdir(patient_path):

        if not file.endswith(".png"):
            continue

        img_path = os.path.join(patient_path, file)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (256,256))

        # approximate skeletal muscle
        mask = (img > 80) & (img < 200)

        muscle_pixels += np.sum(mask)
        total_pixels += img.size

        muscle_intensity.append(np.mean(img[mask]))

    if total_pixels == 0:
        continue

    rows.append({
        "Patient_ID": patient,
        "muscle_ratio": muscle_pixels / total_pixels,
        "muscle_density": np.mean(muscle_intensity)
    })

features = pd.DataFrame(rows)

features.to_csv("muscle_features.csv", index=False)
