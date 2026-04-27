import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from skimage.transform import warp_polar

# CONFIGURATION    
IMG_SIZE = 32
NUM_CLASSES = 43   

cnn_model = load_model("code1_cnn_model.h5")
polar_model = load_model("code2_polar_cnn_model.h5")
print("Models Loaded Successfully by URVISH BHAVSAR")
# =========================
# LOAD & PREPROCESS IMAGE
img = cv2.imread("00041_00000_00025.png" )

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
img_norm = img_rgb / 255.0

# =========================
# CNN PREDICTION
cnn_input = np.expand_dims(img_norm, axis=0)
cnn_pred = cnn_model.predict(cnn_input, verbose=0)[0]

cnn_class = np.argmax(cnn_pred)
cnn_conf = np.max(cnn_pred)

# =========================
# POLAR CNN PREPROCESSING
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

polar = warp_polar(gray, scaling='log')
polar = cv2.resize(polar, (IMG_SIZE, IMG_SIZE))

polar = cv2.cvtColor((polar * 255).astype("uint8"), cv2.COLOR_GRAY2BGR)
polar = polar.astype("float32") / 255.0
polar_input =  np.expand_dims(polar, axis=0)

# =========================
# POLAR CNN PREDICTION
polar_pred = polar_model.predict(polar_input, verbose=0)[0]

polar_class = np.argmax(polar_pred)
polar_conf = np.max(polar_pred)

# =========================
# PRINT RESULTS
print("\n=========== RESULT COMPARISON BY URVISH BHAVSAR ==========")
print(f"CNN Prediction      → Class: {cnn_class}, Confidence: {cnn_conf:.3f}")
print(f"Polar-CNN Prediction→ Class: {polar_class}, Confidence: {polar_conf:.3f}")

if polar_conf > cnn_conf:
    print("Polar-CNN performs BETTER than CNN")
else:
    pass

# =========================
# VISUAL COMPARISON
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img_rgb)
plt.title("Input Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(img_rgb)
plt.title(f"CNN Output\nClass: {cnn_class}\nConfidence: {cnn_conf:.2f}")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(img_rgb)
plt.title(f"Polar-CNN Output\nClass: {polar_class}\nConfidence: {polar_conf:.2f}")
plt.axis("off")

plt.tight_layout()
plt.show()
