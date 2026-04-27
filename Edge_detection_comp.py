import cv2 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score , root_mean_squared_error 
from skimage.metrics import peak_signal_noise_ratio as psnr

#-------------- LIVE CAMERA CAPTURE BY URVISH LAPTOP ----------
def capture_image():
    cap = cv2.VideoCapture(0)
    print("Capturing image... Press SPACE to capture.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Live Feed - Press SPACE to Capture", frame)
        key = cv2.waitKey(1)
        if key == 32:  # SPACE key
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (128, 128))
            break
    cap.release()
    cv2.destroyAllWindows()
    return image

image = capture_image()
#image = cv2.imread('images.jpeg')
gt_edge = cv2.Canny(image, 100, 200)  # use Canny as pseudo ground truth

# ---------- EDGE DETECTION METHODS ---------- #  sobel , prewitt , canny ---------
def apply_edge_methods(img):
    sobel = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title('sobel')
    plt.imshow(sobel)

    prewitt_x = cv2.filter2D(img, -1, np.array([[1,0,-1],[1,0,-1],[1,0,-1]]))
    prewitt_y = cv2.filter2D(img, -1, np.array([[1,1,1],[0,0,0],[-1,-1,-1]]))
    prewitt = prewitt_x + prewitt_y
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title('prewitt')
    plt.imshow(prewitt)

    canny = cv2.Canny(img,100,100)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title('canny')
    plt.imshow(canny)
    return {'Sobel': sobel, 'Prewitt': prewitt, 'Canny': canny}
    
def compute_metrics(name, pred, gt):
    pred_bin = (pred > 50).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)
    acc = accuracy_score(gt_bin.flatten(), pred_bin.flatten())
    prec = precision_score(gt_bin.flatten(), pred_bin.flatten(), zero_division=0)
    f1 = f1_score(gt_bin.flatten(), pred_bin.flatten(), zero_division=0)
    return [name, acc, prec, f1]

results = []  

classical_edges = apply_edge_methods(image)
for method, edge in classical_edges.items():
    results.append(compute_metrics(method, edge,gt_edge))

def compute_metrics(name, pred, gt):
    rmse_val = root_mean_squared_error(gt, pred)
    psnr_val = psnr(gt, pred.astype(np.uint8))
    return [name, rmse_val,psnr_val]

results1 = []

for method, edge in classical_edges.items():
    results1.append(compute_metrics(method, edge,gt_edge))

# ---------- EXPORT AND PLOT ----------
df = pd.DataFrame(results, columns=['Method', 'Accuracy', 'Precision', 'F1 Score'])
df.to_csv('edge_detection_live_metrics.csv', index=False)
print(df)

df.plot(x='Method', kind='bar', figsize=(12, 6), title='Live Edge Detection Method Comparison by URVISH BHAVSAR', colormap='plasma')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

df1 = pd.DataFrame(results1, columns=['Method', 'RMSE' , 'PSNR'])
df1.to_csv('edge_detection_live_metrics_by_urvish_bhavsar.csv', index=False)
print(df1)

df1.plot(x='Method', kind='line', figsize=(12,6), title='Live Edge Detection Method Comparison by URVISH BHAVSAR', colormap='plasma')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()