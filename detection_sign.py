import cv2
import numpy as np
from keras.models import load_model
from skimage.transform import warp_polar
import serial
import os
import datetime
from sympy import true
import pygame
import pyttsx3

#serial connection to arduino
arduino = serial.Serial(port='COM3',baudrate=115200 , timeout=1)

# Load your trained PolarCNN model
model = load_model("polar_cnn_model.h5")

# list of 43 GTSRB class names
class_names = {
    0 :"Speed limit (20km/h)", 1:"Speed limit (30km/h)", 2:"Speed limit (50km/h)",
    3:"Speed limit (60km/h)", 4:"Speed limit (70km/h)", 5:"Speed limit (80km/h)",
    6:"End of speed limit (80km/h)", 7:"Speed limit (100km/h)", 8:"Speed limit (120km/h)",
    9:"No passing", 10:"No passing for vehicles over 3.5 metric tons", 11:"Right-of-way at intersection",
    12:"Priority road", 13:"Yield", 14:"Stop", 15:"No vehicles", 16:"Vehicles over 3.5 metric tons prohibited",
    17:"No entry", 18:"General caution", 19:"Dangerous curve to the left", 20:"Dangerous curve to the right",
    21:"Double curve", 22:"Bumpy road", 23:"Slippery road", 24:"Road narrows on the right", 25:"Road work",
    26:"Traffic signals", 27:"Pedestrians", 28:"Children crossing", 29:"Bicycles crossing", 30:"Beware of ice/snow",
    31:"Wild animals crossing", 32:"End of all speed and passing limits", 33:"Turn right ahead",
    34:"Turn left ahead", 35:"Ahead only", 36:"Go straight or right", 37:"Go straight or left", 38:"Keep right",
    39:"Keep left", 40:"Roundabout mandatory", 41:"End of no passing", 42:"End of no passing by vehicles > 3.5 tons"
}

# Load test image
#image = cv2.imread("00000_00005_00027.png")
#image = cv2.imread("00001_00000_00000.png")
#image = cv2.imread("00002_00004_00029.png")
#image = cv2.imread("00003_00044_00028.png")
#image = cv2.imread("00004_00000_00017.png")
#image = cv2.imread("00005_00058_00021.png")
#image = cv2.imread("00007_00013_00026.png")
#image = cv2.imread("00008_00039_00017.png")
#image = cv2.imread("00009_00000_00009.png")
image = cv2.imread("00010_00057_00021.png")
#image = cv2.imread("00011_00002_00019.png")
#image = cv2.imread("00012_00000_00027.png")
#image = cv2.imread("00013_00000_00029.png")
#image = cv2.imread("00014_00000_00024.png")
#image = cv2.imread("00015_00015_00027.png")
#image = cv2.imread("00016_00000_00002.png")
#image = cv2.imread("00017_00021_00029.png")
#image = cv2.imread("00018_00000_00011.png")
#image = cv2.imread("00019_00000_00008.png")
# .image = cv2.imread("00020_00000_00015.png")
#image = cv2.imread("00021_00000_00023.png")
#image = cv2.imread("00022_00006_00026.png")
#image = cv2.imread("00023_00015_00028.png")
#image = cv2.imread("00024_00002_00025.png")
#image = cv2.imread("00025_00000_00027.png")
#image = cv2.imread("00026_00000_00010.png")
#image = cv2.imread("00027_00001_00026.png")
#image = cv2.imread("00028_00002_00027.png")
# . image = cv2.imread("00029_00001_00027.png")
#image = cv2.imread("00030_00004_00022.png")
#.image = cv2.imread("00031_00009_00023.png")
#image = cv2.imread("00032_00003_00029.png")
#.image = cv2.imread("00033_00000_00009.png")
#image = cv2.imread("00034_00000_00017.png")
#image = cv2.imread("00035_00003_00025.png")
#image = cv2.imread("00036_00008_00025.png")
#image = cv2.imread("00037_00001_00026.png")
#image = cv2.imread("00038_00015_00022.png")
#image = cv2.imread("00039_00000_00016.png")
#image = cv2.imread("00040_00003_00025.png")
#image = cv2.imread("00041_00000_00025.png")
#image = cv2.imread("00042_00000_00000.png")


#image = cv2.imread("00044.png")
#image = cv2.imread("00045.png")
#image = cv2.imread("000461.png")
#image = cv2.imread("00462.png")

zoomed_image = cv2.resize(image,(512,512), interpolation=cv2.INTER_LINEAR)

# Canny edge detection
gray = cv2.cvtColor(zoomed_image, cv2.COLOR_BGR2GRAY)
h,w  = gray.shape
center = (w//2 , h//2)
#max readius 
max_radius = np.sqrt((w/2)**2 + (h/2)**2)
polar_img = cv2.warapPolar(gray,(w,h),center, max_radius,cv2.WRAP_FILL_OUTLIERS)
_,edges = cv2.threshold(gray, 120, 255,cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    print("no countours find")
    exit()

contours = sorted(contours, key=cv2.contourArea , reverse=True)
largest_cnt = contours[0]
# Loop over all detected contours

(x_center , y_center) , radius =  cv2.minEnclosingCircle(largest_cnt)
    

padding = int(radius * 2)
x1 = int(max(0,x_center - radius ))
y1 = int(max(0,y_center - radius ))
x2 = int(min(zoomed_image.shape[1],x_center + radius ))
y2 = int(min(zoomed_image.shape[0],y_center + radius ))
     

roi = zoomed_image[y1:y2 , x1:x2]

rect = cv2.minAreaRect(largest_cnt)
roi_resized = cv2.resize(roi, (32, 32)).astype("float32")/255.0
roi_input = np.expand_dims(roi_resized , axis=0)

pred = model.predict(roi_input)[0]
class_id = int(np.argmax(pred))
confidence = float(np.max(pred) ) 
label= f" {class_names[class_id]}"

# arduino rgb module
#red - prohibitory
if 0 <=class_id <=10:
    r,g,b = 255,0,0

#blue - warning    
elif 11 <= class_id <= 31:
     r,g,b = 0,255,0
    
#green- mandotaroy
elif 32 <= class_id <= 40:
     r,g,b = 0,0,255

else:
    r, g, b = 40,40,40

arduino.write(f"{r},{g},{b}\n".encode())
    
# sounds
print("Initializing pygame...")
pygame.mixer.init()

print("Loading sounds...")
sound_map = {}
for i in range(43):
    try:
        sound = pygame.mixer.Sound(f"sounds/{i}.wav")
        sound.set_volume(1.0)
        sound_map[i] = sound
    except:
        print(f"Missing sound: sounds/{i}.wav")

print("Initializing voice...")
engine = pyttsx3.init('sapi5')
engine.setProperty('volume',5.0)
engine.setProperty('rate',800)

voices = engine.getProperty('voices')
engine.setProperty('voice',voices[0].id)

def play_sound(class_id):
    if class_id in sound_map:
        sound_map[class_id].play()

def speak_text(text):
    engine.stop()
    engine.say(text)
    engine.runAndWait()

play_sound(class_id)


# bounding box and text
cv2.rectangle(zoomed_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

labe_size, _ = cv2.getTextSize(label , cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
label_w , label_h = labe_size
text_y = y1 - 5 if y1 - 5 > label_h else y1 + label_h + 5
cv2.rectangle(zoomed_image, (x1,  text_y - label_h - 5),(x1 + label_w +  10, text_y + 5),(0,0,0),-1) 
cv2.putText(zoomed_image, label, (x1 + 5 , text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

cv2.imshow("Detected Traffic Signs", zoomed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()