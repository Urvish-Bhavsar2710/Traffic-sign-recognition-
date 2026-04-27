import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate
from keras.utils import to_categorical
import glob
import random
import matplotlib.pyplot as plt


# Parameters
img_size = (32, 32)
data_dir = "Train"
images, labels = [], []


# Load and preprocess dataset
for class_id in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_id)
    if not os.path.isdir(class_path): continue
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.resize(img, img_size)
        #img = augment(image=img)['image']
        edge = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)
        edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(img, 0.8, edge, 0.2, 0)
        images.append(blended)
        labels.append(int(class_id))

# Prepare dataset
X = np.array(images).astype("float32") / 255.0
y = np.array(labels)
y_cat = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# PolarCNN architecture
input_layer = Input(shape=(32, 32, 3))
x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
x1 = MaxPooling2D(2, 2)(x1)

x2 = Conv2D(32, (5, 5), activation='relu', padding='same')(input_layer)
x2 = MaxPooling2D(2, 2)(x2)

x3 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)
x3 = MaxPooling2D(2, 2)(x3)

merged = concatenate([x1, x2, x3])
x = Conv2D(64, (3, 3), activation='relu')(merged)
x = MaxPooling2D(2, 2)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(y_cat.shape[1], activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, batch_size=32, epochs=15, validation_split=0.1)

# Evaluate model
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

acc = accuracy_score(y_true, y_pred_labels)
f1 = f1_score(y_true, y_pred_labels, average='weighted')
mse = mean_squared_error(y_true, y_pred_labels)
psnr = 20 * np.log10(1.0 / np.sqrt(mse))

print("\nPolarCNN Traffic Sign Detection Results by URVISH BHAVSAR:")
print(f"Accuracy: {acc * 100:.4f}%")
print(f"F1 Score: {f1:.4f}")
print(f"MSE: {mse:.4f}")
print(f"PSNR: {psnr:.2f} dB")

plt.figure()
plt.plot(history.history['accuracy'],label= 'Train Accuracy')
plt.plot(history.history['val_accuracy'],label= 'Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Epoch vs Accuracy find by URVISH BHAVSAR')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(history.history['loss'],label= 'Train loss')
plt.plot(history.history['val_loss'],label= 'Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch vs Loss find by URVISH BHAVSAR')
plt.legend()
plt.grid(True)
plt.show()

model.save("polar_cnn_model.h5")