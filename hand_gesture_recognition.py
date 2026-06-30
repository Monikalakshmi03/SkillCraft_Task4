
# Hand Gesture Recognition using CNN (TensorFlow/Keras)
# Folder structure:
# handgestures/
# ├── fist
# ├── fist_moved
# ├── l
# └── palm

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

DATASET_PATH = "handgestures"   # change if needed
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 15

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

print("\nClasses:", train_data.class_indices)

model = Sequential([
    Conv2D(32,(3,3),activation="relu",input_shape=(64,64,3)),
    MaxPooling2D(2,2),

    Conv2D(64,(3,3),activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128,(3,3),activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128,activation="relu"),
    Dropout(0.5),
    Dense(train_data.num_classes,activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining Model...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

loss, acc = model.evaluate(val_data, verbose=0)
print(f"\nValidation Accuracy : {acc*100:.2f}%")

# Accuracy/Loss Graph
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"],label="Train")
plt.plot(history.history["val_accuracy"],label="Validation")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"],label="Train")
plt.plot(history.history["val_loss"],label="Validation")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("accuracy_loss.png")
plt.show()

# Predictions
pred = model.predict(val_data)
pred_classes = np.argmax(pred, axis=1)
true_classes = val_data.classes

print("\nClassification Report\n")
print(classification_report(
    true_classes,
    pred_classes,
    target_names=list(val_data.class_indices.keys())
))

cm = confusion_matrix(true_classes,pred_classes)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=list(val_data.class_indices.keys())
)
disp.plot(cmap="Blues",xticks_rotation=45)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# Sample Predictions
labels = list(val_data.class_indices.keys())

images, truths = next(val_data)
preds = model.predict(images)

plt.figure(figsize=(10,10))
for i in range(min(9,len(images))):
    plt.subplot(3,3,i+1)
    plt.imshow(images[i])
    true_label = labels[np.argmax(truths[i])]
    pred_label = labels[np.argmax(preds[i])]
    color = "green" if true_label == pred_label else "red"
    plt.title(pred_label,color=color)
    plt.axis("off")

plt.suptitle("Sample Predictions")
plt.tight_layout()
plt.savefig("sample_predictions.png")
plt.show()

model.save("hand_gesture_model.h5")
print("\nModel saved as hand_gesture_model.h5")
print("Task 4 Completed Successfully!")
