import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Paths
DATASET_DIR = "dataset"
MODEL_SAVE_PATH = "saved_models/recycle_mobilenetv2.keras"
IMG_SIZE = (180, 180)
BATCH_SIZE = 16
EPOCHS = 10

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)
val_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Base model
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base

# Custom head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy'
)
earlystop_cb = tf.keras.callbacks.EarlyStopping(
    patience=5, restore_best_weights=True, monitor='val_accuracy'
)

# Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, earlystop_cb]
)

print(f"Training complete. Model saved to {MODEL_SAVE_PATH}")
