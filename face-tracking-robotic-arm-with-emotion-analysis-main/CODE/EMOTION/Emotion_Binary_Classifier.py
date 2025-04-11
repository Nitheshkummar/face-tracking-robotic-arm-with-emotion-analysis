import numpy as np
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dropout, Flatten, Dense, MaxPooling2D, Conv2D, Input, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# Path to the dataset
dataset_folder = 'CK+48-20250410T091656Z-001/CK+48'
print("Loading dataset from:", dataset_folder)

# Get all sub-folders
sub_folders = os.listdir(dataset_folder)
print("Found emotion folders:", sub_folders)

# Read images and labels
images = []
labels = []
skipped_folders = []

for sub_folder in sub_folders:
    sub_folder_index = sub_folders.index(sub_folder)
    label = sub_folder_index
    
    # Define labels for binary classification (positive:0, negative:1)
    # Skipping 'contempt', 'disgust', and 'fear' folders
    if label in [4, 6]:  # 'happy', 'surprise'
        new_label = 0    # positive emotion
    elif label in [0, 5]:  # 'anger', 'sadness'
        new_label = 1    # negative emotion
    else:  # 'contempt', 'disgust', 'fear'
        skipped_folders.append(sub_folder)
        continue  # Skip these folders
    
    path = os.path.join(dataset_folder, sub_folder)
    sub_folder_images = os.listdir(path)
    
    print(f"Processing {sub_folder} images ({len(sub_folder_images)} files)...")
    
    # Reading images in the sub folder, one at a time
    for image_name in sub_folder_images:
        image_path = os.path.join(path, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (48, 48))
        images.append(image)
        labels.append(new_label)

print(f"Skipped folders (contempt, disgust, fear): {skipped_folders}")

# Convert to numpy arrays
images_x = np.array(images)
labels_y = np.array(labels)

# Normalize pixel values to [0,1]
images_x = images_x/255.0

print(f"Dataset loaded: {images_x.shape} images with {len(np.unique(labels_y))} emotion classes")

# Reshape for PCA
n_samples, height, width = images_x.shape
X_flat = images_x.reshape(n_samples, height * width)

# Apply PCA for feature extraction
print("Applying PCA for feature extraction...")
# Keep 95% of variance
pca = PCA(n_components=0.95, svd_solver='full')
X_pca = pca.fit_transform(X_flat)
print(f"Original features: {height * width}, PCA features: {X_pca.shape[1]}")

# Save PCA model for later use
import pickle
with open('emotion_pca_model_binary.pkl', 'wb') as f:
    pickle.dump(pca, f)

# Encode the labels (for binary classification)
num_of_classes = 2
labels_y_encoded = tf.keras.utils.to_categorical(labels_y, num_classes=num_of_classes)

# Split into Train / Test
X_train, X_test, y_train, y_test = train_test_split(X_pca, labels_y_encoded, test_size=0.25, random_state=10)

# Convert PCA features back to image format for CNN
def reconstruct_from_pca(X_pca, pca, original_shape):
    # Reconstruct from PCA
    X_reconstructed = pca.inverse_transform(X_pca)
    # Reshape to original dimensions
    return X_reconstructed.reshape(-1, *original_shape)

X_train_img = reconstruct_from_pca(X_train, pca, (48, 48))
X_test_img = reconstruct_from_pca(X_test, pca, (48, 48))

# Reshape for CNN input (add channel dimension)
X_train_img = X_train_img.reshape(-1, 48, 48, 1)
X_test_img = X_test_img.reshape(-1, 48, 48, 1)

# Define a more efficient CNN model for Raspberry Pi
def create_optimized_model():
    # Input layer
    input_img = Input(shape=(48, 48, 1))
    
    # First Conv Block - fewer filters
    x = Conv2D(16, (3, 3), padding='same', kernel_regularizer=l2(0.001))(input_img)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)
    
    # Second Conv Block
    x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)
    
    # Third Conv Block
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)
    
    # Flatten and Dense layers
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output layer - only 2 classes now
    output = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=input_img, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create and display model
model = create_optimized_model()
model.summary()

# Set up callbacks for better training
checkpoint = ModelCheckpoint(
    'emotion_model_binary.h5',
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.00001
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

callbacks = [checkpoint, reduce_lr, early_stopping]

# Train the model
history = model.fit(
    X_train_img,
    y_train,
    batch_size=32,
    epochs=50,
    validation_data=(X_test_img, y_test),
    callbacks=callbacks
)

# Save the final model if early stopping didn't save the best one
model.save('emotion_model_binary_final.h5')

# Plot training history
train_loss = history.history['loss']
test_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
test_accuracy = history.history['val_accuracy']

fig, ax = plt.subplots(ncols=2, figsize=(15, 7))
ax = ax.ravel()

ax[0].plot(train_loss, label='Train Loss', color='royalblue', marker='o', markersize=4)
ax[0].plot(test_loss, label='Test Loss', color='orangered', marker='o', markersize=4)
ax[0].set_xlabel('Epochs', fontsize=14)
ax[0].set_ylabel('Categorical Crossentropy', fontsize=14)
ax[0].legend(fontsize=12)
ax[0].tick_params(axis='both', labelsize=12)

ax[1].plot(train_accuracy, label='Train Accuracy', color='royalblue', marker='o', markersize=4)
ax[1].plot(test_accuracy, label='Test Accuracy', color='orangered', marker='o', markersize=4)
ax[1].set_xlabel('Epochs', fontsize=14)
ax[1].set_ylabel('Accuracy', fontsize=14)
ax[1].legend(fontsize=12)
ax[1].tick_params(axis='both', labelsize=12)

fig.suptitle("Loss and Accuracy of Binary Emotion CNN Model by Epochs", fontsize=16)
plt.show()

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_img, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Convert to TFLite model for Raspberry Pi
print("Converting model to TFLite format...")

# Function to convert to TFLite
def convert_to_tflite(model, quantize=True):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        # Quantize model to reduce size and improve CPU performance
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        # Representative dataset for quantization
        def representative_dataset():
            for i in range(min(100, len(X_train_img))):
                yield [X_train_img[i:i+1].astype(np.float32)]
                
        converter.representative_dataset = representative_dataset
    
    tflite_model = converter.convert()
    return tflite_model

# Create standard TFLite model
tflite_model = convert_to_tflite(model, quantize=False)
with open('emotion_model_binary.tflite', 'wb') as f:
    f.write(tflite_model)

# Create quantized TFLite model
try:
    tflite_model_quant = convert_to_tflite(model, quantize=True)
    with open('emotion_model_binary_quantized.tflite', 'wb') as f:
        f.write(tflite_model_quant)
    print("Successfully created quantized TFLite model")
except Exception as e:
    print(f"Error creating quantized model: {e}")
    print("Skipping quantized model creation")

# Compare model sizes
import os
h5_size = os.path.getsize('emotion_model_binary_final.h5') / (1024 * 1024)
tflite_size = os.path.getsize('emotion_model_binary.tflite') / (1024 * 1024)
try:
    tflite_quant_size = os.path.getsize('emotion_model_binary_quantized.tflite') / (1024 * 1024)
    print(f"Model size comparison:\n  Original: {h5_size:.2f} MB\n  TFLite: {tflite_size:.2f} MB\n  TFLite Quantized: {tflite_quant_size:.2f} MB")
except:
    print(f"Model size comparison:\n  Original: {h5_size:.2f} MB\n  TFLite: {tflite_size:.2f} MB")

print("Training complete. Binary emotion models saved.")