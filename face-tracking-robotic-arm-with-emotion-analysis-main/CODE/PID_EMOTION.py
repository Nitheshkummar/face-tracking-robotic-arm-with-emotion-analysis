import cv2
import numpy as np
import tensorflow as tf
import pickle
import time
from collections import deque
from adafruit_pca9685 import PCA9685
from board import SCL, SDA
import busio

# ======================= Init Shared Components =======================
i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c)
pca.frequency = 50

# Load a smaller, faster face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ======================= Load TFLite Model =======================
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# ======================= Load PCA Model =======================
def load_pca_model(pca_path):
    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)
    return pca

# ======================= Optimized Image Preprocessing =======================
def preprocess_image(img, pca_model, input_details):
    # Get image dimensions for ROI processing
    h, w = img.shape[:2]
    
    # Process only central region to improve speed
    roi_size = min(h, w)
    start_x = (w - roi_size) // 2
    start_y = (h - roi_size) // 2
    roi = img[start_y:start_y+roi_size, start_x:start_x+roi_size]
    
    # Resize directly to target size (much faster)
    resized = cv2.resize(roi, (48, 48))
    
    # Convert to grayscale after resize (faster)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) > 2 else resized
    
    # Normalize
    normalized = gray.astype(np.float32) / 255.0
    
    # Fast PCA transformation
    flattened = normalized.reshape(1, -1)
    pca_features = pca_model.transform(flattened)
    reconstructed = pca_model.inverse_transform(pca_features)
    
    # Reshape for model input (48x48x1)
    reconstructed_img = reconstructed.reshape(1, 48, 48, 1)
    
    # Prepare based on input type
    if input_details[0]['dtype'] == np.uint8:
        input_scale, input_zero_point = input_details[0]['quantization']
        reconstructed_img = reconstructed_img / input_scale + input_zero_point
        reconstructed_img = np.clip(reconstructed_img, 0, 255).astype(np.uint8)
    else:
        reconstructed_img = reconstructed_img.astype(np.float32)
    
    return reconstructed_img

# ======================= Simplified Confidence Computation =======================
def compute_confidence(probabilities, output_details=None):
    # If model output is quantized, dequantize it
    if output_details and output_details[0]['dtype'] == np.uint8:
        output_scale, output_zero_point = output_details[0]['quantization']
        probabilities = (probabilities.astype(np.float32) - output_zero_point) * output_scale
    
    # Simple max probability as confidence (faster)
    return np.max(probabilities)

# ======================= Run Inference =======================
def run_inference(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    if output_details[0]['dtype'] == np.uint8:
        output_scale, output_zero_point = output_details[0]['quantization']
        output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
    
    return output_data, output_details, inference_time

# ======================= Simplified Emotion Stabilizer =======================
class EmotionStabilizer:
    def _init_(self, history_size=10, stability_threshold=0.65, smoothing_factor=0.3):
        self.history_size = history_size
        self.stability_threshold = stability_threshold
        self.smoothing_factor = smoothing_factor
        self.emotion_history = deque(maxlen=history_size)
        self.confidence_history = deque(maxlen=history_size)
        self.current_emotion = None
        self.current_confidence = 0.0
        self.smoothed_confidence = 0.0
        self.frames_in_current_emotion = 0
        
    def update(self, emotion_idx, confidence, probabilities):
        # Add new data to history
        self.emotion_history.append(emotion_idx)
        self.confidence_history.append(confidence)
        
        # Update current values
        self.current_emotion = emotion_idx
        self.current_confidence = confidence
        
        # Apply exponential smoothing to confidence
        if len(self.confidence_history) == 1:
            self.smoothed_confidence = confidence
        else:
            self.smoothed_confidence = (self.smoothing_factor * confidence + 
                                       (1 - self.smoothing_factor) * self.smoothed_confidence)
        
        # Process only if we have enough history
        if len(self.emotion_history) >= 3:
            # Simple counter-based approach (faster than weighted probabilities)
            emotion_counts = {}
            for emotion in self.emotion_history:
                if emotion in emotion_counts:
                    emotion_counts[emotion] += 1
                else:
                    emotion_counts[emotion] = 1
            
            most_common_emotion = max(emotion_counts, key=emotion_counts.get)
            stability_score = emotion_counts[most_common_emotion] / len(self.emotion_history)
            
            if stability_score > self.stability_threshold:
                if most_common_emotion != self.current_emotion:
                    self.current_emotion = most_common_emotion
                    self.frames_in_current_emotion = 1
                else:
                    self.frames_in_current_emotion += 1
        
        stability_factor = min(1.0, self.frames_in_current_emotion / 10)
        return self.current_emotion, self.smoothed_confidence, stability_factor

# ======================= Simple Kalman Filter =======================
class KalmanFilter:
    def _init_(self, dt=0.1):
        # Simplified 2D position-only Kalman filter
        self.dt = dt
        self.X = np.matrix([[0], [0]])
        self.P = np.eye(2) * 0.5
        self.Q = np.eye(2) * 0.01
        self.R = np.eye(2) * 0.1
        self.A = np.eye(2)
        self.H = np.eye(2)

    def update(self, Z):
        # Prediction step
        self.P = self.P + self.Q
        
        # Update step
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)
        self.X = self.X + K @ (Z - self.X)
        self.P = (np.eye(2) - K) @ self.P
        
        return self.X

# ======================= PID Controller =======================
class PID:
    def _init_(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, target, current, dt=0.1):
        error = target - current

        # Anti-windup for integral term
        if abs(error) < 15:  # Only accumulate integral when close to target
            self.integral += error * dt
            # Limit integral to prevent excessive buildup
            self.integral = max(-20, min(20, self.integral))
        else:
            self.integral = 0
            
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        
        return self.kp * error + self.ki * self.integral + self.kd * derivative

# ======================= Servo Joint Class =======================
class ServoJoint:
    def _init_(self, channel, home, min_angle, max_angle, pid, alpha=0.5):
        self.channel = channel
        self.angle = home
        self.home = home
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.pid = pid
        self.alpha = alpha
        self.last_output = 0
        self.idle_counter = 0
        self.idle_threshold = 3
        self.is_idle = False
        self.set_angle(self.angle)

    def set_angle(self, angle):
        angle = max(self.min_angle, min(self.max_angle, angle))
        pwm_val = int(150 + (angle / 180.0) * 450)
        pca.channels[self.channel].duty_cycle = int(pwm_val * 65535 / 4096)

    def update(self, offset, active=True, return_speed=0.2):
        if active:
            # Only calculate PID if offset is significant
            if abs(offset) > 3:
                angle_change = self.pid.compute(0, offset)
                angle_change *= return_speed
                angle_change = (1 - self.alpha) * self.last_output + self.alpha * angle_change
                self.last_output = angle_change
                
                if abs(angle_change) < 0.5:
                    self.idle_counter += 1
                else:
                    self.idle_counter = 0
                    self.is_idle = False
                    
                if self.idle_counter >= self.idle_threshold:
                    self.is_idle = True
                else:
                    self.angle += angle_change
                    self.angle = max(self.min_angle, min(self.max_angle, self.angle))
                    self.set_angle(self.angle)
            else:
                self.idle_counter += 1
                if self.idle_counter >= self.idle_threshold:
                    self.is_idle = True
        
        elif not active and not self.is_idle:
            # Only return to home if not at home
            if abs(self.angle - self.home) > 2:
                angle_change = self.pid.compute(self.home, self.angle)
                angle_change *= return_speed * 2  # return faster
                self.angle += angle_change
                self.angle = max(self.min_angle, min(self.max_angle, self.angle))
                self.set_angle(self.angle)
            else:
                self.is_idle = True
                
# ======================= Main Function =======================
def main():
    # Load TFLite and PCA models
    print("Loading models...")
    try:
        tflite_model_path = 'emotion_model_binary_quantized.tflite'
        interpreter = load_tflite_model(tflite_model_path)
        print("Loaded quantized TFLite model")
    except Exception as e:
        print(f"Error loading quantized model: {e}")
        try:
            tflite_model_path = 'emotion_model_binary.tflite'
            interpreter = load_tflite_model(tflite_model_path)
            print("Loaded standard TFLite model")
        except Exception as e:
            print(f"Error loading standard model: {e}")
            return
    
    try:
        pca_model = load_pca_model('emotion_pca_model_binary.pkl')
        print("Loaded PCA model")
    except Exception as e:
        print(f"Error loading PCA model: {e}")
        return
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    emotion_labels = ['Positive', 'Negative']
    
    # Create emotion stabilizer (reduced history size)
    stabilizer = EmotionStabilizer(history_size=10, stability_threshold=0.6, smoothing_factor=0.3)
    
    # Initialize simplified robot arm components
    kalman = KalmanFilter()
    base_pid = PID(0.06, 0.0005, 0.01)
    palm_pid = PID(0.06, 0.0005, 0.01)
    shoulder_pid = PID(0.06, 0.0005, 0.01)

    base = ServoJoint(channel=3, home=90, min_angle=60, max_angle=120, pid=base_pid)
    palm = ServoJoint(channel=1, home=20, min_angle=25, max_angle=40, pid=palm_pid)
    shoulder = ServoJoint(channel=2, home=60, min_angle=30, max_angle=70, pid=shoulder_pid)
    target_width = 120  # for shoulder Z-depth reference
    
    print("Starting video capture...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video capture device")
        return
    
    # Lower resolution for performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    
    # Performance metrics
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    # Face tracking variables
    last_face = None
    face_smoothing_factor = 0.3
    
    # Skip frame counter for face detection (only run detection every N frames)
    detection_interval = 2
    frame_skip_counter = 0
    
    # Skip frame counter for emotion detection (only run emotion every N frames)
    emotion_interval = 3
    emotion_skip_counter = 0
    
    # Last emotion data
    last_emotion = 0
    last_confidence = 0.5
    
    print("[INFO] Optimized arm tracking with emotion detection started")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
 
        # FPS calculation
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # Only process every Nth frame for face detection (reduces CPU load)
        process_faces = (frame_skip_counter % detection_interval == 0)
        frame_skip_counter += 1
        
        # Create a copy of the frame for drawing (only if we'll show it)
        display_frame = frame.copy()
        
        found_face = False
        if process_faces:
            # Convert to grayscale and downscale for faster face detection
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            # Use more aggressive scaling factor for speed
            faces = face_cascade.detectMultiScale(gray, 1.2, 3, minSize=(30, 30))
            
            if len(faces) > 0:
                found_face = True
                # Convert coordinates back to original scale
                faces = [(int(x*2), int(y*2), int(w*2), int(h*2)) for x, y, w, h in faces]
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                
                # Smooth face location
                if last_face is not None:
                    smoothed_face = (
                        int(face_smoothing_factor * largest_face[0] + (1 - face_smoothing_factor) * last_face[0]),
                        int(face_smoothing_factor * largest_face[1] + (1 - face_smoothing_factor) * last_face[1]),
                        int(face_smoothing_factor * largest_face[2] + (1 - face_smoothing_factor) * last_face[2]),
                        int(face_smoothing_factor * largest_face[3] + (1 - face_smoothing_factor) * last_face[3])
                    )
                    last_face = smoothed_face
                else:
                    smoothed_face = largest_face
                    last_face = largest_face
                
                x, y, w, h = smoothed_face
                
                # Extract face region (with smaller margin to reduce processing)
                y_start = max(0, y-10)
                y_end = min(frame.shape[0], y+h+10)
                x_start = max(0, x-10)
                x_end = min(frame.shape[1], x+w+10)
                
                face_roi = frame[y_start:y_end, x_start:x_end]
                
                # Draw rectangle around face
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw point at nose position
                cx, cy = x + w//2, y + h//2
                nose_y = y + h//3
                cv2.circle(display_frame, (cx, nose_y), 3, (0, 0, 255), -1)
                
                # Process emotion only on some frames
                process_emotion = (emotion_skip_counter % emotion_interval == 0) and not face_roi.size == 0
                emotion_skip_counter += 1
                
                if process_emotion:
                    try:
                        # Preprocess image for emotion detection
                        input_data = preprocess_image(face_roi, pca_model, input_details)
                        
                        # Run inference
                        output_data, out_details, _ = run_inference(interpreter, input_data)
                        
                        # Get emotion prediction
                        emotion_probs = output_data[0]
                        raw_emotion_idx = np.argmax(emotion_probs)
                        
                        # Calculate confidence score
                        raw_confidence = compute_confidence(emotion_probs, out_details)
                        
                        # Stabilize emotion and confidence
                        stable_emotion_idx, stable_confidence, stability_factor = stabilizer.update(
                            raw_emotion_idx, raw_confidence, emotion_probs)
                        
                        # Update last emotion data
                        last_emotion = stable_emotion_idx
                        last_confidence = stable_confidence
                        
                    except Exception as e:
                        pass
                
                # Display emotion from last processed frame
                emotion_label = emotion_labels[last_emotion]
                confidence_pct = int(last_confidence * 100)
                
                # Use simple coloring based on just confidence
                if last_confidence > 0.7:
                    text_color = (0, 255, 0)  # Green
                elif last_confidence > 0.5:
                    text_color = (0, 165, 255)  # Orange
                else:
                    text_color = (0, 0, 255)  # Red
                    
                # Put text above face
                cv2.putText(display_frame, f"{emotion_label}: {confidence_pct}%", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                
                # Update robot arm tracking with simplified Kalman filter
                Z = np.matrix([[cx], [cy]])
                smoothed = kalman.update(Z)

                frame_center_x = frame.shape[1] // 2
                frame_center_y = frame.shape[0] // 2
                offset_x = smoothed[0, 0] - frame_center_x  # base (horizontal)
                offset_y = smoothed[1, 0] - frame_center_y  # elbow (vertical)
                offset_z = w - target_width                 # shoulder (depth)

                # Only update servos if offsets exceed threshold
                if abs(offset_x) > 5:
                    base.update(offset_x)
                if abs(offset_y) > 5:
                    palm.update(offset_y)
                if abs(offset_z) > 5:
                    shoulder.update(offset_z)
        
        # If no face found this frame but we had one before
        if not found_face and process_faces:
            # Only reset tracking if we've gone several frames without a face
            if last_face is not None:
                last_face = None
                base.update(0, active=False)
                palm.update(0, active=False)
                shoulder.update(0, active=False)
        
        # Display FPS
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show servo angles
        cv2.putText(display_frame, f"Base: {int(base.angle)}  Palm: {int(palm.angle)}  Shoulder: {int(shoulder.angle)}", 
                    (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Show the frame
        cv2.imshow('Robot Arm Tracker (Optimized)', display_frame)
        
        # Check for key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed")

if _name_ == "_main_":
    main()