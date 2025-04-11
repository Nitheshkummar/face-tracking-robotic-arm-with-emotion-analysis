import cv2
import numpy as np
import tensorflow as tf
import pickle
import threading
import time
import math
from scipy.stats import entropy
from collections import deque

class KalmanFilter:
    def __init__(self, dt=0.1, u_x=0.5, u_y=0.5, std_acc=0.8, x_std_meas=0.1, y_std_meas=0.1):
        # Control input variables
        self.U = np.matrix([[u_x], [u_y]])
        
        # Initial State
        self.X = np.matrix([[0], [0], [0], [0]])
        
        # State Transition Matrix
        self.A = np.matrix([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        
        # Control Input Matrix
        self.B = np.matrix([[(dt**2)/2, 0],
                           [0, (dt**2)/2],
                           [dt, 0],
                           [0, dt]])
        
        # Measurement Mapping Matrix
        self.H = np.matrix([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        
        # Process Noise Covariance
        self.Q = np.matrix([[(dt**4)/4, 0, (dt**3)/2, 0],
                           [0, (dt**4)/4, 0, (dt**3)/2],
                           [(dt**3)/2, 0, dt**2, 0],
                           [0, (dt**3)/2, 0, dt**2]]) * std_acc**2
        
        # Measurement Noise Covariance
        self.R = np.matrix([[x_std_meas**2, 0],
                           [0, y_std_meas**2]])
        
        # Covariance Matrix
        self.P = np.eye(self.A.shape[1])
    
    def predict(self):
        # Predicted State Estimate
        self.X = self.A * self.X + self.B * self.U
        # Predicted Estimate Covariance
        self.P = self.A * self.P * self.A.T + self.Q
        return self.X, self.P
    
    def update(self, Z):
        # Measurement Residual
        Y = Z - (self.H * self.X)
        # Residual Covariance
        S = self.H * self.P * self.H.T + self.R
        # Kalman Gain
        K = self.P * self.H.T * np.linalg.inv(S)
        # Updated State Estimate
        self.X = self.X + K * Y
        # Updated Estimate Covariance
        self.P = (np.eye(self.P.shape[0]) - K * self.H) * self.P
        return self.X, self.P

class EmotionStabilizer:
    def __init__(self, history_size=15, stability_threshold=0.65, smoothing_factor=0.3):
        self.history_size = history_size
        self.stability_threshold = stability_threshold
        self.smoothing_factor = smoothing_factor
        
        # Initialize tracking variables
        self.emotion_history = deque(maxlen=history_size)
        self.confidence_history = deque(maxlen=history_size)
        self.current_emotion = None
        self.current_confidence = 0.0
        self.smoothed_confidence = 0.0
        self.stable_since = 0
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
        
        # Check if we have enough history
        if len(self.emotion_history) >= 5:
            # Calculate emotion frequency
            emotions = list(self.emotion_history)
            emotion_count = {}
            for emotion in emotions:
                if emotion in emotion_count:
                    emotion_count[emotion] += 1
                else:
                    emotion_count[emotion] = 1
            
            # Calculate weighted probabilities (recent frames count more)
            weighted_emotions = {}
            for i, emotion in enumerate(emotions):
                weight = (i + 1) / sum(range(1, len(emotions) + 1))  # Higher weights for more recent frames
                if emotion in weighted_emotions:
                    weighted_emotions[emotion] += weight
                else:
                    weighted_emotions[emotion] = weight
            
            # Find the most stable emotion
            most_stable_emotion = max(weighted_emotions, key=weighted_emotions.get)
            stability_score = weighted_emotions[most_stable_emotion]
            
            # Check if stability threshold is met
            if stability_score > self.stability_threshold:
                # If the stable emotion is different from current, change it
                if most_stable_emotion != self.current_emotion:
                    self.current_emotion = most_stable_emotion
                    self.stable_since = time.time()
                    self.frames_in_current_emotion = 1
                else:
                    self.frames_in_current_emotion += 1
        
        # Calculate weighted confidence based on stability
        stability_factor = min(1.0, self.frames_in_current_emotion / 10)
        return self.current_emotion, self.smoothed_confidence, stability_factor

# Load the TFLite model
def load_tflite_model(model_path):
    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Load the PCA model
def load_pca_model(pca_path):
    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)
    return pca

# Function to preprocess image for model input
def preprocess_image(img, pca_model, input_details):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize
    resized = cv2.resize(gray, (48, 48))
    
    # Normalize
    normalized = resized / 255.0
    
    # Flatten for PCA
    flattened = normalized.reshape(1, -1)
    
    # Apply PCA transformation
    pca_features = pca_model.transform(flattened)
    
    # Reconstruct from PCA
    reconstructed = pca_model.inverse_transform(pca_features)
    
    # Reshape for model input (48x48x1)
    reconstructed_img = reconstructed.reshape(1, 48, 48, 1)
    
    # Check if the model expects UINT8 input
    if input_details[0]['dtype'] == np.uint8:
        # Quantize the input from float to uint8
        input_scale, input_zero_point = input_details[0]['quantization']
        reconstructed_img = reconstructed_img / input_scale + input_zero_point
        reconstructed_img = np.clip(reconstructed_img, 0, 255).astype(np.uint8)
    else:
        reconstructed_img = reconstructed_img.astype(np.float32)
    
    return reconstructed_img

# Function to compute confidence score based on probability theory
def compute_confidence(probabilities, output_details=None):
    """
    Compute confidence score based on probability distribution.
    
    Mathematical basis:
    1. Uses maximum probability directly (P(class))
    2. Incorporates Shannon entropy calculation to measure uncertainty
    
    The formula balances these for a robust confidence score:
    confidence = w1 * max_prob + w2 * (1 - normalized_entropy)
    """
    # If model output is quantized, dequantize it
    if output_details and output_details[0]['dtype'] == np.uint8:
        output_scale, output_zero_point = output_details[0]['quantization']
        probabilities = (probabilities.astype(np.float32) - output_zero_point) * output_scale
    
    # Ensure probabilities sum to 1 and are non-negative
    probabilities = np.clip(probabilities, 1e-10, 1.0)
    probabilities = probabilities / np.sum(probabilities)
    
    max_prob = np.max(probabilities)
    
    # For binary classification, analytical approach to entropy:
    # Using the binary entropy function H(p) = -p*log(p) - (1-p)*log(1-p)
    p = probabilities[0]  # Probability of positive class
    
    # Avoid log(0) issues
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1-epsilon)
    
    # Calculate normalized binary entropy (0 = certain, 1 = maximum uncertainty)
    binary_entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)
    norm_entropy = binary_entropy / 1.0  # Binary entropy max is 1.0
    
    # Convert to confidence measure (higher = more confident)
    entropy_confidence = 1 - norm_entropy
    
    # Weighted combination (70% max probability, 30% entropy-based confidence)
    confidence = 0.7 * max_prob + 0.3 * entropy_confidence
    
    return confidence

# Function to run inference
def run_inference(interpreter, input_data):
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Dequantize output if it's quantized
    if output_details[0]['dtype'] == np.uint8:
        output_scale, output_zero_point = output_details[0]['quantization']
        output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
    
    return output_data, output_details

class FaceTracker:
    def __init__(self, emotion_detector=None, callback=None, display=True):
        # Initialize Kalman Filter for smooth tracking
        self.kf = KalmanFilter()
        
        # Create Haar Cascade Classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize variables
        self.running = False
        self.display = display
        self.callback = callback
        self.last_position = None
        self.frame_width = 640
        self.frame_height = 480
        
        # Performance improvements
        self.skip_frames = 1  # Process every other frame
        self.frame_count = 0
        
        # Emotion detection components
        self.emotion_detector = emotion_detector
        
        # Initialize threading resources
        self.thread = None
        self.lock = threading.Lock()
        
        # Face tracking to reduce jitter
        self.last_face = None
        self.face_smoothing_factor = 0.3
        
        # Performance metrics
        self.fps = 0
    
    def start(self):
        """Start tracking in a separate thread"""
        self.running = True
        self.thread = threading.Thread(target=self._tracking_loop)
        self.thread.daemon = True
        self.thread.start()
        return self.thread
    
    def stop(self):
        """Stop the tracking thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
    def get_current_position(self):
        """Get the latest estimated face position"""
        with self.lock:
            return self.last_position
    
    def _tracking_loop(self):
        """Main tracking loop running in separate thread"""
        # Open webcam
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        # Time tracking for FPS calculation
        frame_count = 0
        start_time = time.time()
        
        while self.running:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Update FPS calculation
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1:
                self.fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Create a copy of the frame for drawing
            display_frame = frame.copy() if self.display else None
            
            # Performance optimization: Resize frame for faster detection
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with optimized parameters
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.2,  # Faster detection with slightly reduced accuracy
                minNeighbors=3,   # Fewer required neighbors for speed
                minSize=(30, 30)  # Minimum face size (adjusted for smaller frame)
            )
            
            # Process detected face(s)
            if len(faces) > 0:
                # For simplicity, just process the largest face
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                # Scale coordinates back to original frame size
                largest_face = largest_face * 2  # Multiply by 2 because we resized by 0.5
                
                # Smooth face location to reduce jitter
                if self.last_face is not None:
                    smoothed_face = (
                        int(self.face_smoothing_factor * largest_face[0] + (1 - self.face_smoothing_factor) * self.last_face[0]),
                        int(self.face_smoothing_factor * largest_face[1] + (1 - self.face_smoothing_factor) * self.last_face[1]),
                        int(self.face_smoothing_factor * largest_face[2] + (1 - self.face_smoothing_factor) * self.last_face[2]),
                        int(self.face_smoothing_factor * largest_face[3] + (1 - self.face_smoothing_factor) * self.last_face[3])
                    )
                    self.last_face = smoothed_face
                else:
                    smoothed_face = largest_face
                    self.last_face = largest_face
                
                x, y, w, h = smoothed_face
                
                # Get face center for Kalman tracking
                centerx, centery = (x + w / 2, y + h / 2)
                
                # Kalman Filter Prediction
                predicted_state, _ = self.kf.predict()
                
                # Kalman Filter Update
                Z = np.matrix([[centerx], [centery]])
                estimated_state, _ = self.kf.update(Z)
                
                # Get the estimated position
                est_x, est_y = int(estimated_state[0]), int(estimated_state[1])
                
                # Normalize position to [-1, 1] range
                norm_x = (est_x / self.frame_width) * 2 - 1
                norm_y = 1 - (est_y / self.frame_height)
                
                # Save the position
                with self.lock:
                    self.last_position = (norm_x, norm_y)
                
                # Call the callback function if provided
                if self.callback:
                    self.callback(norm_x, norm_y)
                
                # Extract face region with some margin for emotion detection
                y_start = max(0, y-20)
                y_end = min(frame.shape[0], y+h+20)
                x_start = max(0, x-20)
                x_end = min(frame.shape[1], x+w+20)
                
                face_roi = frame[y_start:y_end, x_start:x_end]
                
                # Process emotion if face ROI is valid and emotion detector is available
                emotion_result = None
                if face_roi.size > 0 and self.emotion_detector:
                    emotion_result = self.emotion_detector.process_face(face_roi)
                
                # Draw visualization if display is enabled
                if self.display:
                    # Draw rectangle around face
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 191, 255), 2)
                    
                    # Draw Kalman tracking information
                    # Draw measured position
                    rectcenter = int(centerx), int(centery)
                    cv2.circle(display_frame, rectcenter, 5, (0, 191, 255), 2)
                    
                    # Draw predicted position
                    pred_x, pred_y = int(predicted_state[0]), int(predicted_state[1])
                    cv2.circle(display_frame, (pred_x, pred_y), 5, (0, 255, 0), 2)
                    
                    # Draw estimated position
                    cv2.circle(display_frame, (est_x, est_y), 5, (0, 0, 255), 2)
                    
                    # Add position labels
                    cv2.putText(display_frame, "Est", (est_x + 10, est_y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    cv2.putText(display_frame, "Pred", (pred_x + 10, pred_y + 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    # Display emotion results if available
                    if emotion_result:
                        emotion_label, confidence, stability = emotion_result
                        # Set text color based on confidence and stability
                        if confidence > 0.7 and stability > 0.8:
                            text_color = (0, 255, 0)  # Bright green for high confidence and stability
                        elif confidence > 0.5:
                            text_color = (0, 165, 255)  # Orange for medium confidence
                        else:
                            text_color = (0, 0, 255)  # Red for low confidence
                        
                        # Put emotion text above the face rectangle
                        text = f"{emotion_label}: {int(confidence * 100)}%"
                        cv2.putText(display_frame, text, (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
            else:
                # If no face detected, reset face tracking
                self.last_face = None
            
            if self.display:
                # Display FPS
                cv2.putText(display_frame, f"FPS: {self.fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show the frame
                cv2.imshow('Face Tracking with Emotion Detection', display_frame)
                
                # Check for key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Release resources
        cap.release()
        if self.display:
            cv2.destroyAllWindows()

class EmotionDetector:
    def __init__(self, model_path=None, pca_path=None):
        self.model_path = model_path if model_path else 'emotion_model_binary_quantized.tflite'
        self.pca_path = pca_path if pca_path else 'emotion_pca_model_binary.pkl'
        self.emotion_labels = ['Positive', 'Negative']
        self.stabilizer = EmotionStabilizer(history_size=15, stability_threshold=0.65, smoothing_factor=0.3)
        
        # Load models
        self.load_models()
        
    def load_models(self):
        print("Loading emotion detection models...")
        try:
            # Try to load the quantized model first
            self.interpreter = load_tflite_model(self.model_path)
            print(f"Loaded TFLite model from {self.model_path}")
            
            self.pca_model = load_pca_model(self.pca_path)
            print(f"Loaded PCA model from {self.pca_path}")
            
            # Get model details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"Model input type: {self.input_details[0]['dtype']}")
            print(f"Model output type: {self.output_details[0]['dtype']}")
            print(f"Model input shape: {self.input_details[0]['shape']}")
            
            self.is_ready = True
        except Exception as e:
            print(f"Error loading emotion detection models: {e}")
            self.is_ready = False
    
    def process_face(self, face_roi):
        """Process face ROI and return emotion detection results"""
        if not self.is_ready or face_roi.size == 0:
            return None
        
        try:
            # Preprocess image
            input_data = preprocess_image(face_roi, self.pca_model, self.input_details)
            
            # Run inference
            output_data, out_details = run_inference(self.interpreter, input_data)
            
            # Get emotion prediction
            emotion_probs = output_data[0]
            raw_emotion_idx = np.argmax(emotion_probs)
            
            # Calculate confidence score using probability theory
            raw_confidence = compute_confidence(emotion_probs, out_details)
            
            # Stabilize emotion and confidence
            stable_emotion_idx, stable_confidence, stability_factor = self.stabilizer.update(
                raw_emotion_idx, raw_confidence, emotion_probs)
            
            # Get emotion label
            emotion_label = self.emotion_labels[stable_emotion_idx]
            
            return (emotion_label, stable_confidence, stability_factor)
            
        except Exception as e:
            print(f"Error processing face for emotion: {e}")
            return None

def main():
    print("Initializing Face Tracking with Emotion Detection...")
    
    # Initialize emotion detector
    emotion_detector = EmotionDetector()
    
    # Optional: Define callback for face position (for external applications)
    def face_position_callback(x, y):
        # This could be used to control external systems based on face position
        pass
    
    # Create face tracker with emotion detector
    tracker = FaceTracker(
        emotion_detector=emotion_detector,
        callback=face_position_callback,
        display=True
    )
    
    # Start tracking
    tracker.start()
    
    print("Press 'q' in the video window to quit")
    
    try:
        # Keep main thread alive
        while tracker.thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping application...")
    finally:
        # Clean up
        tracker.stop()
        print("Application closed")

if __name__ == "__main__":
    main()