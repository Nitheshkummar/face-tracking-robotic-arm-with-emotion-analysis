import time
import sys
import os

# Add the current directory to the path to find the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the required modules
from face_tracking_kalman import FaceTracker, EmotionDetector
from robot_arm_control import RobotArmController

def main():
    # Create emotion detector for face tracking
    emotion_detector = EmotionDetector()
    
    # Create arm controller
    print("Initializing robot arm controller...")
    arm = RobotArmController()
    
    if not arm.connect():
        print("Failed to connect to CoppeliaSim. Exiting.")
        return
    
    if not arm.get_joint_handles():
        print("Failed to get joint handles. Exiting.")
        return
    
    # Set initial position
    print("Setting initial arm position...")
    arm.set_initial_position()
    
    # Start the arm movement thread for smooth operation
    arm.start_movement_thread()
    
    # Define callback function for face tracking
    def face_position_callback(x, y):
        # Move arm to follow the face
        arm.move_to_position(x, y)
    
    # Create and start face tracker
    print("Starting face tracker...")
    tracker = FaceTracker(
        emotion_detector=emotion_detector,
        callback=face_position_callback, 
        display=True
    )
    tracker.start()
    
    try:
        print("System running. Press Ctrl+C to stop.")
        # Keep main thread alive while tracker is running
        while tracker.thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        # Clean up resources
        tracker.stop()
        arm.cleanup()
        print("System stopped.")

if __name__ == "__main__":
    main()