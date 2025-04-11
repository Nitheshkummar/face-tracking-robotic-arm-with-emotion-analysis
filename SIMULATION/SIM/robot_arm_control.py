import sim
import time
import math
import threading
import numpy as np

class RobotArmController:
    def __init__(self):
        # Robot arm parameters
        self.L1 = 0.5  # Length of link 1 (in meters)
        self.L2 = 0.5  # Length of link 2 (in meters)
        self.L3 = 0.5  # Length of link 3 (in meters)
        
        # CoppeliaSim connection
        self.client_id = -1
        self.joint_handles = []
        
        # Current arm state
        self.current_angles = [0, 0, 0, 0, 0]
        
        # Movement parameters
        self.target_angles = [0, 0, 0, 0, 0]
        self.movement_step = 0.05
        self.movement_pause = 0.02  # Time between movement steps
        
        # Threading resources
        self.running = False
        self.move_thread = None
        self.lock = threading.Lock()
    
    def connect(self):
        """Connect to CoppeliaSim simulator"""
        sim.simxFinish(-1)  # Close any existing connections
        self.client_id = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
        if self.client_id != -1:
            print("Connected to CoppeliaSim")
            return True
        else:
            print("Failed to connect to CoppeliaSim")
            return False
    
    def get_joint_handles(self):
        """Get handles for all robot joints"""
        self.joint_handles = []
        for i in range(1, 6):
            return_code, handle = sim.simxGetObjectHandle(
                self.client_id, f'joint_{i}', sim.simx_opmode_blocking
            )
            if return_code != sim.simx_return_ok:
                print(f"[ERROR] Failed to get handle for joint_{i}")
                return False
            else:
                print(f"[SUCCESS] Handle for joint_{i} retrieved: {handle}")
            self.joint_handles.append(handle)
        return True
    
    def set_initial_position(self):
        """Set the robotic arm to a straight initial position"""
        initial_angles = [0, 0, 0, 0, 0]  # Set all joints to zero
        return self.move_to_angles(initial_angles, wait=True)
    
    def forward_kinematics_3dof(self, theta1, theta2, theta3):
        """
        Calculate the end effector position from 3 joint angles
        Returns: (x, y, z) position
        """
        # Calculate the position after the first joint
        x1 = self.L1 * math.cos(theta1)
        y1 = self.L1 * math.sin(theta1)
        z1 = 0
        
        # Calculate the position after the second joint
        x2 = x1 + self.L2 * math.cos(theta1 + theta2)
        y2 = y1 + self.L2 * math.sin(theta1 + theta2)
        z2 = 0
        
        # Calculate the final position including the third joint
        # The third joint affects the z-coordinate
        x3 = x2
        y3 = y2
        z3 = self.L3 * math.sin(theta3)  # Third joint contributes to z-axis movement
        
        return x3, y3, z3
    
    def forward_kinematics(self, theta1, theta2):
        """
        Calculate the end effector position from first 2 joint angles
        Returns: (x, y) position
        """
        x = self.L1 * math.cos(theta1) + self.L2 * math.cos(theta1 + theta2)
        y = self.L1 * math.sin(theta1) + self.L2 * math.sin(theta1 + theta2)
        return x, y
    
    def jacobian_3dof(self, theta1, theta2, theta3):
        """
        Calculate the Jacobian matrix for the 3-link arm
        Returns: 3x3 Jacobian matrix
        """
        # Partial derivatives for x,y,z with respect to theta1, theta2, theta3
        j11 = -self.L1 * math.sin(theta1) - self.L2 * math.sin(theta1 + theta2)
        j12 = -self.L2 * math.sin(theta1 + theta2)
        j13 = 0  # x position doesn't depend on theta3
        
        j21 = self.L1 * math.cos(theta1) + self.L2 * math.cos(theta1 + theta2)
        j22 = self.L2 * math.cos(theta1 + theta2)
        j23 = 0  # y position doesn't depend on theta3
        
        j31 = 0  # z position doesn't depend on theta1
        j32 = 0  # z position doesn't depend on theta2
        j33 = self.L3 * math.cos(theta3)  # z depends on theta3
        
        # Return Jacobian matrix
        return np.array([[j11, j12, j13], [j21, j22, j23], [j31, j32, j33]])
    
    
    
    def inverse_kinematics_3dof(self, target_x, target_y, target_z=0, max_iterations=100, tolerance=0.001):
        """
        Solve inverse kinematics for 3-DOF arm using Jacobian method
        Returns angles for all three joints (theta1, theta2, theta3)
        """
        # Start with current angles as initial guess
        theta1, theta2, theta3 = self.current_angles[0], self.current_angles[1], self.current_angles[2]
        
        # Check if target is reachable in xy plane
        max_reach_xy = self.L1 + self.L2
        r_xy = math.sqrt(target_x**2 + target_y**2)
        if r_xy > max_reach_xy * 0.99:
            # Target is too far in xy plane, scale it down to maximum reach
            scale = max_reach_xy * 0.99 / r_xy
            target_x *= scale
            target_y *= scale
        
        # Check if target_z is reachable
        max_reach_z = self.L3
        if abs(target_z) > max_reach_z * 0.99:
            # Target is too high/low, scale it down
            target_z = math.copysign(max_reach_z * 0.99, target_z)
        
        # Iterative refinement for all three joints
        for i in range(max_iterations):
            # Calculate current position with all three links
            current_x, current_y, current_z = self.forward_kinematics_3dof(theta1, theta2, theta3)
            
            # Calculate error
            dx = target_x - current_x
            dy = target_y - current_y
            dz = target_z - current_z
            error = math.sqrt(dx**2 + dy**2 + dz**2)
            
            # Check if we've converged
            if error < tolerance:
                break
            
            # Calculate full 3DOF Jacobian
            J = self.jacobian_3dof(theta1, theta2, theta3)
            
            # Handle singular matrices with pseudoinverse
            J_inv = np.linalg.pinv(J)
            
            # Calculate joint angle adjustments
            dtheta = np.matmul(J_inv, np.array([[dx], [dy], [dz]]))
            
            # Update angles (with damping to prevent large steps)
            damping = 0.5
            theta1 += damping * dtheta[0, 0]
            theta2 += damping * dtheta[1, 0]
            theta3 += damping * dtheta[2, 0]
            
            # Apply joint limits
            theta1 = max(min(theta1, math.pi), -math.pi)
            theta2 = max(min(theta2, math.pi), -math.pi)
            theta3 = max(min(theta3, math.pi * 0.8), -math.pi * 0.8)
        
        # Print debugging info
        print(f"Target: ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})")
        print(f"Calculated angles: theta1={theta1:.2f}, theta2={theta2:.2f}, theta3={theta3:.2f}")
        
        return theta1, theta2, theta3

    
    def move_to_angles(self, target_angles, wait=False):
        """
        Move joints to specified angles
        target_angles: List of 5 angles in radians
        wait: If True, waits for movement to complete
        """
        # Print target angles for debugging
        print(f"Moving joints to angles: {[round(angle, 2) for angle in target_angles]}")
        
        # Update target angles thread-safely
        with self.lock:
            self.target_angles = target_angles.copy()
        
        if wait:
            # If waiting, perform movement in this thread
            self._execute_movement()
            return True
        return True
    
    def start_movement_thread(self):
        """Start continuous movement thread"""
        if self.move_thread is not None and self.move_thread.is_alive():
            print("Movement thread already running")
            return
        
        self.running = True
        self.move_thread = threading.Thread(target=self._movement_loop)
        self.move_thread.daemon = True
        self.move_thread.start()
    
    def stop_movement_thread(self):
        """Stop the movement thread"""
        self.running = False
        if self.move_thread:
            self.move_thread.join(timeout=1.0)
            self.move_thread = None
    
    def _movement_loop(self):
        """Background thread for continuous smooth movement"""
        # Get initial joint positions
        for i, handle in enumerate(self.joint_handles):
            _, angle = sim.simxGetJointPosition(self.client_id, handle, sim.simx_opmode_blocking)
            self.current_angles[i] = angle
        
        while self.running:
            self._execute_movement_step()
            time.sleep(self.movement_pause)
    
    def _execute_movement(self):
        """Execute complete movement until target is reached"""
        # Get initial joint positions
        for i, handle in enumerate(self.joint_handles):
            _, angle = sim.simxGetJointPosition(self.client_id, handle, sim.simx_opmode_blocking)
            self.current_angles[i] = angle
        
        # Continue moving until close to target
        move_count = 0
        while move_count < 50:  # Timeout after 50 steps
            if not self._execute_movement_step():
                break  # Stop if we're close enough
            move_count += 1
            time.sleep(self.movement_pause)
    
    def _execute_movement_step(self):
        """Execute a single step of movement, returns True if more movement needed"""
        # Get current target thread-safely
        with self.lock:
            target = self.target_angles.copy()
        
        # Check distance to target
        total_diff = 0
        for i in range(len(self.joint_handles)):
            diff = abs(target[i] - self.current_angles[i])
            total_diff += diff
            
            # Move one step towards target
            self.current_angles[i] += (target[i] - self.current_angles[i]) * self.movement_step
            
            # Set joint position
            sim.simxSetJointPosition(
                self.client_id, 
                self.joint_handles[i], 
                self.current_angles[i], 
                sim.simx_opmode_oneshot
            )
        
        # Return whether we need more movement (with small threshold)
        return total_diff > 0.01
    
    def move_to_position(self, x, y, z=None):
        """Move the arm end effector to the specified position using true 3-DOF"""
        try:
            # If z is not provided, calculate it from y to create visible movement for joint 3
            if z is None:
                # Scale y to create a visible z movement
                z = y * 0.5
            
            # Store original coordinates
            original_coords = (x, y, z)
            
            # FLIP THE COORDINATES so the arm points in the correct direction
            # This will make the arm face TOWARD the face instead of away from it
            scaled_x = -x * (self.L1 + self.L2) * 0.75  # Negative x to flip orientation
            scaled_y = y * (self.L1 + self.L2) * 0.75
            scaled_z = z * self.L3 * 1.5  # Amplify z movement
            
            # Adjust based on face position
            if abs(x) < 0.2:  # If face is near center horizontally
                # For centered faces, make sure arm is oriented correctly
                scaled_y += 0.3
            else:
                # For non-centered positions
                scaled_y += 0.2
            
            print(f"Original input: ({x:.2f}, {y:.2f}, {z:.2f})")
            print(f"Scaled target: ({scaled_x:.2f}, {scaled_y:.2f}, {scaled_z:.2f})")
            
            # Calculate joint angles using full 3DOF inverse kinematics
            theta1, theta2, theta3 = self.inverse_kinematics_3dof(scaled_x, scaled_y, scaled_z)
            
            # Set target angles for all joints
            target_angles = [theta1, theta2, theta3, 0, 0]
            return self.move_to_angles(target_angles)
        except Exception as e:
            print(f"IK Error: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_movement_thread()
        sim.simxFinish(self.client_id)


# Example usage
if __name__ == "__main__":
    # Create and initialize arm controller
    arm = RobotArmController()
    if not arm.connect():
        exit(1)
    
    if not arm.get_joint_handles():
        exit(1)
    
    # Set initial position
    arm.set_initial_position()
    
    # Start movement thread for smooth operation
    arm.start_movement_thread()
    
    # Test some movements with varying coordinates to demonstrate 3DOF movement
    try:
        print("Moving to position 1 - right side, high")
        arm.move_to_position(0.5, 0.5, 0.3)
        time.sleep(2)
        
        print("Moving to position 2 - left side, high")
        arm.move_to_position(-0.5, 0.5, 0.3)
        time.sleep(2)
        
        print("Moving to position 3 - center, highest")
        arm.move_to_position(0, 0.8, 0.5)
        time.sleep(2)
        
        print("Moving to position 4 - center, lowest")
        arm.move_to_position(0, 0.2, -0.5)
        time.sleep(2)
        
        print("Moving to initial position")
        arm.set_initial_position()
        time.sleep(2)
    finally:
        # Clean up
        arm.cleanup()