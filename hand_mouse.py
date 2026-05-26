"""
Hand Gesture Mouse Controller
- Move cursor by moving your middle finger
- Click by pinching (bringing thumb and index finger together)

NOTE: Run with Python 3.10: python hand_mouse.py
"""

import cv2
import mediapipe as mp
import pyautogui
import math

# Disable PyAutoGUI fail-safe (optional - remove if you want fail-safe)
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Camera settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Variables for smoothing mouse movement
smooth_factor = 5
prev_x, prev_y = 0, 0

# Click state
clicked = False
click_threshold = 40  # Distance threshold for pinch detection


def get_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def main():
    global prev_x, prev_y, clicked
    
    print("Hand Gesture Mouse Controller Started!")
    print("- Move your MIDDLE FINGER to move the cursor")
    print("- PINCH (thumb + index finger together) to click")
    print("- Press 'Q' to quit")
    print("-" * 50)
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hand detection
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )
                
                # Get landmark positions
                landmarks = hand_landmarks.landmark
                
                # Middle finger tip (landmark 12) - for cursor movement
                middle_tip = landmarks[12]
                middle_x = int(middle_tip.x * frame_width)
                middle_y = int(middle_tip.y * frame_height)
                
                # Index finger tip (landmark 8) - for click detection
                index_tip = landmarks[8]
                index_x = int(index_tip.x * frame_width)
                index_y = int(index_tip.y * frame_height)
                
                # Thumb tip (landmark 4) - for click detection
                thumb_tip = landmarks[4]
                thumb_x = int(thumb_tip.x * frame_width)
                thumb_y = int(thumb_tip.y * frame_height)
                
                # Draw circles on fingertips
                cv2.circle(frame, (middle_x, middle_y), 10, (0, 255, 0), -1)  # Green - cursor control
                cv2.circle(frame, (index_x, index_y), 10, (255, 0, 0), -1)    # Blue - click
                cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 255, 255), -1)  # Yellow - click
                
                # Calculate screen coordinates from middle finger position
                screen_x = int(middle_tip.x * screen_width)
                screen_y = int(middle_tip.y * screen_height)
                
                # Smooth the mouse movement
                curr_x = prev_x + (screen_x - prev_x) / smooth_factor
                curr_y = prev_y + (screen_y - prev_y) / smooth_factor
                
                # Move the mouse cursor
                pyautogui.moveTo(curr_x, curr_y)
                
                prev_x, prev_y = curr_x, curr_y
                
                # Check for pinch gesture (click detection)
                distance = get_distance((index_x, index_y), (thumb_x, thumb_y))
                
                # Draw line between thumb and index
                if distance < click_threshold:
                    cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 0, 255), 3)
                else:
                    cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 0), 2)
                
                # Perform click
                if distance < click_threshold:
                    if not clicked:
                        pyautogui.click()
                        clicked = True
                        cv2.putText(frame, "CLICK!", (50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        print("Click!")
                else:
                    clicked = False
                
                # Display distance on screen
                cv2.putText(frame, f"Distance: {int(distance)}", (10, frame_height - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display instructions on frame
        cv2.putText(frame, "Middle finger: Move | Thumb+Index: Click | Q: Quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Show the frame
        cv2.imshow("Hand Gesture Mouse Control", frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Program ended.")


if __name__ == "__main__":
    main()
