import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create a blank canvas
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
canvas = np.zeros((height, width, 3), dtype=np.uint8)

# Drawing parameters
draw_color = (0, 0, 255)  # Red for drawing
eraser_color = (0, 0, 0)  # Black for erasing
draw_thickness = 5  # Thickness for drawing
erase_thickness = 20  # Larger thickness for erasing
prev_x, prev_y = None, None

# Function to calculate distance between two points
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Function to detect finger gestures
def detect_gesture(hand_landmarks, h, w):
    # Get landmarks for finger tips and PIP joints
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]

    # Convert to pixel coordinates
    index_tip_x, index_tip_y = int(index_tip.x * w), int(index_tip.y * h)
    middle_tip_x, middle_tip_y = int(middle_tip.x * w), int(middle_tip.y * h)

    # Check if fingers are extended (tip y-coordinate above PIP y-coordinate)
    index_extended = index_tip.y < index_pip.y - 0.05
    middle_extended = middle_tip.y < middle_pip.y - 0.05
    ring_extended = ring_tip.y < ring_pip.y - 0.05
    pinky_extended = pinky_tip.y < pinky_pip.y - 0.05

    # Calculate distance between index and middle finger tips
    index_middle_distance = calculate_distance(index_tip_x, index_tip_y, middle_tip_x, middle_tip_y)

    # Detect gestures
    if (index_extended and not middle_extended and not ring_extended and not pinky_extended):
        return 'draw', index_tip_x, index_tip_y  # Index finger alone for drawing
    elif (index_extended and middle_extended and index_middle_distance < 30 and 
          not ring_extended and not pinky_extended):
        return 'erase', index_tip_x, index_tip_y  # Index and middle together for erasing
    elif index_extended and middle_extended and ring_extended and pinky_extended:
        return 'pause', None, None  # Open palm for pause
    return 'none', None, None  # No action for other gestures

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get frame dimensions
            h, w, _ = frame.shape

            # Detect gesture and get drawing coordinates
            action, x, y = detect_gesture(hand_landmarks, h, w)

            # Perform action based on gesture
            if action == 'draw':
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, draw_thickness)
                prev_x, prev_y = x, y
                # Draw a circle to indicate the tracked point
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            elif action == 'erase':
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), eraser_color, erase_thickness)
                prev_x, prev_y = x, y
                # Draw a circle to indicate the tracked point
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            else:
                prev_x, prev_y = None, None  # Reset for pause or no action

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    else:
        prev_x, prev_y = None, None  # Reset when hand is not detected

    # Combine frame and canvas
    output = cv2.add(frame, canvas)

    # Display action
    action = 'none' if not results.multi_hand_landmarks else detect_gesture(hand_landmarks, h, w)[0]
    cv2.putText(output, f'Action: {action.capitalize()}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the output in a normal window
    cv2.imshow('Air Drawing', output)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()