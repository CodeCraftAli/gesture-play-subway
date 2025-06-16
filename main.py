import cv2
import mediapipe as mp
import numpy as np
from pynput.keyboard import Controller, Key
import time

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize Keyboard Controller
keyboard = Controller()


# Main loop
cap = cv2.VideoCapture(1)  # use 0 if 1 doesn't work
prev_time = 0
last_action_time = 0
action_cooldown = 0.5
last_action = None
action_duration = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_height, frame_width, _ = frame.shape
    mid_x, mid_y = frame_width // 2, frame_height // 2

    # Draw lines
    cv2.line(frame, (mid_x, 0), (mid_x, frame_height), (0, 255, 0), 2)
    cv2.line(frame, (0, mid_y), (frame_width, mid_y), (0, 255, 0), 2)

    results = hands.process(rgb_frame)
    action = None

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Use index finger tip (landmark 8)
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        x = int(index_tip.x * frame_width)
        y = int(index_tip.y * frame_height)

        # Determine direction
        if x < mid_x - 50:
            action = "LEFT"
        elif x > mid_x + 50:
            action = "RIGHT"
        elif y < mid_y - 50:
            action = "UP"
        elif y > mid_y + 50:
            action = "DOWN"
        else:
            action = last_action


    # Simulate key presses

    current_time = time.time()
    if action and action != last_action and current_time - last_action_time > action_cooldown:
        print(f"Action performed: {action}")
        if action == "LEFT":
            keyboard.press(Key.left)
            keyboard.release(Key.left)
        elif action == "RIGHT":
            keyboard.press(Key.right)
            keyboard.release(Key.right)
        elif action == "UP":
            keyboard.press(Key.up)
            keyboard.release(Key.up)
        elif action == "DOWN":
            keyboard.press(Key.down)
            keyboard.release(Key.down)
        last_action_time = current_time
        last_action = action

    # Display action

    display_action = last_action if current_time - last_action_time <= action_duration else "None"
    cv2.putText(frame, f"Action: {display_action}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)


    cv2.imshow("Hand Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
