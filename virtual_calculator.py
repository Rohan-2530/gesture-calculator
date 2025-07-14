import cv2
import mediapipe as mp
import numpy as np

# --- Calculator button layout ---
button_labels = [
    ["7", "8", "9", "/"],
    ["4", "5", "6", "*"],
    ["1", "2", "3", "-"],
    ["C", "0", "=", "+"]
]

#added new line

# --- Size and color settings for BIG, YELLOW calculator ---
button_size = (140, 100)  # width, height (bigger)
button_gap = 30           # space between buttons
start_x, start_y = 60, 180  # top-left corner (move down for bigger display)

button_color = (0, 255, 255)   # Yellow (BGR)
button_text_color = (0, 0, 0)  # Black text for contrast
display_color = (230, 230, 230)
display_text_color = (0, 0, 0)

current_input = ""
result = ""

def draw_buttons(frame):
    # Draw input/result display (bigger)
    cv2.rectangle(
        frame,
        (start_x, start_y - 130),
        (start_x + 4 * (button_size[0] + button_gap) - button_gap, start_y - 30),
        display_color,
        -1
    )
    cv2.putText(
        frame,
        result if result else current_input,
        (start_x + 20, start_y - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.5,
        display_text_color,
        5
    )
    # Draw buttons
    for i, row in enumerate(button_labels):
        for j, label in enumerate(row):
            x = start_x + j * (button_size[0] + button_gap)
            y = start_y + i * (button_size[1] + button_gap)
            cv2.rectangle(
                frame,
                (x, y),
                (x + button_size[0], y + button_size[1]),
                button_color,
                -1  # Filled rectangle
            )
            cv2.rectangle(
                frame,
                (x, y),
                (x + button_size[0], y + button_size[1]),
                (0, 200, 200),
                5  # Border
            )
            cv2.putText(
                frame,
                label,
                (x + 45, y + 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.5,
                button_text_color,
                5
            )

# --- Mouse click detection ---
clicked_button = None
def mouse_callback(event, x, y, flags, param):
    global clicked_button
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, row in enumerate(button_labels):
            for j, label in enumerate(row):
                bx = start_x + j * (button_size[0] + button_gap)
                by = start_y + i * (button_size[1] + button_gap)
                if bx < x < bx + button_size[0] and by < y < by + button_size[1]:
                    clicked_button = label

# --- MediaPipe setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
cv2.namedWindow("Virtual Calculator")
cv2.setMouseCallback("Virtual Calculator", mouse_callback)

last_pressed = None  # For gesture debounce

while True:
    ret, frame = cap.read()
    if not ret:
        print("No video feed from camera. Please check your webcam connection and permissions.")
        break

    frame = cv2.flip(frame, 1)  # Mirror the image for natural interaction

    # --- Hand gesture detection ---
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    finger_tip = None
    thumb_tip = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            # Index finger tip (8), Thumb tip (4)
            x_index = int(hand_landmarks.landmark[8].x * w)
            y_index = int(hand_landmarks.landmark[8].y * h)
            x_thumb = int(hand_landmarks.landmark[4].x * w)
            y_thumb = int(hand_landmarks.landmark[4].y * h)
            finger_tip = (x_index, y_index)
            thumb_tip = (x_thumb, y_thumb)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    draw_buttons(frame)

    # --- Gesture button press detection (thumb-index touch) ---
    if finger_tip and thumb_tip:
        distance = np.hypot(finger_tip[0] - thumb_tip[0], finger_tip[1] - thumb_tip[1])
        if distance < 50:  # Adjusted for bigger buttons
            for i, row in enumerate(button_labels):
                for j, label in enumerate(row):
                    bx = start_x + j * (button_size[0] + button_gap)
                    by = start_y + i * (button_size[1] + button_gap)
                    if bx < finger_tip[0] < bx + button_size[0] and by < finger_tip[1] < by + button_size[1]:
                        # Highlight button
                        cv2.rectangle(
                            frame,
                            (bx, by),
                            (bx + button_size[0], by + button_size[1]),
                            (0, 0, 255),
                            8
                        )
                        # Debounce: only register new press if finger moves to a new button
                        if last_pressed != label:
                            clicked_button = label
                            last_pressed = label
                        break
                else:
                    continue
                break
            else:
                last_pressed = None
        else:
            last_pressed = None
    else:
        last_pressed = None

    # --- Handle button press (mouse or gesture) ---
    if clicked_button:
        if clicked_button in "0123456789":
            current_input += clicked_button
            result = ""
        elif clicked_button in "+-*/":
            if current_input and current_input[-1] not in "+-*/":
                current_input += clicked_button
                result = ""
        elif clicked_button == "C":
            current_input = ""
            result = ""
        elif clicked_button == "=":
            try:
                result = str(eval(current_input))
                current_input = ""
            except Exception:
                result = "Error"
                current_input = ""
        clicked_button = None

    cv2.imshow("Virtual Calculator", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
hands.close()
cv2.destroyAllWindows()

