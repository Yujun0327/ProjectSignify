import mediapipe as mp
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load Mediapipe for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Data collection variables for left and right hands
left_data = []
left_labels = []
right_data = []
right_labels = []


# Load pre-trained left hand model
left_model = tf.keras.models.load_model('left_hand_model_v7.keras')

# Load pre-trained right hand model
right_model = tf.keras.models.load_model('right_hand_model_v7.keras')

# Real-time prediction for both hands
def classify_image(image):
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
        right_hand_detected = False  # Initialize right_hand_detected at the beginning of the function
        current_right_label = ''  # Initialize current_right_label to avoid scope issues
          
        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_label = results.multi_handedness[idx].classification[0].label
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmark_list = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                input_data = np.array(landmark_list).flatten().reshape(1, -1)
                
                if hand_label == 'Left':
                    left_prediction = left_model.predict(input_data)
                    left_predicted_index = np.argmax(left_prediction)
                    left_predicted_label = 'shoot' if left_predicted_index == 1 else 'not_shoot'

                    # Display the current left-hand prediction
                    cv2.putText(image, f'Left: {left_predicted_label}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    # Save gesture if conditions are met
                    current_time = cv2.getTickCount() / cv2.getTickFrequency()
                    if left_predicted_label == 'shoot' and (current_time - last_shoot_time >= 1):
                        if right_hand_detected:
                            saved_gestures.append(current_right_label)
                            last_shoot_time = current_time

                    right_hand_detected = False  # Reset after processing left hand

                elif hand_label == 'Right':
                    right_prediction = right_model.predict(input_data)
                    right_predicted_index = np.argmax(right_prediction)
                    current_right_label = chr(right_predicted_index + ord('A')) if right_predicted_index < 26 else '-'
                    right_hand_detected = True  # Mark that right hand was detected
                else:
                    right_hand_detected = False

                # Display the current right-hand prediction
                cv2.putText(image, f'Right: {current_right_label}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display saved gestures
        cv2.putText(image, f'Saved: {" ".join(saved_gestures)}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return current_right_label if right_hand_detected else ''


    cap.release()
    cv2.destroyAllWindows()

# Run real-time prediction
predict_sign()
