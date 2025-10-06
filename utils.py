import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split

import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)


# Function to preprocess the input image
# Load Mediapipe for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# Load the ASL Keras model (ensure the model is only loaded once)
model = tf.keras.models.load_model('models/signify_ASL_image_classification_model_ver4.h5')


# Data collection variables for left and right hands
left_data = []
left_labels = []
right_data = []
right_labels = []

from gtts import gTTS
import pygame
import os

def tts(text):
    # Convert text to speech and save as an mp3 file
    tts = gTTS(text=text, lang='en')
    filename = "temp_audio.mp3"
    tts.save(filename)


    #Initialize pygame mixer to play the sound
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    # Wait for the sound to finish playing
    while pygame.mixer.music.get_busy():
        continue

    # Cleanup: remove the temporary audio file
    os.remove(filename)

try:
    import sounddevice as sd
except Exception:
    sd = None
    print("⚠️ sounddevice not available — skipping audio features")

import speech_recognition as sr
from scipy.io.wavfile import write
import numpy as np

def record_audio(duration, filename="temp_audio.wav", fs=44100):
    print("Recording...")
    if sd is None:
        raise RuntimeError("Audio recording not supported in this environment")

    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    write(filename, fs, audio_data)  # Save as WAV file
    print("Recording finished.")

def recognize_speech(filename="temp_audio.wav"):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            print("You said:", text)
            return text
        except sr.UnknownValueError:
            print("Sorry, I did not understand the audio.")
        except sr.RequestError:
            print("Sorry, there was an issue with the API request.")


def make_png(text):
    output_filename = "combined_image.png"
    create_combined_image(text, output_filename)

    
from PIL import Image

def create_combined_image(text, output_filename, space_width=80):
    images = []
    
    # 각 문자를 이미지 또는 공백으로 변환
    for char in text:
        if char == ' ':
            # 공백은 투명 배경의 빈 이미지를 추가
            img = Image.new('RGBA', (space_width, 100))  # 높이 100으로 기본 설정
            images.append(img)
        else:
            image_filename = f"static/images/ASL_alphabet/{char.upper()}.png"  # 예를 들어 A.png, B.png 등
            try:
                img = Image.open(image_filename)
                images.append(img)
            except FileNotFoundError:
                print(f"{image_filename} 파일을 찾을 수 없습니다.")
                continue
    
    # 연결할 이미지의 총 너비와 최대 높이 계산
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    
    # 새로운 빈 이미지 생성
    combined_image = Image.new('RGBA', (total_width, max_height))  # RGBA로 설정하여 투명 배경 지원
    
    # 이미지를 옆으로 이어붙이기
    x_offset = 0
    for img in images:
        combined_image.paste(img, (x_offset, (max_height - img.height) // 2))  # 세로 중앙 정렬
        x_offset += img.width
    
    # 최종 결과물 저장
    combined_image.save(output_filename)
    print(f"결과 이미지가 {output_filename} 파일로 저장되었습니다.")

from openai import OpenAI
client = OpenAI(api_key="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")



def kslstringcorrection(string):
    print(f"STRING CORRECTION REQUESTED: {string}")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "reconcile the given korean letters into sentence. don't add or remove any letters."
            },
            {
                "role": "user",
                "content": string
            }
        ],
        temperature=0.7,
        max_tokens=64,
        top_p=1
    )
    
    correctedText = response.choices[0].message.content  # Corrected access method

    print(f"GPT Response:" + correctedText)
    
    return correctedText



def aslstringcorrection(string):
    print(f"STRING CORRECTION REQUESTED: {string}")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You will be provided with statements, and your task is to convert them to standard English, in one word. Even if it's gibberish, just create, to your best of your ability, the best word you think that matches. Don't write anything else, just the one word. If it is already a word in the dictionary, do not change it. eg. if the current word is happiness, do not change it to joy. keep it as happiness."
            },
            {
                "role": "user",
                "content": string
            }
        ],
        temperature=0.7,
        max_tokens=64,
        top_p=1
    )
    
    correctedText = response.choices[0].message.content  # Corrected access method

    print(f"GPT Response:" + correctedText)
    
    return correctedText

# test = input("Type the sentence you want to correct: ")
# print(stringcorrection(test))

# Load pre-trained left hand model
left_model = tf.keras.models.load_model('models/left_hand_model_vfinal.h5')

# Load pre-trained right hand model
right_model = tf.keras.models.load_model('models/right_hand_model_vfinal.h5')

def classify_image_old(image):
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
          
        image = cv2.flip(image,1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract landmark coordinates
                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.append([lm.x, lm.y, lm.z])

                # Flatten and reshape
                input_data = np.array(landmark_list).flatten().reshape(1, -1)
                prediction = model.predict(input_data)
                predicted_index = np.argmax(prediction)
                predicted_label = chr(predicted_index + ord('A')) if predicted_index < 26 else '-'
                print("predicted = ",predicted_label)
                return predicted_label
    return '-'




saved_gestures = []
right_hand_detected = False

# Function to classify an ASL gesture in an image
def classify_image(image):

    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
          
        image = cv2.flip(image,1)
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

                    if hand_label == 'Left' and left_predicted_label == 'shoot':
                        current_time = cv2.getTickCount() / cv2.getTickFrequency()
                        if 'last_shoot_time' not in locals() or current_time - last_shoot_time >= 1:
                            if right_hand_detected is True:
                                saved_gestures.append(current_right_label)
                                last_shoot_time = current_time
                    right_hand_detected = False
                elif hand_label == 'Right':
                    right_prediction = right_model.predict(input_data)
                    right_predicted_index = np.argmax(right_prediction)
                    current_right_label = chr(right_predicted_index + ord('A')) if right_predicted_index < 26 else '-'
                    right_hand_detected = True
                else:
                    right_hand_detected = False

                # Display the current right-hand prediction
                cv2.putText(image, f'Right: {current_right_label}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
            # Display saved gestures
            cv2.putText(image, f'Saved: {" ".join(saved_gestures)}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
           # print("saved_gestures = ",saved_gestures)
           # print("saved_gestures = ",current_right_label)
 
            return current_right_label
    return ''



