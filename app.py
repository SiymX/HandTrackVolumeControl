import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import time

class HandTrackingModule:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return img

    def find_position(self, img, hand_no=0, draw=True):
        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]

            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        
        return lm_list

    def draw_volume_line(self, img, x1, y1, x2, y2, distance):
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        radius = int(np.interp(distance, [50, 300], [10, 50]))
        cv2.circle(img, (center_x, center_y), radius, (0, 255, 0), cv2.FILLED)

tracker = HandTrackingModule()

cap = cv2.VideoCapture(0)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]


while True:
    start = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    frame = tracker.find_hands(frame)
    lm_list = tracker.find_position(frame)
    current_vol = None

    if lm_list:
        
        x1, y1 = lm_list[4][1], lm_list[4][2]  
        x2, y2 = lm_list[8][1], lm_list[8][2]  

        distance = math.hypot(x2 - x1, y2 - y1)

        vol = np.interp(distance, [50, 300], [min_vol, max_vol])
        volume.SetMasterVolumeLevel(vol, None)

        current_vol = volume.GetMasterVolumeLevelScalar()
        vol_percentage = int(current_vol * 100)

        tracker.draw_volume_line(frame, x1, y1, x2, y2, distance)

    end = time.time()
    fps = 1 / (end - start)

    cv2.putText(frame, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    if current_vol is not None:
        cv2.putText(frame, f'Vol: {vol_percentage}%', (10, 120), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Handtrack", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Handtrack", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()