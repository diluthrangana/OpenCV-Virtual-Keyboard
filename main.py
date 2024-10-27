import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

# Initialize the VideoCapture object with the camera index (usually 0 for the default camera)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width to 1280
cap.set(4, 720)   # Set height to 720

# Initialize the HandDetector with higher confidence for robust detection
detector = HandDetector(detectionCon=0.9, maxHands=2)

# Define a Button class for the virtual keyboard
class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text

    # Draw the button on the provided image
    def draw(self, img):
        x, y = self.pos
        w, h = self.size

        # Draw a semi-transparent white rectangle with a white border
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 255), cv2.FILLED)
        img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)  # Adjust transparency here

        # Draw the white border
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # Draw the text on the button
        cv2.putText(img, self.text, (x + 20, y + 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 5)

        return img

    # Check if the fingertip is hovering over the button
    def is_hover(self, x, y):
        return self.pos[0] < x < self.pos[0] + self.size[0] and self.pos[1] < y < self.pos[1] + self.size[1]

# Define keys for each row of the keyboard
keys = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
    ["Z", "X", "C", "V", "B", "N", "M"]
]

# Create Button objects for each key, arranged by rows
buttons = []
for i, row in enumerate(keys):
    for j, key in enumerate(row):
        buttons.append(Button([100 + j * 90, 100 + i * 100], key))

# Initialize an empty string to store the typed text
typed_text = ""

# Threshold distance to detect "click" between index and middle fingers
click_threshold = 30  # Adjust based on finger distance tolerance

# Draw the keyboard overlay with transparency
def draw_keyboard_overlay(img, buttons):
    overlay = img.copy()
    for button in buttons:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 255), cv2.FILLED)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(img, button.text, (x + 20, y + 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 5)

    # Apply transparency
    img = cv2.addWeighted(overlay, 0.2, img, 0.8, 0)
    return img

while True:
    success, img = cap.read()
    if not success:
        break  # Break the loop if the camera fails to capture

    # Flip the image horizontally
    img = cv2.flip(img, 1)

    # Draw keyboard overlay with transparency
    img = draw_keyboard_overlay(img, buttons)

    # Detect hands in the frame and get the list of hands
    hands, img = detector.findHands(img)  # Draws hand landmarks on the image

    # Process detected hands
    if hands:
        # Retrieve the landmark list for the first detected hand
        lmList = hands[0]["lmList"]  # List of 21 Landmark points (x, y, z)

        # Check if the index finger is over any button
        for button in buttons:
            if button.is_hover(lmList[8][0], lmList[8][1]):  # lmList[8] is the index fingertip
                # Measure distance between index finger (lmList[8]) and middle finger (lmList[12])
                distance = math.hypot(lmList[8][0] - lmList[12][0], lmList[8][1] - lmList[12][1])

                # If fingers are close enough, simulate a button click
                if distance < click_threshold:
                    # Add the key to typed_text and print it
                    typed_text += button.text
                    print("Typed Text:", typed_text)

                    # Display click effect by changing button color temporarily
                    cv2.rectangle(img, (button.pos[0], button.pos[1]),
                                  (button.pos[0] + button.size[0], button.pos[1] + button.size[1]),
                                  (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, button.text, (button.pos[0] + 20, button.pos[1] + 60),
                                cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 5)

                    # Add delay to prevent repeated detection
                    cv2.waitKey(300)

    # Display the typed text
    cv2.rectangle(img, (50, 400), (1200, 500), (50, 50, 50), cv2.FILLED)
    cv2.putText(img, typed_text, (60, 470), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

    # Show the image
    cv2.imshow("Virtual Keyboard", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on pressing 'q'
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
