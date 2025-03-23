import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Capture Video from Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Make Detector
detector = HandDetector(detectionCon=0.8)
StartDist = None
scale = 1.0  # Start with normal size
cx, cy = 640, 360  # Center the image initially

# Load Image
img1 = cv2.imread('tanjiro.jpg')
if img1 is None:
    raise ValueError("Error: 'tanjiro.jpg' not found or could not be loaded.")

# Initialize previous values for smoothing (use float for precision)
prev_cx, prev_cy, prev_scale = float(cx), float(cy), float(scale)
alpha = 0.1  # Lower = Smoother, Higher = More Responsive

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from webcam")
        continue  # Skip iteration if no frame is captured

    hands, img = detector.findHands(img)

    if len(hands) == 2:
        fingers1 = detector.fingersUp(hands[0])
        fingers2 = detector.fingersUp(hands[1])

        print(f"Fingers Up (Hand 1): {fingers1}")
        print(f"Fingers Up (Hand 2): {fingers2}")

        # Detect if both hands are showing two fingers
        if sum(fingers1) == 2 and sum(fingers2) == 2:
            lmList1 = hands[0]["lmList"]
            lmList2 = hands[1]["lmList"]

            if len(lmList1) > 8 and len(lmList2) > 8:
                p1 = lmList1[8][:2]  # Index finger (x, y)
                p2 = lmList2[8][:2]  # Index finger (x, y)

                # Calculate distance
                length, info, img = detector.findDistance(p1, p2, img)

                # Dynamically update StartDist for better zooming
                if StartDist is None:
                    StartDist = length

                # New scale factor based on hand distance
                new_scale = (length / StartDist)  # Ratio-based scaling

                # Avoid excessive shrinking
                new_scale = max(0.2, min(new_scale, 3.0))  # Keep scale in a valid range

                # Find midpoint between hands
                new_cx, new_cy = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2

                # Apply Exponential Moving Average (EMA) for smooth transitions
                scale = alpha * new_scale + (1 - alpha) * prev_scale
                cx = alpha * new_cx + (1 - alpha) * prev_cx
                cy = alpha * new_cy + (1 - alpha) * prev_cy

                prev_scale, prev_cx, prev_cy = scale, cx, cy  # Store for next frame

                print(f"Smooth Scale: {scale:.2f}, Center: ({cx:.2f}, {cy:.2f})")  # Debugging

    else:
        StartDist = None  # Reset scale when hands are not detected

    # Resize Image Based on Scale Factor
    h1, w1, _ = img1.shape
    newH, newW = int(h1 * scale), int(w1 * scale)

    # Ensure even dimensions for stability
    newH = max(50, (newH // 2) * 2)  # Min size 50px
    newW = max(50, (newW // 2) * 2)

    img1_resized = cv2.resize(img1, (newW, newH))

    # Ensure cropping region stays within bounds of img
    h, w, _ = img.shape  # Get frame size

    x1, x2 = max(0, int(cx - newW // 2)), min(w, int(cx + newW // 2))
    y1, y2 = max(0, int(cy - newH // 2)), min(h, int(cy + newH // 2))

    # Ensure the img1 overlay fits inside img correctly
    try:
        overlay = img1_resized[:y2 - y1, :x2 - x1]
        background = img[y1:y2, x1:x2]

        # Alpha blending (change opacity level with `alpha_value`)
        alpha_value = 0.5  # Adjust transparency (0 = fully transparent, 1 = fully opaque)
        blended = cv2.addWeighted(overlay, alpha_value, background, 1 - alpha_value, 0)

        img[y1:y2, x1:x2] = blended  # Place blended image back
    except:
        print("Image size out of bounds, adjusting...")  # Debugging

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()