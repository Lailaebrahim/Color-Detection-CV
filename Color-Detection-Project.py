import cv2
import numpy as np
from get_limits import get_limits

# cap = cv2.VideoCapture('./videos/1.mp4')
# cap = cv2.VideoCapture('./videos/1.mp4')
cap = cv2.VideoCapture('./videos/27.04.2025_22.08.42_REC.mp4')
# cap = cv2.VideoCapture('./videos/4762243-hd_1920_1080_25fps.mp4')

def cleck_event(event, x, y, flags, param):
    if event == cv2.EVENT_RBUTTONDOWN:     
        cv2.waitKey(-1)

    elif event == cv2.EVENT_LBUTTONDOWN:
        cv2.waitKey(1)


while True:
    # read frame
    ret, frame = cap.read()

    # if no frame is captured, break the loop
    if not ret: break
    
    ################################# preprocessing #################################
    # resize the frame to a fixed size to reduce processing time 
    frame = cv2.resize(frame, (640, 480))

    # using HSV color space for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # apply Gaussian blur to reduce noise and improve color detection
    blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
    
    # mask the image to get the desired color pixels
    # red mask
    lower_red, upper_red = get_limits((0, 0, 255))
    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    kernel = np.ones((5,5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)

    # yellow mask 
    lower_yellow, upper_yellow = get_limits((0, 255, 255))
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = np.ones((5,5), np.uint8)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)

    # green mask
    lower_green, upper_green = get_limits((0, 255, 0))
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    light_green_lower = np.array([40, 50, 50])
    light_green_upper = np.array([80, 255, 255])
    mask_green += cv2.inRange(hsv, light_green_lower, light_green_upper)

    kernel = np.ones((5,5), np.uint8)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
    ################################# end preprocessing #################################



    ################################# feature extraction #################################
    # find contours for each mask separately
    red_contours, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    yellow_contours, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    green_contours, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ################################# end feature extraction #################################



    ################################# draw contours #################################
    for cnt in red_contours:
        if cv2.contourArea(cnt) > 1000:
            # x,y top left corner coordinates and w, h width and height of the bounding rectangle
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    for cnt in yellow_contours:
        if cv2.contourArea(cnt) > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    for cnt in green_contours:
        if cv2.contourArea(cnt) > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    ############################### end draw contours #################################

    # show frame after color detection
    cv2.imshow("Tracking", frame)

    # add mouse callback to the window -> left click stop the program, right click to continue
    cv2.setMouseCallback('Tracking', cleck_event)

    # press 'q' to exit the program
    if cv2.waitKey(10) & 0xFF == ord('q'): 
        break


cap.release()
cv2.destroyAllWindows()