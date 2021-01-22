import cv2
import dlib
import time
import threading
import math
import imutils
import queue
import numpy as np
from imutils import contours

carCascade = cv2.CascadeClassifier('../Classifier/myhaar.xml')
lineCascade = cv2.CascadeClassifier('../Classifier/linecascade.xml')
video = cv2.VideoCapture('../Videos/default.mp4')

WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# def estimateSpeed(location1, location2):
#     d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
#     # ppm = location2[2] / carWidht
#     ppm = 8.8
#     d_meters = d_pixels / ppm
#     # print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
#     fps = video.get(cv2.CAP_PROP_FPS)
#     speed = d_meters * fps * 3.6
#     return speed

def estimateSpeed(location1, location2, lineTracker, frame, cnts, resultImage):
    d_pixels = distanceFormula(location1[0], location1[1], location2[0], location2[1])
    closestLine = findClosestLine(lineTracker, location2)

    # Use closest traffic line to measure pixel per metric
    metric = 3
    ppm = getPixelPerMetric(closestLine, cnts, metric, resultImage, location2)
    if ppm is None:
        return None

    d_meters = d_pixels / ppm
    fps = getFPS()
    speed = d_meters * fps * 3.6
    # print("ppm: ", ppm, "| d_mtrs: ", d_meters, "| spd: ", speed)
    return speed

def distanceFormula(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def getFPS():
    return video.get(cv2.CAP_PROP_FPS)

def getTrafficLines(frame):
    lineTracker = {}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lines = lineCascade.detectMultiScale(gray, 1.1, 4, 10, (1,1), (7,7))
    lineID = 0
    for (_x, _y, _w, _h) in lines:
        x = int(_x)
        y = int(_y)
        w = int(_w)
        h = int(_h)
        lineTracker[lineID] = [x, y, w, h]
        lineID = lineID + 1
    return lineTracker

def findClosestLine(lineTracker, carLocation):
    closest = HEIGHT + WIDTH
    closestLine = None

    # (carX, carY) reflects the midpoint on the bottom bar of vehicle rectangle
    carX = carLocation[0] + carLocation[2] / 2
    carY = carLocation[1] + carLocation[3]
    for lineID in lineTracker.keys():
        lineXCenter = lineTracker[lineID][0] + lineTracker[lineID][2] / 2
        lineYCenter = lineTracker[lineID][1] + lineTracker[lineID][3] / 2
        distance = distanceFormula(carX, carY, lineXCenter, lineYCenter)
        if (distance < closest):
            closest = distance
            closestLine = lineTracker[lineID]
    return closestLine

def showTrafficLines(lines, resultImage):
    for lineID in lines.keys():
        t_x, t_y, t_w, t_h = lines[lineID]
        cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), (0, 255, 255), 1)

def getPixelPerMetric(line, cnts, metric, resultImage, carLocation):
    # find contour of closest detected line
    lx, ly, lw, lh = line
    lineContour = None
    box = None

    for c in reversed(cnts):
        cx, cy, cw, ch = cv2.boundingRect(c)
        if (lx < cx and ly < cy and (lx+lw) > (cx+cw) and (ly+lh) > (cy+ch)):
            lineContour = cv2.boundingRect(c)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            break

    if lineContour is None:
        return None
    else:
        x, y, w, h = lineContour

        # This draws the detected traffic line and and a line leading to it from the car
        cv2.drawContours(resultImage, [box], 0, (255, 255, 0), 2)
        if (distanceFormula(carLocation[0], carLocation[1]+carLocation[3], box[0][0], box[0][1]) <
            distanceFormula(carLocation[0]+carLocation[2], carLocation[1]+carLocation[3], box[0][0], box[0][1])):
            cv2.line(resultImage, (box[0][0],box[0][1]), (carLocation[0], carLocation[1]+carLocation[3]), (255,0,0), 2)
        else:
            cv2.line(resultImage, (box[0][0],box[0][1]), (carLocation[0]+carLocation[2], carLocation[1]+carLocation[3]), (255,0,0), 2)

        pixelsPerMetric = distanceFormula(x, y, x+w, y+h) / metric
        return pixelsPerMetric

def getContourMap(frame):
    # convert to grayscale, and blur it slightly
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=2)
    edged = cv2.erode(edged, None, iterations=2)

    #  show edge map
    # cv2.imshow('Edges', edged)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return sorted(cnts, key=cv2.contourArea)

def trackMultipleObjects():
    fps = getFPS()
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0
    cnts = None
    lineTracker = None
    carTracker = {}
    carLocation = {}
    speed = {}

    # Write output to video file
    out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (WIDTH, HEIGHT))

    while True:
        ret, frame = video.read()

        if type(frame) == type(None):
            break

        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        resultImage = frame.copy()

        # Remove low quality tracked vehicles
        for carID in list(carTracker):
            trackingQuality = carTracker[carID].update(frame)
            if trackingQuality < 6:
                print('Removing low quality tracker ' + str(carID))
                del carTracker[carID]
                del carLocation[carID]
                del speed[carID]

        # Identifying the vehicles, traffic lines, and contour lines are expensive so we limit it to once per second
        # Map identified vehicles to ones we are already tracking, then add new trackers for the ones we are not
        if (frameCounter % fps == 0):
            cnts = getContourMap(frame)
            lineTracker = getTrafficLines(frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.1, 20, 18, (25, 25))

            for (_x, _y, _w, _h) in cars:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)

                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h

                matchCarID = None

                for carID in carTracker.keys():
                    trackedPosition = carTracker[carID].get_position()

                    t_x = int(trackedPosition.left())
                    t_y = int(trackedPosition.top())
                    t_w = int(trackedPosition.width())
                    t_h = int(trackedPosition.height())

                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h

                    if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (
                            x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                        matchCarID = carID

                if matchCarID is None:
                    print('Creating new tracker ' + str(currentCarID))

                    tracker = dlib.correlation_tracker()
                    tracker.start_track(frame, dlib.rectangle(x, y, x + w, y + h))

                    carTracker[currentCarID] = tracker
                    carLocation[currentCarID] = [x, y, w, h]
                    speed[currentCarID] = queue.Queue()

                    currentCarID = currentCarID + 1

        # test traffic line classifier
        showTrafficLines(lineTracker, resultImage)

        # go through tracker, get predicted position from tracker and compare to previous position for estimated speed
        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()

            x2 = int(trackedPosition.left())
            y2 = int(trackedPosition.top())
            w2 = int(trackedPosition.width())
            h2 = int(trackedPosition.height())

            cv2.rectangle(resultImage, (x2, y2), (x2+w2, y2+h2), rectangleColor, 3)

            if carLocation[carID] != [x2, y2, w2, h2]:
                q = speed[carID]
                # speed[carID] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])
                curSpeed = estimateSpeed(carLocation[carID], [x2, y2, w2, h2], lineTracker, frame, cnts, resultImage)
                if (curSpeed):
                    q.put(curSpeed)
                    if (q.qsize() > fps):
                        q.get()
                if (q.qsize() == fps):
                    averageSpeed = sum(list(q.queue)) / fps
                    cv2.putText(resultImage, str(int(averageSpeed)) + " km/hr", (int(x2 + w2 / 2), int(y2 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                carLocation[carID] = [x2, y2, w2, h2]

        cv2.imshow('Drawn Frame', resultImage)

        # Write the frame into the file 'output.avi'
        out.write(resultImage)

        if cv2.waitKey(33) == 27:
            break

        frameCounter = frameCounter + 1

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    trackMultipleObjects()
