import cv2
import dlib
import time
import threading
import math
import imutils
import numpy as np
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours

carCascade = cv2.CascadeClassifier('../Classifier/myhaar.xml')
lineCascade = cv2.CascadeClassifier('../Classifier/linecascade.xml')
video = cv2.VideoCapture('../Videos/default.mp4')

WIDTH = 1280
HEIGHT = 720

def estimateSpeed(location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    # ppm = location2[2] / carWidht
    ppm = 8.8
    d_meters = d_pixels / ppm
    # print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
    fps = video.get(cv2.CAP_PROP_FPS)
    speed = d_meters * fps * 3.6
    return speed

def estimateSpeed2(location1, location2, lineTracker, image, cnts):
    d_pixels = distanceFormula(location1[0], location1[1], location2[0], location2[1])
    closestLine = findClosestLine(lineTracker, location2)
    #find closest line ppm
    ppm = getPixelPerMetric(closestLine, cnts, 3)
    if (ppm == None):
        return None

    d_meters = d_pixels / ppm
    fps = video.get(cv2.CAP_PROP_FPS)
    speed = d_meters * fps * 3.6
    # print("ppm: ", ppm, "| d_mtrs: ", d_meters, "| spd: ", speed)
    return speed

def distanceFormula(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def getTrafficLines(image):
    lineTracker = {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lines = lineCascade.detectMultiScale(gray, 1.1, 3, 10, (1, 1),(5,5))
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
    closest = 10000000000000
    closestLine = None
    carX = carLocation[0]
    carY = carLocation[1] + carLocation[3]
    for lineID in lineTracker.keys():
        lineXCenter = lineTracker[lineID][0] + lineTracker[lineID][2] / 2
        lineYCenter = lineTracker[lineID][1] + lineTracker[lineID][3] / 2
        distance = distanceFormula(carX, carY, lineXCenter, lineYCenter)
        if (distance < closest):
            closest = distance
            closestLine = lineTracker[lineID]
    return closestLine

def testLines(image, resultImage):
    lines = getTrafficLines(image)
    for lineID in lines.keys():
        t_x = lines[lineID][0]
        t_y = lines[lineID][1]
        t_w = lines[lineID][2]
        t_h = lines[lineID][3]
        cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), (255, 255, 255), 4)

def getPixelPerMetric(line, cnts, metric):
    # find contour belonging to line param
    lx, ly, lw, lh = line
    lineContour = None
    for c in cnts:
        cx, cy, cw, ch = cv2.boundingRect(c)
        if (lx < cx and ly < cy and (lx+lw) > (cx+cw) and (ly+lh) > (cy+ch)):
            lineContour = cv2.boundingRect(c)
            break

    if (lineContour == None):
        return None
    else:
        # print("Match: ", line, lineContour)
        x, y, w, h = lineContour
        pixelsPerMetric = distanceFormula(x, y, x+w, y+h) / metric
        return pixelsPerMetric

def getContourMap(image):
    # convert to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 25, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    #  show edge map
    # cv2.imshow('Edges', edged)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts

def trackMultipleObjects():
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0
    fps = 0

    carTracker = {}
    carNumbers = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 1000

    cnts = None

    # Write output to video file
    out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (WIDTH, HEIGHT))

    while True:
        start_time = time.time()
        rc, image = video.read()
        if type(image) == type(None):
            break

        image = cv2.resize(image, (WIDTH, HEIGHT))
        resultImage = image.copy()

        if frameCounter == 0:
            cnts = getContourMap(image)

        frameCounter = frameCounter + 1

        carIDtoDelete = []

        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(image)

            if trackingQuality < 7:
                carIDtoDelete.append(carID)

        for carID in carIDtoDelete:
            print('Removing carID ' + str(carID) + ' from list of trackers.')
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)

        if not (frameCounter % 10):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (20, 20))

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
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))

                    carTracker[currentCarID] = tracker
                    carLocation1[currentCarID] = [x, y, w, h]

                    currentCarID = currentCarID + 1

        # cv2.line(resultImage,(0,480),(1280,480),(255,0,0),5)

        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()

            t_x = int(trackedPosition.left())
            t_y = int(trackedPosition.top())
            t_w = int(trackedPosition.width())
            t_h = int(trackedPosition.height())

            cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)

            # speed estimation
            carLocation2[carID] = [t_x, t_y, t_w, t_h]

        end_time = time.time()

        if not (end_time == start_time):
            fps = 1.0 / (end_time - start_time)

        # cv2.putText(resultImage, 'FPS: ' + str(int(fps)), (620, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # testing traffic line classifier
        # testLines(image, resultImage)
        lineTracker = getTrafficLines(image)

        for i in carLocation1.keys():
            if frameCounter % 1 == 0:
                [x1, y1, w1, h1] = carLocation1[i]
                [x2, y2, w2, h2] = carLocation2[i]

                # print 'previous location: ' + str(carLocation1[i]) + ', current location: ' + str(carLocation2[i])
                carLocation1[i] = [x2, y2, w2, h2]

                # print 'new previous location: ' + str(carLocation1[i])
                if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                    # speed[i] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])
                    speed[i] = estimateSpeed2([x1, y1, w1, h1], [x2, y2, w2, h2], lineTracker, image, cnts)

                    # if y1 > 275 and y1 < 285:
                    if speed[i] != None:
                        cv2.putText(resultImage, str(int(speed[i])) + " km/hr", (int(x1 + w1 / 2), int(y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                # print ('CarID ' + str(i) + ': speed is ' + str("%.2f" % round(speed[i], 0)) + ' km/h.\n')

                # else:
                #	cv2.putText(resultImage, "Far Object", (int(x1 + w1/2), int(y1)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # print ('CarID ' + str(i) + ' Location1: ' + str(carLocation1[i]) + ' Location2: ' + str(carLocation2[i]) + ' speed is ' + str("%.2f" % round(speed[i], 0)) + ' km/h.\n')
        cv2.imshow('result', resultImage)

        # Write the frame into the file 'output.avi'
        out.write(resultImage)

        if cv2.waitKey(33) == 27:
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    trackMultipleObjects()
