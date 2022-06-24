#I want to create a generic Face detection module

#Useful libraries that I would be working with -->
import numpy as np
import matplotlib.pyplot as plt 
import datetime
import time
import face_mesh as fm
import cv2 as cv
import mediapipe as mp
from threading import Thread


#Declaring the neccessary variables
class faceDetector():
    #Declaring the various arguments
    def __init__(self, minDetectionConfidence = 0.6, modelSelection = 1): 
        print("FACE DETECTION MODULE \n")

        self.minDetectionConfidence = minDetectionConfidence
        self.modelSelection = modelSelection

        #We are instantiating the objects of hand tracking functions
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(minDetectionConfidence, modelSelection)
        self.mpDraw = mp.solutions.drawing_utils

    #This function handles the various pose landmarks in the frame
    def findFace(self, image, draw = True):
        rgbImage = cv.cvtColor(image, cv.COLOR_BGR2RGB) #We are converting the format because mediapipe works with RGB format
        global results
        results = self.faceDetection.process(rgbImage) #This gets the various points in the hands
        #print(f"Results: {results}")
        #print(f"Result landmarks: {results.multi_hand_landmarks} \n")
        bboxs = []
        if results.detections:
            #poseLms = results.pose_landmarks
            for id, detection in enumerate(results.detections):
                bbox = detection.location_data.relative_bounding_box
                print(f"BBOX: {detection} ------------------------------------>")
                #print(f"Detection score {id}: {int(detection.score[0] * 100)}% ------------------------------------>")
                ih, iw, ic = image.shape
                bbox = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    image = self.borderDraw(image, bbox)

                cv.rectangle(image, bbox, (255, 0, 255), 2)
                cv.putText(image, f"Confidence: {int(detection.score[0] * 100)}%", (bbox[0], bbox[1] - 20), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2) 

        else:
            pass
        return image, bboxs

    #This function handles the thickening of detected squares borders
    def borderDraw(self, image, bbox, l = 30, thickness = 10):
        x, y, w, h = bbox
        x1, y1, = x + w, y + h
        cv.rectangle(image, bbox, (255, 0, 255), 2)

        #Declaring the various borders
        #Top left
        cv.line(image, (x, y), (x + l, y), (255, 0, 255), thickness)
        cv.line(image, (x, y), (x, y + l), (255, 0, 255), thickness)
        #Top right
        cv.line(image, (x1, y), (x1 - l, y), (255, 0, 255), thickness)
        cv.line(image, (x1, y), (x1, y + l), (255, 0, 255), thickness)
        #Bottom left
        cv.line(image, (x, y1), (x + l, y1), (255, 0, 255), thickness)
        cv.line(image, (x, y1), (x, y1 - l), (255, 0, 255), thickness)
        #Bottom right
        cv.line(image, (x1, y1), (x1 - l, y1), (255, 0, 255), thickness)
        cv.line(image, (x1, y1), (x1, y1 - l), (255, 0, 255), thickness)

        return image

#This handles the main function
def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)
    detector = faceDetector()

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame, bboxs = detector.findFace(frame)

            #Getting the frames per second
            cTime = time.perf_counter()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv.putText(frame, f"FPS: {int(fps)}", (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            cv.imshow("Frame", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()

#This threads both the face mesh and face detection script
def thread_face():
    t1 = Thread(target = main)
    t1.start()
    t2 = Thread(target = fm.main)
    t2.start()

    for t in [t1, t2]:
        t.join()

if __name__ == "__main__":
    thread_face()
