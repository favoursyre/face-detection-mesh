#I want to create a Face mesh module

#Useful libraries I would be working with -->
import numpy as np
import matplotlib.pyplot as plt 
import datetime
import time
import cv2 as cv
import mediapipe as mp


#Declaring the neccessary variables
class faceMesh():
    def __init__(self, staticImageMode = False, maxFaces = 1, refineLandmarks = False, minDetectionConfidence = 0.6, minTrackingConfidence = 0.6):
        self.staticImageMode = staticImageMode
        self.maxFaces = maxFaces
        self.refineLandmarks = refineLandmarks
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence = minTrackingConfidence

        #We are instantiating the objects of hand tracking functions
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(staticImageMode, maxFaces, refineLandmarks, minDetectionConfidence, minTrackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness = 1, circle_radius = 2)

    #This function handles the various pose landmarks in the frame
    def findFaceMesh(self, image, draw = True):
        rgbImage = cv.cvtColor(image, cv.COLOR_BGR2RGB) #We are converting the format because mediapipe works with RGB format
        #global results
        results = self.faceMesh.process(rgbImage) #This gets the various points in the hands
        #print(f"Results: {results}")
        #print(f"Result landmarks: {results.multi_hand_landmarks} \n")
        faces = []
        if results.multi_face_landmarks:
            #poseLms = results.pose_landmarks
            for faceLms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = image.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    #cv.putText(image, f"{id}", (x, y), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2) 
                    face.append([x, y])
                faces.append(face)
        else:
            pass
        return image, faces

#This handles the main function
def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)
    detector = faceMesh(maxFaces = 7)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame, faces = detector.findFaceMesh(frame)

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


if __name__ == "__main__":
    print("FACE MESH MODULE\n")

    main()

    print("\nExecuted successfully!")