import numpy as np
import cv2
import sys 
from config import tflite

def run_sanity_check() : 
    print("---STEP 1 : SYSTEM SQMITY CHECK ---")
    
    #check python version
    print(f"[INFO] Python version : {sys.version.split()[0]}")
    #check opencv (technical word :library versioning)
    print (f"[INFO] OpenCV Version :{cv2.__version__}")
    #check AI Engine (technical word : Inference engine)
    #we xheck if the interpreter clqss is available
    try : 
        test_interpreter = tflite.Interpreter 
        print ("[SUCCESS] AI Inference Engine : READY")
    except Exception as e :
        print (f"[ERROR] AI Inference Engine : FAILED -> {e}")
        return
    
    #check hardware (device initialization)
    print ("[INFO] Initializing Camera ...")
    cap = cv2.VideoCapture(0) #0 defqult webcam ID
    if cap.isOpened():
        print("[SUCCESS] Hqrdware : Camera DETECTED")
    #read one frame to confirm data flow (frame buffer)
        ret, frame = cap.read()
        if ret:
            print("[INFO] Camera Data Flow: OK")
    #show a window for 3 secs to see image
            cv2.imshow("Sanity check - camera stream", frame)
            cv2.waitKey(3000)
            cap.release()
            cv2.destroyAllWindows()
            print("[INFO] Camera released successfully.")
        else :
            print("[ERROR] Hardware : camera not found or busy")
        print("---CHECK COMPLETED---")
if __name__=="__main__":
    run_sanity_check()        