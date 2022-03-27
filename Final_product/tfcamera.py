from picamera import PiCamera
from time import sleep
def takepic():
    camera = PiCamera()
    camera.start_preview()
    sleep(2)
    camera.capture("/home/pi/CleanVision/TF/test.jpg")
    camera.stop_preview()
    camera.close()
