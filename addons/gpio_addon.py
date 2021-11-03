import RPi.GPIO as GPIO
import time
import threading
import logging 

class AddonGPIO:

    def __init__(self, pin = 17, active_state = True):
        self.is_active = False
        self.pin = pin
        self.active_state = active_state
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT)

    def control_gpio(self):

        if not self.is_active:
            self.is_active = True

            thread = threading.Thread(target=self.write_to_pin)
            thread.start()
         
    def write_to_pin(self, duration = 5):

        start_time = time.time()

        logging.info("GPIO activated")
        while (time.time() - start_time) < duration:
            GPIO.output(self.pin, self.active_state)
        logging.info("GPIO deactivated")

        self.is_active = False
        GPIO.output(self.pin, not self.active_state)