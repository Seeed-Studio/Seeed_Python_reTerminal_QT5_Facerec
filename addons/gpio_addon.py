from gpiozero import DigitalOutputDevice
import time
import threading
import logging 

class AddonGPIO:

    def __init__(self, pin = 16, active_state = True):
        self.is_active = False
        self.pin = pin
        self.active_state = active_state
        self.device = DigitalOutputDevice(pin=pin, active_high=active_state)

    def __del__(self):
        try:
            self.device.off()
            self.device.close()
        except AttributeError:
            pass

    def activate_gpio(self):

        if not self.is_active:
            self.is_active = True

            thread = threading.Thread(target=self.write_to_pin)
            thread.start()
         
    def write_to_pin(self, duration = 5):

        start_time = time.time()

        logging.info("GPIO activated")
        self.device.on()
        while (time.time() - start_time) < duration:
            time.sleep(0.001)
        logging.info("GPIO deactivated")

        self.is_active = False
        self.device.off()

if __name__ == "__main__":
    GPIO = AddonGPIO(16, True)
    GPIO.activate_gpio()
    print("kk")