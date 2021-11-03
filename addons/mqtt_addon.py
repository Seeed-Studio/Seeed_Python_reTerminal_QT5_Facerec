import paho.mqtt.client as mqtt
import time 
import logging

class AddonMQTT:

    def __init__(self, address, port) -> None:
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(address, port, 5)
        self.client.loop_start()
        logging.info("MQTT Init")

    def __delattr__(self, name: str) -> None:
        self.client.loop_stop()        
        self.client.disconnect()
        logging.info("MQTT deinit")

    def on_connect(self, client, userdata, flags, rc):
        logging.info("Connected to with result code "+str(rc))
        client.subscribe("face_rec/verified")

    def on_message(self, client, userdata, msg):
        logging.info(msg.topic+" " + str(msg.payload))

    def send_data(self, topic, msg):
        self.client.publish(topic, msg)

if __name__ == "__main__":
    MQTT = AddonMQTT("localhost", 1883)

    i = 0
    try:
        while True:
            time.sleep(0.5)
            MQTT.send_data("mqtt/pimylifeup", i)
            i += 1
    except KeyboardInterrupt:
        del MQTT