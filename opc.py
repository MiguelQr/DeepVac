from opcua import Client
import json
import time

client = Client("opc.tcp://127.0.0.1:12345") # Real PC 192.168.88.166:12345
client.connect()

readings = []

start_time = time.time()


while time.time() - start_time < 5.0:
    reads = {
        "timestamp": time.time(),
        "temp": client.get_node("ns=2;s=temp").get_value(),
        "temp_raw": client.get_node("ns=2;s=temp_raw").get_value(),
    }
    readings.append(reads)
    time.sleep(0.5)

json_data = json.dumps(readings, indent=2)
print(json_data)

with open("opcua_readings.json", "w") as f:
    json.dump(readings, f, indent=2)

client.disconnect()


#temp = client.get_node("ns=2;s=temp").get_value()
#temp_raw = client.get_node("ns=2;s=temp_raw").get_value()

#print("temp:", temp)
#print("temp_raw:", temp_raw)

# chamber = 200 ms < 500 ms
#client.get_node("ns=2;s=temp1").set_value(35.0)
