from opcua import Client
import json
import time

client = Client("opc.tcp://192.168.88.166:12345") # Real PC 192.168.88.166:12345
client.connect()

readings = []

start_time = time.time()

# missing temp_u, temp_u_p, temp_u_i,temp_u_d (Control temperatures base + PID)

while time.time() - start_time < 10.0:
    reads = {
        "timestamp": time.time(),
        "temp": client.get_node("ns=2;s=Testa chamber.temp").get_value(),
        "temp_raw": client.get_node("ns=2;s=Testa chamber.temp_raw").get_value(),
        "state": client.get_node("ns=2;s=Testa chamber.state").get_value(),
        "pressure": client.get_node("ns=2;s=Testa chamber.pres").get_value(),
        "con": client.get_node("ns=2;s=Testa chamber.temp_u").get_value(),
        "con_p": client.get_node("ns=2;s=Testa chamber.temp_u_p").get_value(),
        "con_i": client.get_node("ns=2;s=Testa chamber.temp_u_i").get_value(),
        "con_d": client.get_node("ns=2;s=Testa chamber.temp_u_d").get_value(),
    }
    readings.append(reads)
    time.sleep(0.5)

json_data = json.dumps(readings, indent=2)
print(json_data)

with open("opcua_readings2.json", "w") as f:
    json.dump(readings, f, indent=2)

#client.get_node("ns=2;s=Testa chamber.temp").set_value(118.8)
#final_temp = client.get_node("ns=2;s=Testa chamber.temp").get_value()

#print("Final temp " + str(final_temp))

client.disconnect()


#temp = client.get_node("ns=2;s=temp").get_value()
#temp_raw = client.get_node("ns=2;s=temp_raw").get_value()
# chamber = 200 ms < 500 ms
#client.get_node("ns=2;s=temp1").set_value(35.0)
