import serial
import time
import sys

def monitor(port="COM5", baud=115200):
    try:
        ser = serial.Serial(port, baud, timeout=1.0)
        ser.setDTR(False)
        time.sleep(0.1)
        ser.setDTR(True)
        time.sleep(1)
        print(f"Monitoring {port} @ {baud}...")
        
        start = time.time()
        while time.time() - start < 8:
            line = getattr(ser, "readline", lambda: b"")()
            if line:
                print(line.decode('utf-8', errors='replace').strip())
        ser.close()
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    monitor()
