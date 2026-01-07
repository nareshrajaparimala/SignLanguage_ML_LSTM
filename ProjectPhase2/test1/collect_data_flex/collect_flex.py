import serial
import time

# -------------------------------------------------
# 1. OPEN SERIAL PORT
# -------------------------------------------------
# Arduino Nano (CH340) usually appears as /dev/ttyUSB0
# Arduino Original usually appears as /dev/ttyACM0

PORT = "/dev/ttyUSB0"     # Change to /dev/ttyACM0 if needed
BAUD = 9600               # Must match Arduino Serial.begin(9600)

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)  # Wait for Arduino reset
    print("\n📡 Listening to Arduino data...\n")
except Exception as e:
    print("❌ ERROR: Cannot open serial port:", e)
    exit()


# -------------------------------------------------
# 2. CONTINUOUSLY READ INCOMING DATA
# -------------------------------------------------
while True:
    try:
        # Read one full line
        line = ser.readline().decode("utf-8", errors="ignore").strip()

        # If line is not empty → print it
        if line:
            print("🔹 Received:", line)

    except KeyboardInterrupt:
        print("\n❗ Stopped by user")
        break

    except Exception as e:
        print("❌ Serial read error:", e)
        break