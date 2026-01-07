import serial
import csv
import time
import re
from datetime import datetime

def parse_sensor_data(line):
    # Parse format: F1:694 F2:1023 F3:266 F4:539 F5:426 | AX:-4792 AY:720 AZ:16156 | GX:-177 GY:59 GZ:12
    flex_pattern = r'F(\d):(\d+)'
    accel_pattern = r'A([XYZ]):(-?\d+)'
    gyro_pattern = r'G([XYZ]):(-?\d+)'
    
    flex_data = {}
    accel_data = {}
    gyro_data = {}
    
    # Extract flex sensor values
    for match in re.finditer(flex_pattern, line):
        flex_data[f'F{match.group(1)}'] = int(match.group(2))
    
    # Extract accelerometer values
    for match in re.finditer(accel_pattern, line):
        accel_data[f'A{match.group(1)}'] = int(match.group(2))
    
    # Extract gyroscope values
    for match in re.finditer(gyro_pattern, line):
        gyro_data[f'G{match.group(1)}'] = int(match.group(2))
    
    return flex_data, accel_data, gyro_data

def read_arduino_data(port='/dev/cu.usbserial-*', baudrate=9600):
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)
        
        print("Reading data... Press Ctrl+C to stop")
        
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                if line and 'F1:' in line:
                    flex_data, accel_data, gyro_data = parse_sensor_data(line)
                    
                    if len(flex_data) == 5 and len(accel_data) == 3 and len(gyro_data) == 3:
                        print(f"FLEX: F1:{flex_data.get('F1')} F2:{flex_data.get('F2')} F3:{flex_data.get('F3')} F4:{flex_data.get('F4')} F5:{flex_data.get('F5')} | ACCEL: {accel_data} | GYRO: {gyro_data}")
        
        ser.close()
        
    except KeyboardInterrupt:
        print("\nData collection stopped by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    read_arduino_data(port='/dev/cu.usbserial-A5069RR4')