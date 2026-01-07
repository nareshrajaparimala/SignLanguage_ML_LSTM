import serial
import csv
import time
import re
from datetime import datetime

def parse_sensor_data(line):
    flex_pattern = r'F(\d):(\d+)'
    accel_pattern = r'A([XYZ]):(-?\d+)'
    gyro_pattern = r'G([XYZ]):(-?\d+)'
    
    flex_data = {}
    accel_data = {}
    gyro_data = {}
    
    for match in re.finditer(flex_pattern, line):
        flex_data[f'F{match.group(1)}'] = int(match.group(2))
    
    for match in re.finditer(accel_pattern, line):
        accel_data[f'A{match.group(1)}'] = int(match.group(2))
    
    for match in re.finditer(gyro_pattern, line):
        gyro_data[f'G{match.group(1)}'] = int(match.group(2))
    
    return flex_data, accel_data, gyro_data

def collect_gesture_data(gesture_name, port='/dev/cu.usbserial-A5069RR4', frames=50):
    ser = serial.Serial(port, 9600, timeout=1)
    time.sleep(2)
    
    filename = f"gesture_data_{gesture_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame', 'F1', 'F2', 'F3', 'F4', 'F5', 'AX', 'AY', 'AZ', 'GX', 'GY', 'GZ', 'Gesture'])
        
        print(f"Collecting {frames} frames for gesture: {gesture_name}")
        print("Press Enter when ready to start...")
        input()
        
        frame_count = 0
        while frame_count < frames:
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                except:
                    continue
                if line and 'F1:' in line:
                    flex_data, accel_data, gyro_data = parse_sensor_data(line)
                    
                    if len(flex_data) == 5 and len(accel_data) == 3 and len(gyro_data) == 3:
                        row = [frame_count,
                               flex_data.get('F1', 0), flex_data.get('F2', 0), flex_data.get('F3', 0),
                               flex_data.get('F4', 0), flex_data.get('F5', 0),
                               accel_data.get('AX', 0), accel_data.get('AY', 0), accel_data.get('AZ', 0),
                               gyro_data.get('GX', 0), gyro_data.get('GY', 0), gyro_data.get('GZ', 0),
                               gesture_name]
                        writer.writerow(row)
                        frame_count += 1
                        print(f"Frame {frame_count}/{frames} collected")
    
    ser.close()
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    while True:
        gesture_name = input("\nEnter gesture name (or 'quit' to exit): ").strip()
        
        if gesture_name.lower() == 'quit':
            break
            
        if gesture_name:
            print(f"\n=== Collecting data for {gesture_name} ===")
            collect_gesture_data(gesture_name)
            print("Data collection complete!")
        else:
            print("Please enter a valid gesture name.")
    
    print("Data collection finished.")