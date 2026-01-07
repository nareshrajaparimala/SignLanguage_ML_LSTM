import serial.tools.list_ports

def find_arduino_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if 'usbserial' in port.device or 'Arduino' in port.description:
            print(f"Found Arduino at: {port.device}")
            return port.device
    print("No Arduino found. Available ports:")
    for port in ports:
        print(f"  {port.device}: {port.description}")
    return None

if __name__ == "__main__":
    find_arduino_port()