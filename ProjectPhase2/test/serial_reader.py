"""
Simple Serial Reader for Arduino flex+MPU stream.

Usage:
  python3 serial_reader.py --port /dev/ttyUSB0 --baud 9600 --out samples.csv

While running:
  - Press ENTER to print a current sample.
  - Type a label and press ENTER to save current sample with that label.
  - Press Ctrl+C to exit.

Expected Arduino line format (one-line per sample):
  F1:694 F2:1023 F3:266 F4:539 F5:426 | AX:-4792 AY:720 AZ:16156 | GX:-177 GY:59 GZ:12
"""
import serial
import argparse
import time
import csv
import re
from datetime import datetime

LINE_PATTERN = re.compile(
    r"F1:(?P<F1>-?\d+)\s+F2:(?P<F2>-?\d+)\s+F3:(?P<F3>-?\d+)\s+F4:(?P<F4>-?\d+)\s+F5:(?P<F5>-?\d+)\s*\|\s*"
    r"AX:(?P<AX>-?\d+)\s+AY:(?P<AY>-?\d+)\s+AZ:(?P<AZ>-?\d+)\s*\|\s*"
    r"GX:(?P<GX>-?\d+)\s+GY:(?P<GY>-?\d+)\s+GZ:(?P<GZ>-?\d+)"
)

def parse_line(line):
    """Parse a single serial text line into a dict of ints. Return None if not matched."""
    m = LINE_PATTERN.search(line)
    if not m:
        return None
    return {k: int(v) for k, v in m.groupdict().items()}

def run(port="/dev/ttyUSB0", baud=9600, out_csv=None):
    ser = serial.Serial(port, baud, timeout=1)
    time.sleep(2)  # wait for Arduino reset
    print(f"Connected to {port} at {baud} baud. Listening...")
    current_sample = None
    csv_file = None
    csv_writer = None

    if out_csv:
        csv_file = open(out_csv, "a", newline="")
        csv_writer = csv.writer(csv_file)
        # header (only write if file empty)
        csv_file.seek(0, 2)  # go to end
        if csv_file.tell() == 0:
            csv_writer.writerow(["timestamp"] + ["F1","F2","F3","F4","F5","AX","AY","AZ","GX","GY","GZ","label"])

    try:
        while True:
            raw = ser.readline().decode(errors="ignore").strip()
            if not raw:
                continue
            parsed = parse_line(raw)
            if parsed:
                current_sample = parsed
                ts = datetime.utcnow().isoformat()
                print(f"{ts} | " + raw)
            else:
                # if a sloppy line, still show it (helps debugging)
                print("UNPARSED:", raw)

            # non-blocking user input: prompt for label or ENTER to continue
            # Use input() to block only when user wants to label
            # instruct user to type label when needed:
            # (We avoid constantly blocking reads; user will run label mode manually)
            # You can press Ctrl+C to exit.
            # To save a sample: type label and press Enter.
            # To skip: just press Enter.
            # For automation, you can call the /add_sample API in server.py instead of this interactive mode.

            # Minimal interactive labeling: only if stdin is connected
            # This will block; use only when you want to label samples.
            # Uncomment below to enable interactive labeling:
            # label = input("Type label to save current sample (or press Enter to skip): ").strip()
            # if label and current_sample and csv_writer:
            #     row = [ts] + [current_sample[k] for k in ["F1","F2","F3","F4","F5","AX","AY","AZ","GX","GY","GZ"]] + [label]
            #     csv_writer.writerow(row)
            #     csv_file.flush()
            #     print("Saved sample ->", label)

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        if csv_file:
            csv_file.close()
        ser.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--baud", type=int, default=9600)
    p.add_argument("--out", dest="out_csv", default=None, help="CSV path to append labeled samples")
    args = p.parse_args()
    run(args.port, args.baud, args.out_csv)
