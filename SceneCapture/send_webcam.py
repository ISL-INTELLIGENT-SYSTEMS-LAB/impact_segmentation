"""
send_webcam.py

This script runs on a Raspberry Pi and captures webcam frames using OpenCV.
It then sends the frames over a socket connection to a receiver computer (e.g., a Windows laptop).

Setup Instructions:
1. Replace `server_ip` with the IP address of the receiving computer.
2. Ensure both the Raspberry Pi and the receiving computer are on the same local network.
3. Run the `receive_webcam.py` script on the receiving machine first.
4. Then run this script on the Raspberry Pi.
5. Press Ctrl+C to stop the script.
"""

import cv2
import socket
import struct
import pickle

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set frame width
cap.set(4, 480)  # Set frame height

# Set the IP and port of the receiving computer
server_ip = input("Please enter the IP address shown on the pop-up window: ")

server_port = 9999

# Create a TCP socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the server
client_socket.connect((server_ip, server_port))

print('\n[INFO] Connection made, sending video feed.')

try:
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Serialize the frame using pickle
        data = pickle.dumps(frame)

        # Pack the size of the frame first (as a 4-byte big-endian integer)
        size = struct.pack(">L", len(data))

        # Send the size followed by the serialized frame data
        client_socket.sendall(size + data)

except KeyboardInterrupt:
    print("\n[INFO] Transmission stopped by user.")

finally:
    # Release resources
    cap.release()
    client_socket.close()
    print("[INFO] Connection closed.")
