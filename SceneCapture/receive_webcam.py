"""
receive_webcam.py

This script runs on a receiving computer (e.g., a Windows laptop).
It listens for incoming webcam frame data from a Raspberry Pi over the network
and displays the frames in real-time using OpenCV.

Setup Instructions:
1. Ensure the IP and port used match what the Raspberry Pi sender script is using.
2. Run this script on the Windows laptop first before starting the sender script.
3. Press 'q' or Ctrl+C to exit the stream and close the connection.
"""

import cv2
import socket
import struct
import pickle

# Host settings (0.0.0.0 means accept connections on any network interface)
host_ip = '0.0.0.0'
port = 9999

# Create a TCP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the host and port
server_socket.bind((host_ip, port))

# Start listening for incoming connections
server_socket.listen(1)
print(f"[INFO] Waiting for connection on {host_ip}:{port}...")

# Accept the incoming connection from the Raspberry Pi
conn, addr = server_socket.accept()
print(f"[INFO] Connection established with: {addr}")

data = b''
payload_size = struct.calcsize(">L")  # Size of length header

try:
    while True:
        # Ensure we have enough data to read the length of the frame
        while len(data) < payload_size:
            packet = conn.recv(4096)
            if not packet:
                raise ConnectionError("Socket closed")
            data += packet

        # Unpack the size of the incoming frame
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]

        # Wait until we receive the full frame
        while len(data) < msg_size:
            data += conn.recv(4096)

        # Extract and deserialize the frame data
        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = pickle.loads(frame_data)

        # Display the frame
        cv2.imshow("Webcam Stream", frame)

        # Press 'q' to exit the stream
        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Stream interrupted by user.")

except Exception as e:
    print(f"[ERROR] {e}")

finally:
    # Clean up resources
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()
    print("[INFO] Server shutdown.")
