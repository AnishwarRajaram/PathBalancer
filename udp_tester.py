import socket
import time
import struct
import numpy as np

UDP_IP = "127.0.0.1"
UDP_PORT = 2368
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("🚀 Simulating VLP-16 LiDAR stream... Start modularizer.py now!")

while True:
    # A standard packet has 12 blocks
    packet = bytearray()
    for block in range(12):
        # Header (0xFFEE) + Azimuth (e.g., 90 degrees * 100)
        packet += struct.pack("<HH", 0xEEFF, 9000) 
        
        # 32 points per block
        for point in range(32):
            # Distance (e.g., 5 meters * 500) + Intensity (200)
            distance = int(5.0 / 0.002) 
            intensity = 200
            # Pack distance (2 bytes) and intensity (1 byte)
            packet += struct.pack("<HB", distance, intensity)
            
    # Add status bytes to reach ~1206 bytes (standard LiDAR size)
    packet += b'\x00' * 6 
    
    sock.sendto(packet, (UDP_IP, UDP_PORT))
    time.sleep(0.1)