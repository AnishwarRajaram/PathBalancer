import socket
import struct
import threading
import numpy as np
import time
import signal
import sys
import onnxruntime as ort

# --- CONFIG ---
UDP_IP = "0.0.0.0" 
UDP_PORT = 2368    
GRID_RES = 0.25
GRID_SIZE = 100
PIXEL_COUNT = int(GRID_SIZE / GRID_RES) # 400
LOG_FILE = "driving_log.txt"

# Thread Control
stop_event = threading.Event()

# Shared memory for threads
live_bev_grid = np.zeros((PIXEL_COUNT, PIXEL_COUNT, 4), dtype=np.float32)
grid_lock = threading.Lock()

def signal_handler(sig, frame):
    print("\nStopping Modularizer gracefully...")
    stop_event.set()

def parse_lidar_packet(data):
    points = []
    try:
        # Simplified parser for VLP-16 style packets
        for i in range(12):
            block_offset = i * 100
            azimuth = struct.unpack("<H", data[block_offset+2:block_offset+4])[0] / 100.0
            az_rad = np.deg2rad(azimuth)
            for j in range(32):
                point_offset = block_offset + 4 + (j * 3)
                distance = struct.unpack("<H", data[point_offset:point_offset+2])[0] * 0.002
                intensity = data[point_offset+2]
                if distance > 1.0:
                    x = distance * np.cos(az_rad)
                    y = distance * np.sin(az_rad)
                    z = 0 # Vertical angle parsing can be added based on laser ID j
                    points.append((x, y, z, intensity))
    except Exception:
        pass 
    return points

def udp_listener():
    global live_bev_grid
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(1.0) 
    
    print(f"Modularizer: Listening for LiDAR on port {UDP_PORT}...")
    
    while not stop_event.is_set():
        try:
            data, addr = sock.recvfrom(2048) 
            points = parse_lidar_packet(data) 
            with grid_lock:
                for x, y, z, intensity in points:
                    px = int((-y / GRID_RES) + (PIXEL_COUNT / 2))
                    py = int((-x / GRID_RES) + (PIXEL_COUNT / 2))
                    if 0 <= px < PIXEL_COUNT and 0 <= py < PIXEL_COUNT:
                        # Channel 0: Height
                        live_bev_grid[py, px, 0] = np.clip((z + 2.0) / 5.0, 0, 1)
                        # Channel 1: Intensity
                        live_bev_grid[py, px, 1] = intensity / 255.0
                        # Channel 2: Density (Accumulate)
                        live_bev_grid[py, px, 2] = min(live_bev_grid[py, px, 2] + 0.1, 1.0)
                        # Channel 3 (Roughness) would be calculated via variance of Z if needed
        except socket.timeout:
            continue
    sock.close()
    print("UDP Listener closed.")

def log_driving_decision(preds):
    """
    Analyzes the predicted segmentation map and logs the result.
    Assumes Class IDs: 0: Background/Road, 1: Vehicle, 2: Pedestrian, 3: Obstacle
    """
    center = PIXEL_COUNT // 2
    # Define ROI: 4 meters wide (16 pixels) and 10 meters forward (40 pixels)
    # Adjust indexing based on your coordinate mapping
    roi_width = 8 
    roi_depth = 40
    front_roi = preds[center - roi_width : center + roi_width, center : center + roi_depth]
    
    # Count detections in the critical path
    v_pixels = np.sum(front_roi == 1)
    p_pixels = np.sum(front_roi == 2)
    o_pixels = np.sum(front_roi == 3)
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Simple Threshold Logic
    decision = "PROCEED"
    if p_pixels > 2:
        decision = "EMERGENCY_BRAKE_PEDESTRIAN"
    elif v_pixels > 10 or o_pixels > 10:
        decision = "STOP_OBSTACLE_AHEAD"

    log_entry = f"[{timestamp}] Decision: {decision} | V_px: {v_pixels} | P_px: {p_pixels} | O_px: {o_pixels}\n"
    
    try:
        with open(LOG_FILE, "a") as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Logging error: {e}")

def ai_brain():
    providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    try:
        session = ort.InferenceSession("unet_mobile_v1_weights.onnx", providers=providers)
    except Exception as e:
        print(f"Failed to load ONNX model: {e}")
        return

    input_name = session.get_inputs()[0].name
    print(f"AI Brain (ONNX): Online. Decisions being logged to {LOG_FILE}.")
    
    while not stop_event.is_set():
        with grid_lock:
            # Create a copy for the forward pass
            input_grid = live_bev_grid.copy()
            # Fade the density channel to handle moving objects
            live_bev_grid[:, :, 2] *= 0.90 
        
        # Pre-process for ONNX: (1, 4, 400, 400)
        input_data = np.transpose(input_grid, (2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
        
        # Inference
        outputs = session.run(None, {input_name: input_data})
        preds = np.argmax(outputs[0], axis=1).squeeze()
        
        # Log decision
        log_driving_decision(preds)
        
        # Frequency Control (~30-50 FPS)
        time.sleep(0.01)
        
    print("AI Brain stopped.")

if __name__ == "__main__":
    # Clear or initialize log file
    with open(LOG_FILE, "w") as f:
        f.write("--- MONSTER TRUCK SESSION START ---\n")
        
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start LiDAR Listener in background
    t1 = threading.Thread(target=udp_listener)
    t1.start()
    
    # Run AI in foreground
    ai_brain()
    
    # Clean up
    t1.join()
    print("Modularizer exit complete.")