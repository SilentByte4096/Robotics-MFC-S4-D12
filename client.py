import socket
import json
import cv2
import threading
import queue
import sys
import time
import os
import datetime

# Check if Pi's IP address is provided as a command-line argument
if len(sys.argv) != 2:
    print("Usage: python client.py 192.168.58.189")
    sys.exit(1)

# Configuration
PI_IP = sys.argv[1]  # Get Pi's IP from command line
PI_PORT = 65432      # Port the Pi's socket server is listening on
MJPG_URL = f'http://192.168.58.189:8080/?action=stream'  # Adjust port/path based on your MJPG server setup

# Shared resources
stop_event = threading.Event()  # Event to signal threads to stop
recording_event = threading.Event()  # Event to control recording state
feedback_queue = queue.Queue()  # Queue for user feedback (not used in this version but kept for potential expansion)

def display_video():
    """Thread function to display the live MJPG video feed and handle recording."""
    cap = cv2.VideoCapture(MJPG_URL)
    if not cap.isOpened():
        print("Failed to open video stream")
        stop_event.set()
        return
    
    # Read the first frame to get frame size
    ret, frame = cap.read()
    if not ret:
        print("Failed to get initial frame")
        stop_event.set()
        return
    height, width = frame.shape[:2]
    video_writer = None
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            # Handle recording state transitions
            if recording_event.is_set() and video_writer is None:
                # Start new recording
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join("recordings", f"recording_{timestamp}.avi")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(filename, fourcc, 30, (width, height))
            elif not recording_event.is_set() and video_writer is not None:
                # Stop current recording
                video_writer.release()
                video_writer = None
            
            # Write frame if recording
            if recording_event.is_set() and video_writer is not None:
                video_writer.write(frame)
            
            # Display the frame
            cv2.imshow('Video Feed', frame)
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
        else:
            print("Failed to get frame")
            time.sleep(0.1)  # Brief delay to avoid busy-waiting
    
    # Release resources
    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

def handle_socket():
    """Thread function to handle socket communication with the Pi."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((PI_IP, PI_PORT))
        print(f"Connected to Pi at {PI_IP}:{PI_PORT}")
    except Exception as e:
        print(f"Failed to connect to Pi: {e}")
        stop_event.set()
        return
    
    buffer = ""
    while not stop_event.is_set():
        try:
            data = s.recv(1024).decode()
            if not data:
                break
            
            buffer += data
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    msg = json.loads(line)
                    if msg["type"] == "start_sequence":
                        recording_event.set()
                        print("Start recording")
                    elif msg["type"] == "end_sequence":
                        recording_event.clear()
                        print("Stop recording")
                        stop_event.set()
                        break
                    elif msg["type"] == "sensor":
                        # Log raw sensor data
                        with open("sensor_data.txt", "a") as f:
                            f.write(json.dumps(msg["data"]) + "\n")
                    elif msg["type"] == "turn_complete":
                        # Prompt user for feedback
                        angle = msg["angle"]
                        print(f"Turn completed: {angle:.2f} degrees")
                        feedback = input("Was the turn successful? (y/n): ").strip().lower()
                        if feedback == "y" or feedback == "n":
                            s.sendall(feedback.encode() + b'\n')
                        else:
                            print("Invalid input. Sending 'n' by default.")
                            s.sendall(b'n\n')
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                except Exception as e:
                    print(f"Error processing message: {e}")
                    
        except Exception as e:
            print(f"Socket error: {e}")
            break
    
    s.close()

def main():
    """Main function to start threads and handle program flow."""
    # Create recordings directory if it doesn’t exist
    os.makedirs("recordings", exist_ok=True)
    
    # Start video display thread
    video_thread = threading.Thread(target=display_video)
    video_thread.start()
    
    # Start socket communication thread
    socket_thread = threading.Thread(target=handle_socket)
    socket_thread.start()
    
    # Wait for threads to finish
    video_thread.join()
    socket_thread.join()
    print("Program terminated")

if __name__ == "__main__":
    main()