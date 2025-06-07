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
    print("Usage: python movement.py 192.168.58.189")
    sys.exit(1)

# Configuration
PI_IP = sys.argv[1]  # Get Pi's IP from command line
PI_PORT = 65432      # Port the Pi's socket server is listening on
MJPG_URL = f'http://192.168.58.189:8080/?action=stream'  # Adjust port/path based on your MJPG server setup

# Shared resources
command_queue = queue.Queue()  # Queue to pass commands to the socket thread
stop_event = threading.Event()  # Event to signal threads to stop
recording_event = threading.Event()  # Event to control recording state

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
    
    while not stop_event.is_set():
        try:
            # Get command from queue (non-blocking with timeout)
            cmd = command_queue.get(timeout=0.1)
            # Send JSON command to Pi
            s.sendall(json.dumps(cmd).encode() + b'\n')
            # Receive acknowledgment
            ack = s.recv(1024).decode().strip()
            if not ack:
                print("Connection closed by server")
                break
            elif ack != "ACK":
                print("Did not receive ACK")
        except queue.Empty:
            continue  # No command available, keep looping
        except Exception as e:
            print(f"Socket error: {e}")
            break
    
    s.close()

def main():
    """Main function to start threads and handle user input."""
    # Create recordings directory if it doesn’t exist
    os.makedirs("recordings", exist_ok=True)
    
    # Start video display thread
    video_thread = threading.Thread(target=display_video)
    video_thread.start()
    
    # Start socket communication thread
    socket_thread = threading.Thread(target=handle_socket)
    socket_thread.start()
    
    # Main loop for user input
    try:
        while not stop_event.is_set():
            user_input = input("Enter command (e.g., forward [speed], stop, start_record, stop_record, quit): ").strip().lower()
            parts = user_input.split()
            if not parts:
                continue
            
            action = parts[0]
            if action == "quit":
                stop_event.set()
                break
            elif action in ["forward", "backward", "left", "right"]:
                # Parse optional speed
                if len(parts) > 1:
                    try:
                        speed = int(parts[1])
                        cmd = {"action": action, "speed": speed}
                    except ValueError:
                        print("Invalid speed (must be an integer)")
                        continue
                else:
                    cmd = {"action": action}  # Use default speed on Pi
                command_queue.put(cmd)
            elif action == "stop":
                cmd = {"action": "stop"}
                command_queue.put(cmd)
            elif action == "start_record":
                if not recording_event.is_set():
                    recording_event.set()
                    print("Recording started")
                else:
                    print("Recording already in progress")
            elif action == "stop_record":
                if recording_event.is_set():
                    recording_event.clear()
                    print("Recording stopped")
                else:
                    print("Not currently recording")
            else:
                print("Invalid command. Use: forward [speed], backward [speed], left [speed], right [speed], stop, start_record, stop_record, quit")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        stop_event.set()
    except Exception as e:
        print(f"Error: {e}")
        stop_event.set()
    
    # Wait for threads to finish
    video_thread.join()
    socket_thread.join()
    print("Program terminated")

if __name__ == "__main__":
    main()