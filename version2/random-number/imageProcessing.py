import cv2
import subprocess
import numpy as np
import hashlib
import base64

# Ping raspberrypi.local to get the IP address
result = subprocess.run(['ping', '-c', '1', 'raspberrypi.local'], stdout=subprocess.PIPE)

ip = ''

if result.returncode == 0:
    print("Ping successful")
    ip = result.stdout.decode('utf-8').split()[2][1:-2]
    print(ip)
else:
    print("Ping failed")
    exit()

# The URL of the MJPEG stream
stream_url = f'http://{ip}:8000/stream.mjpg'
print(stream_url)

# Open the video stream
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Set the precision and threshold
precision = 10
threshold = 200
component_size = 60

def generate_random_number(x_array, y_array, width_array, height_array, seed, line_length_array):
    # Initialize a secure hash object with SHA-512 algorithm
    hash_obj = hashlib.sha512()

    # Incorporate all input arrays into the hash
    # Convert numpy types to native int types before converting to bytes
    for x, y, width, height, line_length in zip(x_array, y_array, width_array, height_array, line_length_array):
        hash_obj.update(int(x).to_bytes(4, 'little'))
        hash_obj.update(int(y).to_bytes(4, 'little'))
        hash_obj.update(int(width).to_bytes(4, 'little'))
        hash_obj.update(int(height).to_bytes(4, 'little'))
        hash_obj.update(int(line_length).to_bytes(4, 'little'))

    # Include the seed in the hash computation, converting from numpy int if necessary
    hash_obj.update(int(seed).to_bytes(4, 'little'))

    # Finalize the hash computation and get the digest
    hash_bytes = hash_obj.digest()

    # Convert the hash bytes into an integer
    random_number = int.from_bytes(hash_bytes, 'little')

    return random_number

# Process the video stream
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 0)
    
    cv2.imshow('frame1', frame)

    if not ret:
        print("Error: Can't receive frame (maybe stream ended?). Sum'n wrong, exiting...")
        break

    # Converted the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    frame[thresh == 255] = [255, 255, 255]
    frame[thresh == 0] = [0, 0, 0]
    
    cv2.imshow('frame2', frame)

    # Find connected components in the image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
    

        if area > component_size:
            
            # Draw a rectangle around the connected component
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            perimeter = cv2.arcLength(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]), True)
            circularity = 4 * np.pi * area / perimeter ** 2
            cv2.putText(frame, f'{circularity:.2f}', (x + w, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

            x_array, y_array, width_array, height_array, line_length_array = [], [], [], [], []
            
            
            # Loop through the connected components to find the distance between them
            for j in range(1, num_labels):
                if i != j:
                    x2, y2, w2, h2, area2 = stats[j]
                    if area2 > component_size:
                        cv2.line(frame, (int(centroids[i][0]), int(centroids[i][1])), (int(centroids[j][0]), int(centroids[j][1])), (0, 0, 255), 1)
                        line_length = np.sqrt((centroids[i][0] - centroids[j][0]) ** 2 + (centroids[i][1] - centroids[j][1]) ** 2)
                        midpoint = ((int(centroids[i][0]) + int(centroids[j][0])) // 2, (int(centroids[i][1]) + int(centroids[j][1])) // 2)
                        angle = np.arctan2(centroids[j][1] - centroids[i][1], centroids[j][0] - centroids[i][0]) * 180 / np.pi
                        cv2.putText(frame, f'{line_length:.2f}', midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                        cv2.putText(frame, f'{angle:.2f}', (midpoint[0], midpoint[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

                        x_array.append(int(centroids[i][0]))
                        y_array.append(int(centroids[i][1]))
                        width_array.append(w)
                        height_array.append(h)
                        line_length_array.append(int(line_length))
                        
            # Draw a circle at the centroid of the connected component
            cv2.circle(frame, (int(centroids[i][0]), int(centroids[i][1])), 5, (0, 0, 255), -1)
            cv2.putText(frame, f'({int(centroids[i][0])}, {int(centroids[i][1])})', (int(centroids[i][0]), int(centroids[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            cv2.putText(frame, f'{w}x{h}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

            print(f"Random number: {generate_random_number(x_array, y_array, width_array, height_array, 0, line_length_array)}")

    cv2.imshow('frame3', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
