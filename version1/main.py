import cv2
import subprocess

# ping raspberrypi.local using the command "ping raspberrypi.local" in the terminal use subrocess to run the command

#send only one ping request
result = subprocess.run(['ping', '-c', '1', 'raspberrypi.local'], stdout=subprocess.PIPE)

ip = ''

# check if the ping was successful
if result.returncode == 0:
    print("ping successful")
    
    # get the ip address of the raspberry pi
    ip = result.stdout.decode('utf-8').split()[2][1:-2]
    print(ip)
else:
    print("ping failed")
    exit()
    


# The URL of the MJPEG stream
stream_url = f'http://{ip}:8000/stream.mjpg'

# Open the video stream
cap = cv2.VideoCapture(stream_url)

# Check if the stream was opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# set the number of squares (the lower the number, the higher the precision)
precision = 10

#color precision/threshold
threshold = 180

#while loop to read frames from the stream
while True:

    # read a frame from the stream
    ret, frame = cap.read()
    
    #flip it 180 degrees
    frame = cv2.flip(frame, 0)

    # check if the frame was read successfully
    if not ret:
        print("Error: Can't receive frame (stream end?) sum'n wrong. Exiting ...")
        break
    
    # convert the frame to grayscale for better processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # cv2.imshow('frame2', gray)

    # get the minimum and maximum pixel values in the frame to use for thresholding
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)

    # threshold the frame to get a binary image
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    frame[thresh == 255] = [255, 255, 255]

    frame[thresh == 0] = [0, 0, 0]
    
    # cv2.imshow('frameProcess', frame)

    # loop through the frame in steps of X (precision var) pixels
    for i in range(0, frame.shape[0], precision):

        for j in range(0, frame.shape[1], precision):

            # if the pixel value is 255, draw a rectangle around it
            if thresh[i, j] == 255:
                
                cv2.rectangle(frame, (j, i), (j+precision, i+precision), (0, 255, 0), 1)
                
                #draw a line from the left center of the screen to the middle of each green square
                
                # cv2.line(frame, (frame.shape[1], frame.shape[0]), (j+precision//2, i+precision//2), (0, 0, 255), 1)
                



    # loop through the frame in steps of X (precision var) pixels to count the number of white pixels in each row and column
    for i in range(0, frame.shape[0], precision):

        row_sum = 0
        for j in range(0, frame.shape[1], precision):
            if thresh[i, j] == 255:
                row_sum += 1

        if row_sum > 0:
            cv2.putText(frame, str(
                row_sum), (frame.shape[1]-50, i+precision), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # loop through the frame in steps of X (precision var) pixels to count the number of white pixels in each column
    for j in range(0, frame.shape[1], precision):

        col_sum = 0
        for i in range(0, frame.shape[0], precision):
            if thresh[i, j] == 255:
                col_sum += 1

        if col_sum > 0:
            cv2.putText(frame, str(
                col_sum), (j+precision, frame.shape[0]-precision), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


    final_number = ''

    # here we are concatenating the row and then sum to get the final number
    for i in range(0, frame.shape[0], precision):

        row_sum = 0
        for j in range(0, frame.shape[1], precision):
            if thresh[i, j] == 255:
                row_sum += 1

        if row_sum > 0:
            final_number += str(row_sum)
    
    # here we are concatenating the column and then sum to get the final number
    for j in range(0, frame.shape[1], precision):

        col_sum = 0
        for i in range(0, frame.shape[0], precision):
            if thresh[i, j] == 255:
                col_sum += 1

        if col_sum > 0:
            final_number += str(col_sum)
            


    print(final_number)

    cv2.putText(frame, final_number, (precision, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

