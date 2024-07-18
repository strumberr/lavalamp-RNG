# Generating random values from a lava lamp

This project showcases how to use a Raspberry Pi and a computer on the same local network to generate random numbers based on the blobs of a lava lamp. 

## Requirements

- A Raspberry Pi with a camera module
- A computer running on the same network as the Raspberry Pi
- Python 3.x installed on both devices
- OpenCV installed on the computer

```sh
pip install picamera2
pip install opencv-python
```

## Raspberry Pi Setup

Run the following Python script on your Raspberry Pi to stream video from the camera module. The script sets up an HTTP server to serve the MJPEG stream.


[Raspberry Pi Code](https://github.com/strumberr/lavalamp-RNG/blob/main/raspberrypi/main.py)

## Computer Setup

Run the provided Python script on your computer to connect to the Raspberry Pi's video stream, process the video frames, and generate random numbers based on the lava lamp's visuals.

[Computer Code](https://github.com/strumberr/lavalamp-RNG/blob/main/version2/random-string/imageProcessing.py)


## Usage

1. Ensure both the Raspberry Pi and the computer are connected to the same local network.
2. Run the Raspberry Pi script first to start the video stream.
3. Run the computer script to connect to the video stream and process the frames.
4. The computer script will display the video frames, process them, and generate random numbers.

## Note

- The Raspberry Pi script will print the IP address of the device. Use this IP address in the computer script to connect to the video stream.
- The computer script uses OpenCV to process the video frames, so make sure OpenCV is installed.
