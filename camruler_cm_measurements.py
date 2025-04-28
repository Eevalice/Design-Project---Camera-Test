#-------------------------------
# imports
#-------------------------------

# builtins
import os, sys, time, traceback
from math import hypot

# must be installed using pip
# python3 -m pip install opencv-python
import numpy as np
import cv2

# local clayton libs
import frame_capture
import frame_draw

#-------------------------------
# default settings
#-------------------------------

# camera values
camera_id = 4
camera_width = 1920
camera_height = 1080
camera_frame_rate = 30
camera_fourcc = cv2.VideoWriter_fourcc(*"MJPG")

# auto measure mouse events
auto_percent = 0.2
auto_threshold = 127
auto_blur = 5

# normalization mouse events
norm_alpha = 0
norm_beta = 255

#-------------------------------
# read config file
#-------------------------------

# you can make a config file "camruler_config.csv"
# this is a comma-separated file with one "item,value" pair per line
# you can also use a "=" separated pair like "item=value"
# you can use # to comment a line
# the items must be named like the default variables above

# read local config values
configfile = 'camruler_config.csv'
if os.path.isfile(configfile):
    with open(configfile) as f:
        for line in f:
            line = line.strip()
            if line and line[0] != '#' and (',' in line or '=' in line):
                if ',' in line:
                    item, value = [x.strip() for x in line.split(',', 1)]
                elif '=' in line:
                    item, value = [x.strip() for x in line.split('=', 1)]
                else:
                    continue
                if item in 'camera_id camera_width camera_height camera_frame_rate camera_fourcc auto_percent auto_threshold auto_blur norm_alpha norm_beta'.split():
                    try:
                        exec(f'{item}={value}')
                        print('CONFIG:', (item, value))
                    except:
                        print('CONFIG ERROR:', (item, value))

#-------------------------------
# camera setup
#-------------------------------

# get camera id from argv[1]
# example "python3 camruler.py 2"
if len(sys.argv) > 1:
    camera_id = sys.argv[1]
    if camera_id.isdigit():
        camera_id = int(camera_id)

# camera thread setup
camera = frame_capture.Camera_Thread()
camera.camera_source = camera_id  # SET THE CORRECT CAMERA NUMBER
camera.camera_width = camera_width
camera.camera_height = camera_height
camera.camera_frame_rate = camera_frame_rate
camera.camera_fourcc = camera_fourcc

# Start camera thread
camera.start()

# Initial camera values (shortcuts for below)
width = camera.camera_width
height = camera.camera_height
area = width * height
cx = int(width / 2)
cy = int(height / 2)
dm = hypot(cx, cy)  # max pixel distance
frate = camera.camera_frame_rate
print('CAMERA:', [camera.camera_source, width, height, area, frate])

#-------------------------------
# frame drawing/text module
#-------------------------------

draw = frame_draw.DRAW()
draw.width = width
draw.height = height

#-------------------------------
# conversion (pixels to measure)
#-------------------------------

# distance units designator
unit_suffix = 'cm'  # Changed to cm

# calibrate every N pixels
pixel_base = 10

# maximum field of view from center to farthest edge
# should be measured in unit_suffix
cal_range = 72

# initial calibration values table {pixels:scale}
# this is based on the frame size and the cal_range
cal = dict([(x, cal_range / dm) for x in range(0, int(dm) + 1, pixel_base)])

# calibration loop values
# inside of main loop below
cal_base = 5
cal_last = None

# calibration update
def cal_update(x, y, unit_distance):
    # basics
    pixel_distance = hypot(x, y)
    scale = abs(unit_distance / pixel_distance)
    target = baseround(abs(pixel_distance), pixel_base)

    # low-high values in distance
    low = target * scale - (cal_base / 2)
    high = target * scale + (cal_base / 2)

    # get low start point in pixels
    start = target
    if unit_distance <= cal_base:
        start = 0
    else:
        while start * scale > low:
            start -= pixel_base

    # get high stop point in pixels
    stop = target
    if unit_distance >= baseround(cal_range, pixel_base):
        high = max(cal.keys())
    else:
        while stop * scale < high:
            stop += pixel_base

    # set scale
    for x in range(start, stop + 1, pixel_base):
        cal[x] = scale
        print(f'CAL: {x} {scale}')


# read local calibration data
calfile = 'camruler_cal.csv'
if os.path.isfile(calfile):
    with open(calfile) as f:
        for line in f:
            line = line.strip()
            if line and line[0] in ('d',):
                axis, pixels, scale = [_.strip() for _ in line.split(',', 2)]
                if axis == 'd':
                    print(f'LOAD: {pixels} {scale}')
                    cal[int(pixels)] = float(scale)


# convert pixels to units
def conv(x, y):
    d = distance(0, 0, x, y)
    scale = cal[baseround(d, pixel_base)]
    return x * scale, y * scale


# round to a given base
def baseround(x, base=1):
    return int(base * round(float(x) / base))


# distance formula 2D
def distance(x1, y1, x2, y2):
    return hypot(x1 - x2, y1 - y2)

#-------------------------------
# define frames
#-------------------------------

# define display frame
framename = "CamRuler ~ ClaytonDarwin's Youtube Channel"
cv2.namedWindow(framename, flags=cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)

#-------------------------------
# auto-measure mode
#-------------------------------

# Main logic for auto-measure
# Replace the previous auto-mode logic with the updated version
# which uses Length, Diameter, Width, and Height in centimeters (cm)
elif key_flags['auto']:

    mouse_mark = None

    # auto text data
    text.append('')
    text.append(f'AUTO MODE')
    text.append(f'UNITS: {unit_suffix}')
    text.append(f'MIN PERCENT: {auto_percent:.2f}')
    text.append(f'THRESHOLD: {auto_threshold}')
    text.append(f'GAUSS BLUR: {auto_blur}')

    # gray frame
    frame1 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

    # blur frame
    frame1 = cv2.GaussianBlur(frame1, (auto_blur, auto_blur), 0)

    # threshold frame
    frame1 = cv2.threshold(frame1, auto_threshold, 255, cv2.THRESH_BINARY)[1]

    # invert
    frame1 = ~frame1

    # find contours on thresholded image
    contours, _ = cv2.findContours(frame1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # draw crosshairs (after getting frame1)
    draw.crosshairs(frame0, 5, weight=2, color='green')

    # loop over the contours
    for c in contours:
        x1, y1, w, h = cv2.boundingRect(c)
        x2, y2 = x1 + w, y1 + h
        x3, y3 = x1 + (w / 2), y1 + (h / 2)

        # percent area
        percent = 100 * w * h / area

        # Ignore small or large objects
        if percent < auto_percent or percent > 60:
            continue

        # Convert measurements to cm

```python name=camruler_cm_measurements.py
        x1c, y1c = conv(x1 - cx, y1 - cy)
        x2c, y2c = conv(x2 - cx, y2 - cy)
        width = abs(x1c - x2c) / 10  # Convert to centimeters
        height = abs(y1c - y2c) / 10  # Convert to centimeters
        diameter = hypot(width, height) / 10  # Convert to centimeters
        length = max(width, height) / 10  # Convert to centimeters

        # plot
        draw.rect(frame0, x1, y1, x2, y2, weight=2, color='red')

        # add dimensions
        draw.add_text(frame0, f'Width: {width:.2f} cm', x1 - ((x1 - x2) / 2), min(y1, y2) - 8, center=True, color='red')
        draw.add_text(frame0, f'Height: {height:.2f} cm', x1 - ((x1 - x2) / 2), min(y1, y2) - 24, center=True, color='blue')
        draw.add_text(frame0, f'Diameter: {diameter:.2f} cm', x3, y2 + 8, center=True, top=True, color='green')
        draw.add_text(frame0, f'Length: {length:.2f} cm', x3, y2 + 24, center=True, top=True, color='orange')

#-------------------------------
# Kill sequence and cleanup
#-------------------------------
# Release camera and close windows
camera.stop()
cv2.destroyAllWindows()
exit()