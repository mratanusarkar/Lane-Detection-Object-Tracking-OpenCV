import cv2
import numpy as np


def edge(img):
    # convert color image to grayscale 
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #cv2.imshow('converted to grayscale', gray)
    #cv2.waitKey()
    
    # apply a gaussian blur using nxn kernel:
    # (each px replaced with weighted average)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    #cv2.imshow('after gaussian blur', blur)
    
    # canny function to find edges
    canny = cv2.Canny(blur, 0, 100)
    #cv2.imshow('canny edge detection', canny)
    #cv2.waitKey()
    
    return canny


def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([
            [(0,500), (0,height), (1100,height), (550,250)]
            ])

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def generate_coordinates(img, line_parameters):
    slope, intercept = line_parameters
    y1 = img.shape[0]
    y2 = int(y1*(1/2))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1, x2,y2])


def average_slope_intercept(img, lines):
    height = img.shape[0]
    left_fit = []
    right_fit = []
    center_fit = []
    
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        
        #TODO even better logic!
        if abs(intercept) > height:
            center_fit.append((slope,intercept))
        elif slope>0:
            right_fit.append((slope,intercept))
        else:
            left_fit.append((slope,intercept))

    left_fit_avg = np.average(left_fit, axis=0)
    center_fit_avg = np.average(center_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)
    
    try:
        left_line = generate_coordinates(img, left_fit_avg)
    except Exception:
        print("cannot detect Left lane !!")
        left_line = [0,0,0,0]
    
    try:
        center_line = generate_coordinates(img, center_fit_avg)
    except Exception:
        print("cannot detect Central lane !!")
        center_line = [0,0,0,0]
        
    try:
        right_line = generate_coordinates(img, right_fit_avg)
    except Exception:
        print("cannot detect Right lane !!")
        right_line = [0,0,0,0]
        
        
    return np.array([left_line, center_line, right_line])
    



def display_lines(img, lines, color, thickness):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2), color, thickness)
    return line_image


################################################################

## read the image
#image = cv2.imread('test_image_01.jpg')
#lane_image = np.copy(image)
##cv2.imshow('original image', lane_image)
#
## detect edges
#canny = edge(lane_image)
#
## define ROI
#masked = region_of_interest(canny)
#
## get lines
#lines = cv2.HoughLinesP(masked, 2, 1*(np.pi/180), 100, np.array([]), minLineLength=40, maxLineGap=5)
#
## average out many detected similar lines into one
#averaged_lines = average_slope_intercept(lane_image, lines)
#
## draw lines into the image
#line_image = display_lines(lane_image, averaged_lines)
#
## output modified image with detected lanes
#output_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
#cv2.imshow('result', output_image) 
#cv2.waitKey()


# open video capture and read image frame by frame
videocap = cv2.VideoCapture("test_video_01.mp4")
while(videocap.isOpened()):
    ret, frame = videocap.read()
    
    if ret != True:
        break
    
    lane_image = np.copy(frame)
    #cv2.imshow('original image', lane_image)
    
    # detect edges
    canny = edge(lane_image)
    
    # define ROI
    masked = region_of_interest(canny)
    
    # get lines
    lines = cv2.HoughLinesP(masked, 2, 1*(np.pi/180), 100, np.array([]), minLineLength=40, maxLineGap=5)
    
    # average out many detected similar lines into one
    averaged_lines = average_slope_intercept(lane_image, lines)
    
    # draw lines into the image
    all_lines = display_lines(lane_image, lines, (0,255,0), 1)
    line_image = display_lines(lane_image, averaged_lines, (255,0,0), 3)
    
    # output modified image with detected lanes
    detected_lines = cv2.addWeighted(all_lines, 0.8, line_image, 1, 1)
    output_image = cv2.addWeighted(detected_lines, 0.8, lane_image, 1, 1)
    cv2.imshow('output', output_image) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
videocap.release()
cv2.destroyAllWindows()
