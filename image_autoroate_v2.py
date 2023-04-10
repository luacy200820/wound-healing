from operator import pos
import cv2 
import numpy as np 
import glob 
import pandas as pd 
import matplotlib.pyplot as plt
import imutils
import math
from scipy import signal
from collections import defaultdict
from collections import defaultdict
import sys
import math 

def find_contour(hsv,low,high,img,center):
    mask = cv2.inRange(hsv,low,high)
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,iterations=4)
    plt.imshow(opening,cmap='gray')
    plt.show()
 
    cnts = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    rect_img = img.copy()
    for c in cnts: 
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(rect_img,[box],0,(0,0,255),1)
        plt.imshow(rect_img[:,:,(2,1,0)])
        plt.show()
        M = cv2.moments(c)
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        # cv2.circle(img,(cX,cY),5,(0,255,0),1)
        # plt.imshow(img[:,:,(2,1,0)])
        # plt.show()
        center.append([cY,cX,box])
        # print(cY,cX)
        
    return cnts,center

def find_contour_lab(lab,low,high,img,center,color):
    if color == 'yellow':
        low = np.array([190]) 
        high = np.array([255])
        mask = cv2.inRange(lab,low,high) #find yellow
    elif color == 'green':
        low = np.array([0]) 
        high = np.array([100])
        mask = cv2.inRange(lab,low,high) #find green
        
    kernel = np.ones((5,5),np.uint8)
    opening_white = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,iterations=2)
    plt.imshow(opening_white,cmap='gray')
    plt.show()
    cnts =''
    cnts = cv2.findContours(opening_white.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    rect_img = img.copy()
    for c in cnts: 
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(rect_img,[box],0,(0,0,255),1)
        plt.imshow(rect_img[:,:,(2,1,0)])
        plt.show()
        M = cv2.moments(c)
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        # cv2.circle(img,(cX,cY),5,(0,255,0),1)
        # plt.imshow(img[:,:,(2,1,0)])
        # plt.show()
        center.append([cY,cX,box])
        # print(cY,cX)
        
    return cnts,center

def findYellowBlock(point):
    if len(point) == 2:
        blue = point[0]
        yellow = point[1]
    elif len(point) == 3:
        blue = point[0]
        yellow = point[2]
        
    position = ''
    
    hor_diff = abs(blue[1] -  yellow[1])
    ver_diff = abs(blue[0] - yellow[0])
    # print('hor',hor_diff,ver_diff)

    if blue[1] < yellow[1] and ver_diff <= 15:
        position = 'left'
        # print('blue is left')
    elif blue[0] < yellow[0] and hor_diff <= 15:
        position = 'upper'
        # print('blue is upper')
    elif blue[1] >= yellow[1] and ver_diff <= 15 :
        position = 'right'
        # print('blue is right')
    elif blue[0] >= yellow[0] and hor_diff <=15:
        position = 'down'
        # print('blue is down')

    print('post',position)
    return position


def drawLines(img, lines, color=(0,0,255)):
    """
    Draw lines on an image
    """
    line_arr = []
    save_line = []
    isLine = True
    for line in lines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            # line_arr.append([x1,y1,x2,y2])
            # for index in range(len(line_arr)-1):
            #     ang = angle(line_arr[index],line_arr[len(line_arr)-1])
            #     print('====',len(line_arr))
            #     print('ang',ang)
            # print('len ',len(line_arr))
            if len(line_arr) >= 1:
                for index in range(len(line_arr)-1):
                    ang = angle(line_arr[index],line_arr[len(line_arr)-1])
                    # print('====',len(line_arr))
                    # print('ang',ang,line_arr[index],line_arr[len(line_arr)-1])
                    set_l1 = set(line_arr[index])
                    if index != 0:
                        set_l2 = set(line_arr[index-1])
                        if set_l1 == set_l2:
                            # print('equal')
                            pass
                        else :
                            # print('not equal',set_l1,set_l2)
                            
                            if ang >2 and ang <=70 :
                                isLine = False
                                break
                            elif ang > 70:
                                isLine = True
            
                if isLine:
                    line_arr.append([x1,y1,x2,y2])
                    save_line.append([[rho,theta]])
                    cv2.line(img, (x1,y1), (x2,y2), color, 1)
                    # cv2.imshow('line',img)
                    # cv2.waitKey(0)

            elif len(line_arr) == 0 :
                line_arr.append([x1,y1,x2,y2])
                save_line.append([[rho,theta]])

                cv2.line(img, (x1,y1), (x2,y2), color, 1)
                # cv2.imshow('line',img)
                # cv2.waitKey(0)
    return save_line

# 計算角度
def angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    # print(angle2)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle

def euclideanDistance(p1,p2):
    return math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )


def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """
    Group lines by their angle using k-means clustering.

    Code from here:
    https://stackoverflow.com/a/46572063/1755401
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))

    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # Get angles in [0, pi] radians
    angles = np.array([line[0][1] for line in lines])

    # Multiply the angles by two and find coordinates of that angle on the Unit Circle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)] for angle in angles], dtype=np.float32)

    # Run k-means
    if sys.version_info[0] == 2:
        # python 2.x
        ret, labels, centers = cv2.kmeans(pts, k, criteria, attempts, flags)
    else: 
        # python 3.x, syntax has changed.
        labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]

    labels = labels.reshape(-1) # Transpose to row vector

    # Segment lines based on their label of 0 or 1
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)

    segmented = list(segmented.values())
    print("Segmented lines into two groups: %d, %d" % (len(segmented[0]), len(segmented[1])))

    return segmented


def intersection(line1, line2):
    """
    Find the intersection of two lines 
    specified in Hesse normal form.

    Returns closest integer pixel locations.

    See here:
    https://stackoverflow.com/a/383527/5087436
    """

    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([[np.cos(theta1), np.sin(theta1)],
                  [np.cos(theta2), np.sin(theta2)]])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))

    return [[x0, y0]]


def segmented_intersections(lines):
    """
    Find the intersection between groups of lines.
    """

    intersections = []
    # inter_and_line= []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2)) 
                    # inter_and_line.append(intersection(line1, line2),line1, line2)
    return intersections

def segmented_intersections_point(lines):
    """
    Find the intersection between groups of lines.
    """

    intersections = []
    inter_and_line= []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2)) 
                    inter_and_line.append([intersection(line1, line2),line1, line2])
    return intersections,inter_and_line

def drawLines_(img, lines, color=(0,0,255)):
    """
    Draw lines on an image
    """
    for line in lines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img, (x1,y1), (x2,y2), color, 1)

def drawLines_single(img, lines,position, color=(0,0,255)):
    """
    Draw lines on an image, single find one line
    """
    # for line in lines:
    # 
    for rho,theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img, (x1,y1), (x2,y2), color, 1)
        position.append([ (x1,y1), (x2,y2)])
    
    return position

def line_equation(pt1,pt2,w,h):
    """找兩點直線方程式

    Args:
        pt1 (_type_): 座標1
        pt2 (_type_): 座標2
        w,h : image.shape[:2]
        direction (_type_): 水平或垂直

    Returns:
        _type_: 在0點時的另一個數值
    """
    pt1x,pt1y = pt1[0],pt1[1]
    pt2x,pt2y = pt2[0],pt2[1]
    
    k = (pt2y - pt1y) / (pt2x- pt1x)
    B1 = pt1y - k* pt1x
    B2 = pt2y - k* pt2x
    # print(B1,B2)
    # print('y=',k,'x+',B1)
    # Y1 = k*pt1x+B1
    # print(Y1)
       
    # if direction == 'horizontal': #找y=0 and y = w
    #     x = int(-B1/k)
    #     parameter1 = x
    #     x = int((w-B1)/k)
    #     parameter2 = x
    #     # print(x)
    # elif direction == 'vertical':
    #     y = int(B1 )
    #     parameter1 = y
    #     y = int(k*h+B1)
    #     parameter2 = y
        # print(y)
    y_0 = int(-B1/k)
    y_w = int((w-B1)/k)

    x_0 = int(B1)
    x_h = int(k*h+B1)

    direction=''

    print('x_0,x_h,y_0,y_w',x_0,x_h,y_0,y_w)
    if x_0 <= h and x_0 >=0 and x_h <= h and x_h >=0:
        parameter1 = x_0
        parameter2 = x_h
        direction = 'vertial'
        print('vertial')
    elif y_0 <= w and y_0 >=0 and y_w <=w  and y_w >=0:
        parameter1 = y_0
        parameter2 = y_w
        direction = 'horizontal'
        print('hor')
    else:
        parameter1 = 0
        parameter2 = 0
        print("can't judge")

    return parameter1,parameter2,direction
      
def find_angle(pt1,pt2,img,axis = 'horizontal'):
    h,w = img.shape[:2]
    if axis == 'horizontal':
        angle = math.atan2((pt2-0),(w-pt1)) #弧度，與y軸夾角
    elif axis == 'vertial':
        angle = math.atan2((pt1-h),(0-pt1)) #弧度，與x軸夾角
    
    theta = angle*(180 / math.pi)
    print(theta)
    return theta

# centerpoint = []
# rgb_pixel = []
# # 20x20 pixel average
# avg_color_block = []
# img_name = '170_7_20220901_1_' #170_7_20220901_1_,8_2_20220818_,69_5_20220822_5_,170_7_20220901_3_
# # img = fr'C:\Users\buslab\Desktop\skin_img\unlabel\card\{img_name}.png'
# img = fr'D:\wu\skin_model\data_img\dataset13\images\237_2_20221128_1_.png'
# image_path = sorted(glob.glob(img))
# for img in image_path:
#     name = img.split('\\')[-1]
#     image = cv2.imread(img)
#     image = cv2.resize(image,(600,600))
#     img_ = image.copy()
#     hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
#     lab = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
#     l,a,b = cv2.split(lab)

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blur_img = cv2.bilateralFilter(gray,5,75,75)
#     laplacian_crop = cv2.Laplacian(blur_img,cv2.CV_64F)
#     th,binary = cv2.threshold(laplacian_crop,5,255,cv2.THRESH_BINARY)
#     # plt.imshow(binary,cmap='gray')
#     binary = np.uint8(binary)

#     # Detect lines
#     rho = 1
#     theta = np.pi/180
#     thresh = 170
#     lines = cv2.HoughLines(binary, rho, theta, thresh)

#     # Draw all Hough lines in red
#     img_with_all_lines = np.copy(image)
#     img_with_all_lines_ = np.copy(image)
#     # print('line',lines)
#     filter_lines = drawLines(img_with_all_lines, lines) #處理的
#     filter_lines_ = drawLines_(img_with_all_lines_, lines)

#     # plt.subplot(121),plt.imshow(img_with_all_lines[:,:,(2,1,0)])
#     # plt.subplot(122),plt.imshow(img_with_all_lines_[:,:,(2,1,0)])
#     # plt.show()

#     # Cluster line angles into 2 groups (vertical and horizontal)
#     segmented = segment_by_angle_kmeans(lines, 2)
#     # Find the intersections of each vertical line with each horizontal line
#     # intersections = segmented_intersections(segmented)
#     intersections,inter_point = segmented_intersections_point(segmented)
#     # print(inter_point)

#     img_with_segmented_lines = np.copy(image)

#     # Draw vertical lines in green
#     vertical_lines = segmented[1]
#     img_with_vertical_lines = np.copy(image)
#     drawLines(img_with_segmented_lines, vertical_lines, (0,255,0))

#     # Draw horizontal lines in yellow
#     horizontal_lines = segmented[0]
#     img_with_horizontal_lines = np.copy(image)
#     drawLines(img_with_segmented_lines, horizontal_lines, (0,255,255))

#     # print('inter',inter_point)
#     # print('vertical',vertical_lines)
#     # print('horizontal',horizontal_lines)
#     # Draw intersection points in magenta
#     for point in intersections:
#         pt = (point[0][0], point[0][1])
#         length = 5
#         cv2.line(img_with_segmented_lines, (pt[0], pt[1]-length), (pt[0], pt[1]+length), (255, 0, 255), 1) # vertical line
#         cv2.line(img_with_segmented_lines, (pt[0]-length, pt[1]), (pt[0]+length, pt[1]), (255, 0, 255), 1)

#     cv2.imshow('result line',img_with_all_lines)
#     # plt.imshow( img_with_segmented_lines[:,:,(2,1,0)])
#     # plt.show()

#     h,w= image.shape[:2]
#     # print(h,w)
#     save_position = []
#     test = image.copy()
#     test2 = image.copy()
#     save_position = drawLines_single(test,horizontal_lines[0],save_position) #垂直
#     save_position = drawLines_single(test2,vertical_lines[0],save_position) #水平

#     print('save_position',save_position) #前面為垂直，後面為水平
#     # position_num1,position_num2 = line_equation(save_position[0][0],save_position[0][1],w,h,'horizontal')
#     position_num1,position_num2,dir_line= line_equation(save_position[1][0],save_position[1][1],w,h) #隨便一條
#     """
#     不一定是垂直或水平@@
#     """
#     negative_1 = False
#     negative_2 = False #順時鐘轉為負
#     # position_ver_num1,position_ver_num2,dir_line2 = line_equation(save_position[0][0],save_position[0][1],w,h)
#     print("position_num1",position_num1,position_num2,dir_line)
#     if position_num1 == 0 and position_num2 == 0:
#         continue
#     theta = find_angle(position_num1,position_num2,test,dir_line)
#     line_point = []
#     # 判斷是水平還是垂直
#     if dir_line == 'horizontal':
#         if position_num2 > position_num1:
#             negative_1 = True
#         line_point.append([position_num1,0,position_num2,h])
#         line_point.append([position_num1,0,position_num1,h])
#     elif dir_line == 'vertial':
#         if position_num2 < position_num1:
#             negative_2 = True
#         line_point.append([0,position_num1,w,position_num2])
#         line_point.append([0,position_num1,w,position_num1])

#     # print(line_point)
#     re1 = angle(line_point[0],line_point[1]) # 計算角度
#     # print(re1)
#     # cv2.circle(test,(position_num1,0),8,(0,0,255),-1)
#     # cv2.line(test,(line_point[0][0],line_point[0][1]),(line_point[0][2],line_point[0][3]),(255,0,0),1) #垂直
#     # cv2.line(test,(line_point[1][0],line_point[1][1]),(line_point[1][2],line_point[1][3]),(255,255,0),1) #與垂直線
#     # cv2.imshow('test1',test)

#     # ===================
#     position_num1,position_num2,dir_line= line_equation(save_position[0][0],save_position[0][1],w,h)
#     print("position_num1",position_num1,position_num2,dir_line)
#     if position_num1 == 0 and position_num2 == 0:
#         continue
#     theta = find_angle(position_num1,position_num2,test,dir_line)
#     line_point = []
#     if dir_line == 'horizontal':
#         line_point.append([position_num1,0,position_num2,h])
#         line_point.append([position_num1,0,position_num1,h])
#     elif dir_line == 'vertial':
#         line_point.append([0,position_num1,w,position_num2])
#         line_point.append([0,position_num1,w,position_num1])

#     # print(line_point)
#     re2 = angle(line_point[0],line_point[1])
#     # print(re2)
#     if negative_1 == True:
#         re1 = -re1
#     if negative_2 == True:
#         re2 = -re2 
#     # angle_mean = int((re1+re2)/2)
#     # angle_mean = min(re1,re2)
#     # angle_mean = max(re1,re2)
#     if re1 < 0:
#         angle_mean = re1
#     elif re2 < 0:
#         angle_mean = re2 
#     else:
#         # angle_mean = max(re1,re2)
#         angle_mean = int((re1+re2)/2)
#     print("re1,2:",re1,re2,angle_mean)
#     # ===================
#     h,w = image.shape[:2]
#     center = (w//2,h//2)
#     rotated = image.copy()
#     M = cv2.getRotationMatrix2D(center,(angle_mean),1.0)
#     rotated = cv2.warpAffine(rotated,M,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
#     # cv2.putText(rotated,'Angle: {:.2f} degrees'.format(angle),(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255),2)
#     # cv2.imwrite(fr'C:\Users\buslab\Desktop\skin_card\rotate_{re1}_{re2}_{name}',rotated)
#     # cv2.imshow('rotate',rotated)
#     # cv2.waitKey(0)
    


        