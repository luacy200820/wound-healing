import numpy as np 
import cv2 
import matplotlib.pylab as plt
import glob 
import os 
import imutils
from scipy import signal
import math
from skimage import morphology
import operator
from tqdm import tqdm
from collections import Counter
import pandas as pd
import operator
name = ''
def create_dir(path):
  if not os.path.exists(path):
    os.makedirs(path)
    
def find_contour(hsv,low,high,img,center,pre_area):
    """
    找顏色方框，回傳中心點與四角點
    """
    mask = cv2.inRange(hsv,low,high)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,iterations=2)

    cnts = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    rect_img = img.copy()
    area=0

    for c in cnts: 
        rect = cv2.minAreaRect(c)
        M = cv2.moments(c)
        area = cv2.contourArea(c) #計算方框的面積，如果太小則不考慮
        # print('rect',area)
        if pre_area == 0:
            pre_area = area 
        # print('rect',area)
        if area > 400 and area <= (pre_area +350):
        # if area > 380:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(rect_img,[box],0,(0,0,255),1)
            # cv2.imshow('block',rect_img)
            # cv2.waitKey(0)
            # plt.show()
            M = cv2.moments(c)
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            # cv2.circle(img,(cX,cY),5,(0,255,0),1)
            # plt.imshow(img[:,:,(2,1,0)])
            # plt.show()
            center.append([cY,cX,box])
            break
        
    return cnts,center,area

def findGreenBlock(point):
    """綠色藍色的位置，return 藍對綠的方向"""
    blue = point[0]
    green = point[1]
    position = ''

    hor_diff = abs(blue[1] -  green[1])
    ver_diff = abs(blue[0] - green[0])
    # print('hor',hor_diff,ver_diff)

    if blue[1] < green[1] and ver_diff <= 15:
        position = 'left'
        # print('blue is left')
    elif blue[0] < green[0] and hor_diff <= 15:
        position = 'upper'
        # print('blue is upper')
    elif blue[1] >= green[1] and ver_diff <= 15 :
        position = 'right'
        # print('blue is right')
    elif blue[0] >= green[0] and hor_diff <=15:
        position = 'down'
        # print('blue is down')

    print('post',position)
    return position

def getEdgeline(crop_img):
    """crop_img : 裁剪的影像，做拉布拉斯，再做threshold"""
    gray_crop = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
    blurred_crop = cv2.bilateralFilter(gray_crop,9,75,75)

    # convolute with proper kernels
    laplacian_crop = cv2.Laplacian(blurred_crop,cv2.CV_64F)
    th,binary = cv2.threshold(laplacian_crop,5,255,cv2.THRESH_BINARY)
    # plt.imshow(binary,cmap='gray')
    binary = np.uint8(binary)

    return binary

def getangle(l1,l2):
    """
    計算角度:(point1x,point1y), (point2x,point2y)
    """
    dx1 = l1[0][0] - l1[1][0]
    dy1 = l1[0][1] - l1[1][1]
    dx2 = l2[0][0] - l2[1][0]
    dy2 = l2[0][1] - l2[1][1]
    
    angle1 = math.atan2(dy1,dx1)
    angle1 = int(angle1*180 / math.pi)
    
    angle2 = math.atan2(dy2,dx2)
    angle2 =int(angle2*180 / math.pi)
    
    if angle1 * angle2 >= 0:
        insideAngle = abs(angle1 - angle2)
    else:
        insideAngle = abs(angle1) + abs(angle2)
        if insideAngle > 180:
            insideAngle = 360 - insideAngle
    insideAngle %= 180
    return insideAngle

def rotated_angle(bin,img,angle):
    """
    針對二值影像與crop影像作旋轉
    """
    h,w = img.shape[:2]
    cX,cY = w//2,h//2
    M = cv2.getRotationMatrix2D((cX,cY),angle,1.0)
    rotated = cv2.warpAffine(img,M,(w,h),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_REPLICATE)

    rotated_bin = cv2.warpAffine(bin,M,(w,h),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_REPLICATE)

    return rotated_bin,rotated #回傳旋轉後的圖(二值化and crop)

def rotated_rgb_angle(img,angle):
    """
    針對原始影像作旋轉
    """
    h,w = img.shape[:2]
    cX,cY = w//2,h//2
    M = cv2.getRotationMatrix2D((cX,cY),angle,1.0)
    rotated = cv2.warpAffine(img,M,(w,h),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_REPLICATE)

    return rotated #回傳旋轉後的圖(二值化and crop)

def findLines(binary,crop,minlen=8,maxgap=20):
    """
    找直線，回傳所有找到的線、斜率為0的線、最常出現的斜率
    """
    base = cv2.HoughLinesP(binary, 1, np.pi / 180,15, minLineLength=minlen, maxLineGap=maxgap)
    pixel_array = []
    pixel_freq = []
    counter_dict = dict()
    show_img = crop.copy()
    show_img2= crop.copy()
    h,w = binary.shape[:2]
    if base is not None:
        for line in base:
            x1, y1, x2, y2 = line[0]
            cv2.line(show_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            pixel_array.append([(x1, y1), (x2, y2)])
            # res = getangle(([x1, y1],[ x2, y2]),([h,0],[h,w]))
            res = getangle(([x1, y1],[ x2, y2]),([h,0],[int(h/2),0]))
            #計算角度出現次數
            if res in counter_dict:
                counter_dict[res]+=1
            else:
                counter_dict[res] = 1
                
            if res <= 92 and res >=88:
                cv2.line(show_img2, (x1, y1), (x2, y2), (255, 0, 255), 1)
                pixel_freq.append([(x1, y1), (x2, y2)]) #存斜率為0
            # print('res angle',res)
            

    print('counter_dict',counter_dict)
    counter_dict =(sorted(counter_dict.items(), key=lambda x: x[0]))
    counter_dict = dict((x, y) for x, y in counter_dict)
    print('counter_dict',counter_dict)
    if not counter_dict:
        return -1,-1,-1
    
    collection_words = Counter(counter_dict) 
    print('collection_words',collection_words)
    most_counterNum = collection_words.most_common(5)
    print('most_counterNum',most_counterNum,len(most_counterNum))
    # print('collection_words_sort',collection_words_sort)
    max_key_ = most_counterNum[0][0]
    # max_key = max(counter_dict.items(),key=operator.itemgetter(1))[0] #回傳最常出現的角度
    print('max_key_',max_key_)
    max_key = max_key_
    t=0
    while( (max_key < 88 or max_key > 92 ) and t < 5 ):
        if len(most_counterNum) == t:
            break
        max_key = most_counterNum[t][0]
        t +=1
    if max_key < 88 or max_key > 92 : 
        max_key = max_key_ 
        
    print('max_key',max_key) 
      

    return pixel_array,pixel_freq,max_key

def findVertialLine(crop,line_array):
    """
    crop:裁剪的影像，vertical_array:線的座標位置 
    return vertical_line: 中垂線的座標
    """
    vertical_line = [] 
    line_img = crop.copy()

    for i in range(len(line_array)):
        point1 = line_array[i][0]
        point2 = line_array[i][1]
        
        if point1[0] != point2[0]:
            slope = (point2[1] - point1[1]) / ( point1[0] - point2[0])
        else:
            slope = 0
        
        # 中心點座標
        midpoint = (int((point1[0]+point2[0]) /2) ,int((point1[1]+point2[1]) /2))
        
        x1 = point1[0] -60 
        x2 = point1[0] +60
        # print('pixel array',point1,point2)
        
        y1 = int(slope * ( x1 - midpoint[0]) + midpoint[1] )
        y2 = int(slope * ( x2 - midpoint[0]) + midpoint[1] )

        vertical_line.append([slope,(x2,y2),(x1,y1),midpoint])
        cv2.line(line_img,(x2,y2),(x1,y1),(0,0,255),1)
        
    # plt.imshow(line_img[:,:,(2,1,0)])
    # plt.show()
    return vertical_line

def mid_find_line(mid,crop):
    line_save_posit = []
    h,w = crop.shape[:2]
    test = crop.copy()
    gray = cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)
    negative = 255-gray #負片
    pixel_value = []
    line_img2 = crop.copy()
    
    cv2.line(line_img2,(0,mid),(w,mid),(0,255,0),1)#劃線
    # plt.imshow(line_img2)
    # plt.show()   
    for i in range(w):
        pixel_value.append(negative[mid][i])
    print('pixel_value',pixel_value)
    mean_pixel = np.mean(pixel_value)
    std_pixel = np.std(pixel_value)
    print('mean,std,total',mean_pixel,std_pixel)
    ans = []
    ans_low = []
    for k in range(1,len(pixel_value)-1): #看左右1個
        if pixel_value[k] >= pixel_value[k-1] and pixel_value[k] > pixel_value[k+1] and pixel_value[k] >= mean_pixel:
            ans.append(k)
    for k in range(1,len(pixel_value)-1): #看左右1個
        if pixel_value[k] <= pixel_value[k-1] and pixel_value[k] < pixel_value[k+1] and pixel_value[k] <= mean_pixel:
            ans_low.append(k)
    ##高低要交錯
    tmp_h = 0
    tmp_l = 0
    final_peak = []
    if len(ans) >5 and len(ans_low) >5:
        while(1):
            # print('ans,ans_low',ans[tmp_h],ans_low[tmp_l])
            if ans[tmp_h] < ans_low[tmp_l] and ans_low[tmp_l] < ans[tmp_h+1]:
                final_peak.append(ans[tmp_h])
                tmp_h+=1
                tmp_l+=1
            elif ans[tmp_h] < ans_low[tmp_l] and ans_low[tmp_l] > ans[tmp_h+1]:
                tmp_h +=1
                
            elif ans[tmp_h] > ans_low[tmp_l] and ans_low[tmp_l] < ans[tmp_h+1]:
                tmp_l +=1
            if tmp_h == len(ans)-1 or tmp_l == len(ans_low)-1 :
                if tmp_h < len(ans)-1:
                    final_peak.append(ans[tmp_h])
                break
    # else:
    #     continue
    print('ans, ans low',ans,ans_low)
    
    print('final peak',final_peak)
    if final_peak and len(final_peak) >5:
        test1  = crop.copy()
        # for i in final_peak:
        #     plt.plot(i,mid,  marker='s', color="y")
        #     plt.imshow(test1)
        #     plt.title('slide window final')
        # plt.show()   
        # plt.savefig(fr'C:\Users\buslab\Desktop\time series\69\measure\{name}')   
        # plt.clf()
        diffPixel = final_peak[-1]-final_peak[0]
        line_save_posit.append([mid,diffPixel,len(final_peak)-1,final_peak])
        return line_save_posit
    else:
        return []
def getFrequencyLine_(crop):

    line_save_posit = []
    temp = 255
    h,w = crop.shape[:2]
    test = crop.copy()
    gray = cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)
    negative = 255-gray #負片
    mid = int(h / 2)
    mid_ = mid
    result = mid_find_line(mid,test)
    while(1):
        print('mid_',mid_)
        if mid_ >= h :
            break
        elif result: #不為空
            return result
        elif mid_ <= 1:
            mid_ = int((mid+h)/2)
            result = mid_find_line(mid_,test)
        else:
            mid_ = int(mid_/2)
            result = mid_find_line(mid_,test)
            
    return []

def getFrequencyLine(vertical_array,crop):
    save_paraline = []
    for i in range(len(vertical_array)):
        # print('i',i)
        s = vertical_array[i][0] # 斜率
        if s == 0 : #留下斜率0的
            save_paraline.append(vertical_array[i])

    # print(save_paraline)
    
    save_paraline.sort(key=lambda x:x[3][1])
    # print(len(save_paraline))

    line_save_posit = []
    temp = 255
    istrue = False
   
    for line in range(len(save_paraline)):
        line_img2 = crop.copy()
        hsv_2 = crop.copy()
        hsv_2 = cv2.cvtColor(hsv_2,cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv_2)
    
        cv2.line(line_img2,save_paraline[line][1],save_paraline[line][2],(0,temp,0),1)
        # print(save_paraline[line][1],save_paraline[line][2])
 
        # fig = plt.figure(figsize=(12,12))
        # plt.subplot(131)
        # plt.imshow(line_img2[:,:,(2,1,0)])
        # plt.title(line)
    
        # 變負片、找波峰
        pixel_value = []
        test = crop.copy()
        
        gray = cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)
        negative = 255-gray #負片
        W = crop.shape[1]
        line_ = save_paraline[line][1][1] #找x軸
        # 這裡座標是(Y,X) = (固定,變動)
        
        print('w-line',W-line_,line_)
        if line_ > 3 and W-line_ >4:
            test1  = crop.copy()
            # print(negative.shape)
            # print(line_)
            for w in range(W):
                # if s[line_][w] < 100:
                    # print('s[line][w]', s[line_][w],negative[line_][w])
                pixel_value.append(negative[line_][w])
                
            # plt.subplot(132)
            # plt.imshow(negative,cmap='gray')
            # plt.subplot(133)
            # plt.plot(pixel_value,'-o')
            
            # plt.show()
            mean_pixel = np.mean(pixel_value)
            std_pixel = np.std(pixel_value)
            print('mean,std,total',mean_pixel,std_pixel,mean_pixel+std_pixel)
            
            ###################
            ###slide windows start
            ans = []
            ans_low = []
            for k in range(1,len(pixel_value)-1): #看左右1個
                if pixel_value[k] >= pixel_value[k-1] and pixel_value[k] > pixel_value[k+1] and pixel_value[k] >= mean_pixel:
                    ans.append(k)
            for k in range(1,len(pixel_value)-1): #看左右1個
                if pixel_value[k] <= pixel_value[k-1] and pixel_value[k] < pixel_value[k+1] and pixel_value[k] <= mean_pixel:
                    ans_low.append(k)
            # print('ans',ans)
            # print('ans low',ans_low)
                        
            ##高低要交錯
            tmp_h = 0
            tmp_l = 0
            final_peak = []
            if len(ans) >5 and len(ans_low) >5:
                while(1):
                    # print('ans,ans_low',ans[tmp_h],ans_low[tmp_l])
                    if ans[tmp_h] < ans_low[tmp_l] and ans_low[tmp_l] < ans[tmp_h+1]:
                        final_peak.append(ans[tmp_h])
                        tmp_h+=1
                        tmp_l+=1
                    elif ans[tmp_h] < ans_low[tmp_l] and ans_low[tmp_l] > ans[tmp_h+1]:
                        tmp_h +=1
                        
                    elif ans[tmp_h] > ans_low[tmp_l] and ans_low[tmp_l] < ans[tmp_h+1]:
                        tmp_l +=1
                    if tmp_h == len(ans)-1 or tmp_l == len(ans_low)-1 :
                        if tmp_h < len(ans)-1:
                            final_peak.append(ans[tmp_h])
                        break
            else:
                continue
            
                    
            # plt.plot(pixel_value, 'b',label='polyfit values'),plt.title(name)
            # for ii in range(len(final_peak)):
            #     plt.plot(final_peak[ii], pixel_value[final_peak[ii]],'*',markersize=10)
            # plt.show()
            print('final peak',final_peak)
            # for i in final_peak:
            #     plt.plot(i,line_,  marker='s', color="y")
            #     plt.imshow(test1)
            #     plt.title('slide window final')
            # # plt.show()   
            # # plt.savefig(fr'C:\Users\buslab\Desktop\time series\69\measure\{name}')   
            # plt.clf()
            # print('real cm: ,diff pixel: ',len(final_peak)-1,final_peak[-1]-final_peak[0])
            # print('final_peak',final_peak)
            if len(final_peak) <5:
                print('not this line!!')
                continue
            else:
                istrue = True
                print('find it')
                diffPixel = final_peak[-1]-final_peak[0]
                line_save_posit.append([line_,diffPixel,len(final_peak)-1,final_peak])
                # line_save_posit.append([line_,diffPixel,len(ans)-1,ans])
                #第幾行直線，最大與最小差pixel數，實際毫米數，找到的波峰位置
                return line_save_posit
            ##############
            
    print('istrue is false')
    if istrue is False:
        for line in range(len(save_paraline)):
            head = 0 
            end = save_paraline[line][1][1]
            mid1 = int((head + end )/ 2)
            head = crop.shape[0]
            mid2 = int((head + end) /2)
            
            for mid_line in [mid1,mid2]:
                line_img2 = crop.copy()
                hsv_2 = crop.copy()
                hsv_2 = cv2.cvtColor(hsv_2,cv2.COLOR_BGR2HSV)
                h,s,v = cv2.split(hsv_2)
            
                cv2.line(line_img2,(save_paraline[line][1][0],mid_line),(save_paraline[line][2][0],mid_line),(0,temp,0),1)
    
                # fig = plt.figure(figsize=(12,12))
                # plt.subplot(131)
                # plt.imshow(line_img2[:,:,(2,1,0)])
                # plt.title(line)
            
                # 變負片、找波峰
                pixel_value = []
                test = crop.copy()
                gray = cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)
                negative = 255-gray #負片
                W = crop.shape[1]
                # line_ = save_paraline[line][1][1] #找x軸
                line_ = mid_line
                # 這裡座標是(Y,X) = (固定,變動)
                
                # print('w-line',W-line_,line_)
                if line_ > 3 and W-line_ >4:
                    test2 = crop.copy()
                    for w in range(W):
                        pixel_value.append(negative[line_][w])
                        
                    # plt.subplot(132)
                    # plt.imshow(negative,cmap='gray')
                    # plt.subplot(133)
                    # plt.plot(pixel_value)
                    # plt.show()
                    
                    mean_pixel = np.mean(pixel_value)
                    std_pixel = np.std(pixel_value)
                    print('mean,std,total',mean_pixel,std_pixel,mean_pixel+std_pixel)
                    ###################
                    ###slide windows start
                    ans = []
                    ans_low = []
                    for k in range(1,len(pixel_value)-1): #看左右1個
                        if pixel_value[k] >= pixel_value[k-1] and pixel_value[k] > pixel_value[k+1] and pixel_value[k] >= mean_pixel:
                            ans.append(k)
                    for k in range(1,len(pixel_value)-1): #看左右1個
                        if pixel_value[k] <= pixel_value[k-1] and pixel_value[k] < pixel_value[k+1] and pixel_value[k] <= mean_pixel:
                            ans_low.append(k)
                    print('ans',ans)
                    print('ans low',ans_low)
                    
                    # plt.plot(pixel_value, 'b',label='polyfit values'),plt.title('high')
                    # for ii in range(len(ans)):
                    #     plt.plot(ans[ii], pixel_value[ans[ii]],'h',markersize=10)
                    # plt.show()
                    # for i in ans:
                    #     plt.plot(i,line_,  marker='v', color="r")
                    #     plt.imshow(crop)
                    #     plt.title('slide window high')
                    # plt.show()   
                    
                    # plt.plot(pixel_value, 'b',label='polyfit values'),plt.title('low')
                    # for ii in range(len(ans_low)):
                    #     plt.plot(ans_low[ii], pixel_value[ans_low[ii]],'*',markersize=10)
                    # plt.show()
                    # for i in ans_low:
                    #     plt.plot(i,line_,  marker='X', color="r")
                    #     plt.imshow(crop)
                    #     plt.title('slide window low')
                    # plt.show()  
                    ##slide windows start。 ans,ans_low 都是存位置
                    
                    ##高低要交錯
                    tmp_h = 0
                    tmp_l = 0
                    final_peak = []
                    if len(ans) >5 and len(ans_low) >5:
                        while(1):
                            # print('ans,ans_low',ans[tmp_h],ans_low[tmp_l])
                            if ans[tmp_h] < ans_low[tmp_l] and ans_low[tmp_l] < ans[tmp_h+1]:
                                final_peak.append(ans[tmp_h])
                                tmp_h+=1
                                tmp_l+=1
                            elif ans[tmp_h] < ans_low[tmp_l] and ans_low[tmp_l] > ans[tmp_h+1]:
                                tmp_h +=1
                                
                            elif ans[tmp_h] > ans_low[tmp_l] and ans_low[tmp_l] < ans[tmp_h+1]:
                                tmp_l +=1
                            if tmp_h == len(ans)-1 or tmp_l == len(ans_low)-1 :
                                if tmp_h < len(ans)-1:
                                    final_peak.append(ans[tmp_h])
                                break
                    else:
                        continue
                    
                            
                    # plt.plot(pixel_value, 'b',label='polyfit values'),plt.title(name)
                    # for ii in range(len(final_peak)):
                    #     plt.plot(final_peak[ii], pixel_value[final_peak[ii]],'*',markersize=10)
                    # plt.show()
                    print('final peak',final_peak)
                    # for i in final_peak:
                        # plt.plot(i,line_,  marker='s', color="y")
                        # plt.imshow(test2)
                        # plt.title('slide window final')
                    # plt.show()    
                    # plt.savefig(fr'C:\Users\buslab\Desktop\time series\69\measure\{name}')   
                    # plt.clf()
            
                      
                    # print('real cm: ,diff pixel: ',len(final_peak)-1,final_peak[-1]-final_peak[0])
                    # print('final_peak',final_peak)
                    if len(final_peak) <5:
                        print('not this line!!')
                        continue
                    else:
                        istrue = True
                        print('find it')
                        diffPixel = final_peak[-1]-final_peak[0]
                        line_save_posit.append([line_,diffPixel,len(final_peak)-1,final_peak])
                        # line_save_posit.append([line_,diffPixel,len(ans)-1,ans])
                        #第幾行直線，最大與最小差pixel數，實際毫米數，找到的波峰位置
                        return line_save_posit
                   
    return line_save_posit


def calculatePix(objectMask,pixel):
    w,h = objectMask.shape[:2]
    num = 0 #pixel total number
    for i in range(w):
        for j in range(h):
            if objectMask[i][j] == pixel:
                num +=1
    return num


def getMask(msk):
    wound_image = msk.copy()
    gray2 = cv2.cvtColor(wound_image,cv2.COLOR_BGR2GRAY)
    _,th = cv2.threshold(gray2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #OTSU
    area = findCountour(th,wound_image) #用面積回推
    objectPixel = calculatePix(th,255)
    return objectPixel,area

def findCountour(bin,wound):
    contours,hierarchy = cv2.findContours(bin, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    M = cv2.moments(cnt)
    area = cv2.contourArea(cnt)
    print('area',area)
    return area
    
def calculateArea(line_array,objectPixel):
    """
    用pixel計算大小
    """
    ratio_pixel_cm = line_array[0][1]* line_array[0][1]
    real_area = round(int(line_array[0][2])/10 * int(line_array[0][2])/10,3)
    area = objectPixel * real_area / ratio_pixel_cm
    return area

def sharpen(img, sigma=300):    
    # sigma = 5、15、25
    blur_img = cv2.GaussianBlur(img, (0, 0), sigma)
    usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
    return usm
    
def calculateCountourArea(line_array,objectArea):
    """
    用面積計算大小
    """
    ratio_pixel_cm = line_array[0][1]* line_array[0][1]
    real_area = round(line_array[0][2]/10 * line_array[0][2]/10,3)
    area = objectArea * real_area / ratio_pixel_cm
    return area

def equimage(img):
    #影像強化+骨架化； 輸入為RGB，輸出為GRAY
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
    blurred_crop = cv2.bilateralFilter(equ,5,75,75)
    # convolute with proper kernels
    laplacian_crop = cv2.Laplacian(blurred_crop,cv2.CV_64F)
    th,binary_ = cv2.threshold(laplacian_crop,5,255,cv2.THRESH_BINARY)
    # plt.imshow(binary,cmap='gray')
    binary = np.uint8(binary_)
    binary[binary==255]=1
    skeleton0=morphology.skeletonize(binary)
    binary = skeleton0.astype(np.uint8)*255
    # plt.imshow(binary)
    return binary

def skeleton(binimg): #骨架化
    blurred_crop = cv2.bilateralFilter(binimg,3,25,25)
    th,binary = cv2.threshold(blurred_crop,5,255,cv2.THRESH_BINARY)
    binary[binary==255]=1
    skeleton0=morphology.skeletonize(binary)
    binimg = skeleton0.astype(np.uint8)*255
    return binimg 
 
# if __name__=="__main__":
    
#     # name = '69_5_20220822_5_json' 
#     img_path  = fr'C:\Users\buslab\OneDrive - 國立成功大學 National Cheng Kung University\桌面\time series\246\images\246_4_20221229_1.png'
#     mask_path = fr'C:\Users\buslab\OneDrive - 國立成功大學 National Cheng Kung University\桌面\time series\246\images\246_4_20221229_1.png'
    
#     img_glob = sorted(glob.glob(img_path))
#     mask_glob = sorted(glob.glob(mask_path))
#     print(len(img_glob),len(mask_glob))
    
#     score =[]
#     create_dir(r'C:\Users\buslab\OneDrive - 國立成功大學 National Cheng Kung University\桌面\time series\250\measure')
#     for img,msk in tqdm(zip(img_glob,mask_glob),total=len(img_glob)) :
#         name = img.split('\\')[-1]
#         print(name)
       
#         img = cv2.imdecode(np.fromfile(img,dtype=np.uint8),-1)
#         img = cv2.resize(img,(512,512))
#         # cv2.imshow('img',img)
#         # cv2.waitKey(0)
#         # msk = cv2.imread(msk)
#         msk = cv2.imdecode(np.fromfile(msk,dtype=np.uint8),-1)
#         msk = cv2.resize(msk,(512,512))

#         # msk = cv2.cvtColor(msk,cv2.COLOR_BGR2GRAY)
#         img_ = img.copy()
#         hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        
#         lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
#         l,a,b = cv2.split(lab)
        
#         blue_low = np.array([94,54,54]) #100,50,70
#         blue_high = np.array([147,255,255]) #blue

#         green_low = np.array([43, 135, 100]) #43, 154, 100
#         green_high = np.array([96, 255, 255]) #green 

#         red_low = np.array([0,164,170])
#         red_high = np.array([179,255,255]) #red
        
#         # red,green,blue
#         low_black = np.array([0])
#         high_black = np.array([107])

#         low_white = np.array([180])
#         high_white = np.array([255])
        
#         centerpoint = []
#         rgb_pixel = []

#         avg_color_block = []
#         print("detect color box")
#         # contours,centerpoint = find_contour(b,low_black,high_black,img_,centerpoint)
        
#         pre_area = 0
#         contours ,centerpoint,pre_area= find_contour(hsv,blue_low,blue_high,img_,centerpoint,pre_area) #blue
#         # contours ,centerpoint= find_contour(hsv,green_low,green_high,img_,centerpoint)
#         contours ,centerpoint,pre_area= find_contour(a,low_black,high_black,img_,centerpoint,pre_area) #green
#         # contours ,centerpoint = find_contour(hsv,red_low,red_high,img_,centerpoint)
#         contours ,centerpoint,pre_area= find_contour(a,low_white,high_white,img_,centerpoint,pre_area) # red
        
#         postBlue = findGreenBlock(centerpoint)

#         if postBlue == 'down' or postBlue =='upper':
#             #find red and green
#             x=centerpoint[1][0]-(centerpoint[2][0]-centerpoint[1][0])
#             y=centerpoint[1][1]-(centerpoint[2][1]-centerpoint[1][1])
#         elif postBlue == 'right' or postBlue == 'left':
#             x=centerpoint[1][0]-(centerpoint[0][0]-centerpoint[1][0])
#             y=centerpoint[1][1]-(centerpoint[0][1]-centerpoint[1][1])

#         side_len_x =abs(centerpoint[0][2][1] -  centerpoint[0][2][0]) 
#         side_len_y =abs(centerpoint[0][2][2] -  centerpoint[0][2][1])
#         max_side = max(side_len_x[0],side_len_y[1],side_len_x[1],side_len_y[0]) #找最大長寬
        
#         cropp = img.copy()
#         crop = cropp[x-int(max_side*(3/5)):x+int(max_side*(3/5)),y-int(max_side*(3/5)):y+int(max_side*(3/5))] #裁切
#         res = sharpen(crop) #影像增強
        
#         contrast = 50 #增強對比度
#         brightness = 0
#         res_contrast = res * (contrast/127 + 1) - contrast + brightness # 轉換公式
#         # 轉換公式參考 https://stackoverflow.com/questions/50474302/how-do-i-adjust-brightness-contrast-and-vibrance-with-opencv-python

#         # 調整後的數值大多為浮點數，且可能會小於 0 或大於 255
#         # 為了保持像素色彩區間為 0～255 的整數，所以再使用 np.clip() 和 np.uint8() 進行轉換
#         res_contrast = np.clip(res_contrast, 0, 255)
#         res_contrast = np.uint8(res_contrast)

#         # bin = equimage(res) #做二值化+均值強化:灰階
#         bin = getEdgeline(res_contrast) #做二值化

#         bin[bin==255]=1
#         skeleton0=morphology.skeletonize(bin)
#         bin = skeleton0.astype(np.uint8)*255 #做骨架化
#         print("detect line")
        
#         line_array,line_array0,most_angle = findLines(bin,res,minlen=8) #binary:equ or bin
#         if line_array == -1 and line_array0 == -1 and most_angle == -1:
#             score.append([name,'-1','-1','-1'])
#             continue
#         result_rotated_bin,result_rotated =bin,res
        
#         tmp = 0 # 可以重複做幾次
#         while((most_angle <=88 or most_angle >=92) and tmp < 5) :#重複找斜率為0的
#             if most_angle >= 92:
#                 most_angle %=91
#                 most_angle *= -1
#             else:
#                 most_angle %=91
#             # print('1:',most_angle)
#             result_rotated_bin,result_rotated = rotated_angle(result_rotated_bin,result_rotated,-most_angle)
#             msk = rotated_rgb_angle(msk,-most_angle) #旋轉msk  
#             result_rotated_bin = np.uint8(result_rotated_bin)
#             result_rotated_bin = sharpen(result_rotated_bin,300)
#             result_rotated_bin = skeleton(result_rotated_bin) 
#             line_array,line_array0,most_angle = findLines(result_rotated_bin,result_rotated,minlen=4,maxgap=12)
#             if line_array == -1 and line_array0 == -1 and most_angle == -1:
#                 score.append([name,'-1','-1','-1'])
#                 continue
#             if most_angle == 0:
#                 break
#             tmp +=1 
        
#         if tmp <5:
#             # vertical_array = findVertialLine(result_rotated,line_array0) #找中垂線
#             # point_posit = getFrequencyLine(vertical_array,result_rotated) #找pixel/uint
#             point_posit = getFrequencyLine_(result_rotated) #找pixel/uint
#             ratio_pixel_cm = point_posit[0][1]* point_posit[0][1]
#             real_area = round(int(point_posit[0][2])/10 * int(point_posit[0][2])/10,3)
            
#             print('point_posit',point_posit,real_area/ratio_pixel_cm)
#             #計算面積
#             print("calculate area",point_posit)
#             if point_posit :
#                 objectPixel,contour_area = getMask(msk)
#                 if point_posit[0][0] != -1:
#                     realArea = calculateArea(point_posit,objectPixel)
#                     realContourArea = calculateCountourArea(point_posit,contour_area)
                    
#                     print("final area",objectPixel,realArea)
#                     print("contour area",realContourArea,contour_area)
#                     score.append([name,objectPixel,realArea,realContourArea])
#                 else:
#                     score.append([name,'-1','-1','-1'])
#             else:
#                 score.append([name,'-1','-1','-1'])
#         else:
#             score.append([name,'-1','-1','-1'])
        
#     df = pd.DataFrame(score, columns=["Image","objectPixely","realArea","realContourArea"])
#     df.to_csv(fr"C:\Users\buslab\OneDrive - 國立成功大學 National Cheng Kung University\桌面\time series\250\SCORE_.csv",index=False)
        