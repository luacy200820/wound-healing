from flask import Flask,render_template,request,jsonify,send_from_directory
from werkzeug.utils import secure_filename
from datetime import timedelta
from tensorflow.keras.utils import CustomObjectScope
import segmentation_models as sm
import tensorflow as tf 
import os 
import cv2
import numpy as np 
import time 
from image_autoroate_v2 import *
from measure_wound_rotate2 import * 
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.dirname(__file__)



# def test():
#     return "test web"

ALLOWED_EXTENSIONS = set(['png','jpg','JPG','PNG'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS

app.send_file_max_age_default = timedelta(seconds=2)

# @app.route('/home',methods=['POST','GET'])
@app.route('/',methods=['POST','GET'])
def upload():
    if request.method == "POST":
        f = request.files['file']
        if not (f and allowed_file(f.filename)):
            return jsonify({"effor":1001,"msg":"圖片類型: png, jpg, PNG, JPG"})
        filename = secure_filename(f.filename)
        upload_path = os.path.join(app.config["UPLOAD_FOLDER"],'static\\output',filename)
        f.save(upload_path)
        print("filename",upload_path,filename)
        try:
            rotate(upload_path,filename)
            upload_path = app.config["UPLOAD_FOLDER"]+'\\static\\output\\'+filename
        except:
            print("except")
            upload_path = os.path.join(app.config["UPLOAD_FOLDER"],'static\\output',filename)
 
        # upload_path = os.path.join(app.config["UPLOAD_FOLDER"],'static\\ouput\\rotate_',filename)
        
        print("upload_path",upload_path,filename)
        result = predict_wound(upload_path,filename) #predict wound
        maskname = app.config["UPLOAD_FOLDER"]+'\\static\\output\\bin_'+filename
        area_data = calcuate_area(upload_path,maskname,name)
        print(area_data)
        if area_data[0][0] == 'nan':
            real = "無法判斷"
        else:
            real_num = round(area_data[0][1],2) 
            real = str(real_num ) + " (cm^2)"
        # print(real)
        and_upload_path = app.config["UPLOAD_FOLDER"]+'\\static\\output\\and_'+filename
        result,tissue_area = predict(upload_path,and_upload_path,filename) #predict tissue
        if real == "無法判斷":
            n_area = str(tissue_area[0]) +" %"
            s_area =str( tissue_area[1]) +" %"
            g_area = str(tissue_area[2]) +" %"
        else:
            n_area = str(round(real_num*tissue_area[0],3)) +" (cm^2)"
            s_area = str(round(real_num*tissue_area[1],3))+" (cm^2)"
            g_area = str(round(real_num*tissue_area[2],3))+" (cm^2)"

        
        filename_result = result.split('\\')[-1]
        # filename = secure_filename(result)
        print("result filename",filename)
        return render_template('home.html',outputImageName = filename_result,imagename = filename,area= real, tissue_area_nschar=n_area,tissue_area_slough=s_area,tissue_area_granulation=g_area)
    return render_template('home.html')

def result_mask(result,classes=4):
    # w,h = result.shape[:2]
    result_rgb = np.zeros((256,256,3),dtype=np.uint8)
    n = 0
    s = 0
    g = 0
    for i in range(256):
        for j in range(256):
            if classes == 4:
                max_value = max(result[i,j,0], result[i,j,1],result[i,j,2],result[i,j,3])
                # max_value = max(result[i,j,0], result[i,j,1])
                if max_value == result[i,j,0]:
                    result_rgb[i,j,0] = 0
                    result_rgb[i,j,1] = 0
                    result_rgb[i,j,2] = 0
                elif max_value == result[i,j,1]:
                    result_rgb[i,j,0] = 1
                    result_rgb[i,j,1] = 0
                    result_rgb[i,j,2] = 0
                    n +=1
                elif max_value == result[i,j,2]:
                    result_rgb[i,j,0] = 0
                    result_rgb[i,j,1] = 1
                    result_rgb[i,j,2] = 1
                    s +=1
                elif max_value == result[i,j,3]:
                    result_rgb[i,j,0] = 0
                    result_rgb[i,j,1] = 0
                    result_rgb[i,j,2] = 1 
                    g+=1
            elif classes ==2:
                max_value = max(result[i,j,0], result[i,j,1])
                # max_value = max(result[i,j,0], result[i,j,1])
                if max_value == result[i,j,0]:
                    result_rgb[i,j,0] = 0
                    result_rgb[i,j,1] = 0
                    result_rgb[i,j,2] = 0
                elif max_value == result[i,j,1]:
                    result_rgb[i,j,0] = 1
                    result_rgb[i,j,1] = 1
                    result_rgb[i,j,2] = 1
    if classes == 4:        
        return result_rgb,n,s,g
    elif classes ==2:
        return result_rgb

def predict_wound(filename,name):
    print("predict",filename)
    
    img = cv2.imread(filename)
    img = cv2.resize(img,(256,256))
    
    
    img_ = img
    img = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    img = img /255
    model = sm.Linknet("efficientnetb7",classes = 2, input_shape = (256,256,3),activation='softmax',encoder_freeze=False)
    model.load_weights('linknet_efficientnetb7_4_AUG25_cie_no_class=2_shape=3,256_combo.h5')
    pred = model.predict(img.reshape(1,256,256,3))[0]
    output_image = result_mask(pred,classes=2)
    dst = app.config["UPLOAD_FOLDER"]+'\\static\\output\\overlap_'+name
    print(img_.shape,output_image.shape)
    mix_img = cv2.addWeighted(img_,0.7,output_image*255,0.3,5)

    cv2.imwrite(dst,mix_img)
    # 生成分割出的結果
    dst = app.config["UPLOAD_FOLDER"]+'\\static\\output\\bin_'+name
    cv2.imwrite(dst,output_image*255) 
    # 結果與原始影像合併，留下交集的傷口
    print(img_.shape,output_image.shape)
    overlay = cv2.bitwise_and(img_,output_image*255)
    dst = app.config["UPLOAD_FOLDER"]+'\\static\\output\\and_'+name
    cv2.imwrite(dst,overlay) 
 
    print("finish")
 
    return dst
    
def predict(oriname,filename,name):
    print("predict")
    # name = filename.split('\\')[-1]
    # name = partName
    start = time.time()
    img = cv2.imread(filename)
    
    img = cv2.resize(img,(256,256))
    ori = cv2.imread(oriname)
    ori = cv2.resize(ori,(256,256))
    img_ = img
    img = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    img = img /255.
    with CustomObjectScope({ 'f1-score': sm.metrics.f1_score,'iou_score':sm.metrics.iou_score}):
        model = tf.keras.models.load_model('deeplabv3CIENO.h5')
    
    pred = model.predict(img.reshape(1,256,256,3))[0]
    output_image,n,s,g = result_mask(pred,classes=4)
    total_pix = n+s+g
    n_p = round(n/total_pix,3)
    s_p = round(s/total_pix,3)
    g_p = round(g/total_pix,3)
    area = [n_p,s_p,g_p]
    end = time.time()
    print("time spend",str(end-start))
    # upload_path = os.path.join(app.config["UPLOAD_FOLDER"],'static\\output',output_image)
    dst = app.config["UPLOAD_FOLDER"]+'\\static\\output\\tissue_'+name
    print(img_.shape,output_image.shape)
    mix_img = cv2.addWeighted(ori,0.7,output_image*255,0.3,5)

    cv2.imwrite(dst,mix_img)

    print(dst)
    print("finish")
    # return render_template('home.html',outputImageName = dst)
    return dst,area

def rotate(filename,name):
    image = cv2.imread(filename)
    image = cv2.resize(image,(600,600))
    img_ = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.bilateralFilter(gray,5,75,75)
    laplacian_crop = cv2.Laplacian(blur_img,cv2.CV_64F)
    th,binary = cv2.threshold(laplacian_crop,5,255,cv2.THRESH_BINARY)
    # plt.imshow(binary,cmap='gray')
    binary = np.uint8(binary)
    # Detect lines
    rho = 1
    theta = np.pi/180
    thresh = 170
    lines = cv2.HoughLines(binary, rho, theta, thresh)
    
     # Draw all Hough lines in red
    img_with_all_lines = np.copy(image)
    img_with_all_lines_ = np.copy(image)
    # print('line',lines)
    filter_lines = drawLines(img_with_all_lines, lines) #處理的
    filter_lines_ = drawLines_(img_with_all_lines_, lines)
    
    segmented = segment_by_angle_kmeans(lines, 2)
    intersections,inter_point = segmented_intersections_point(segmented)
    # print(inter_point)

    img_with_segmented_lines = np.copy(image)
    # Draw vertical lines in green
    vertical_lines = segmented[1]
    img_with_vertical_lines = np.copy(image)
    drawLines(img_with_segmented_lines, vertical_lines, (0,255,0))
    # Draw horizontal lines in yellow
    horizontal_lines = segmented[0]
    img_with_horizontal_lines = np.copy(image)
    drawLines(img_with_segmented_lines, horizontal_lines, (0,255,255))
    # for point in intersections:
    #     pt = (point[0][0], point[0][1])
    #     length = 5
    #     cv2.line(img_with_segmented_lines, (pt[0], pt[1]-length), (pt[0], pt[1]+length), (255, 0, 255), 1) # vertical line
    #     cv2.line(img_with_segmented_lines, (pt[0]-length, pt[1]), (pt[0]+length, pt[1]), (255, 0, 255), 1)
        
    h,w= image.shape[:2]
    # print(h,w)
    save_position = []
    test = image.copy()
    test2 = image.copy()
    save_position = drawLines_single(test,horizontal_lines[0],save_position) #垂直
    save_position = drawLines_single(test2,vertical_lines[0],save_position) #水平

    print('save_position',save_position) #前面為垂直，後面為水平
    # position_num1,position_num2 = line_equation(save_position[0][0],save_position[0][1],w,h,'horizontal')
    position_num1,position_num2,dir_line= line_equation(save_position[1][0],save_position[1][1],w,h) #隨便一條
    """
    不一定是垂直或水平@@
    """
    negative_1 = False
    negative_2 = False #順時鐘轉為負
    # position_ver_num1,position_ver_num2,dir_line2 = line_equation(save_position[0][0],save_position[0][1],w,h)
    print("position_num1",position_num1,position_num2,dir_line)
    # if position_num1 == 0 and position_num2 == 0:
    #     continue
    theta = find_angle(position_num1,position_num2,test,dir_line)
    line_point = []
    # 判斷是水平還是垂直
    if dir_line == 'horizontal':
        if position_num2 > position_num1:
            negative_1 = True
        line_point.append([position_num1,0,position_num2,h])
        line_point.append([position_num1,0,position_num1,h])
    elif dir_line == 'vertial':
        if position_num2 < position_num1:
            negative_2 = True
        line_point.append([0,position_num1,w,position_num2])
        line_point.append([0,position_num1,w,position_num1])

    # print(line_point)
    re1 = angle(line_point[0],line_point[1]) # 計算角度

    # ===================
    position_num1,position_num2,dir_line= line_equation(save_position[0][0],save_position[0][1],w,h)
    print("position_num1",position_num1,position_num2,dir_line)
    # if position_num1 == 0 and position_num2 == 0:
    #     continue
    theta = find_angle(position_num1,position_num2,test,dir_line)
    line_point = []
    if dir_line == 'horizontal':
        line_point.append([position_num1,0,position_num2,h])
        line_point.append([position_num1,0,position_num1,h])
    elif dir_line == 'vertial':
        line_point.append([0,position_num1,w,position_num2])
        line_point.append([0,position_num1,w,position_num1])

    # print(line_point)
    re2 = angle(line_point[0],line_point[1])
    
    
    print("ori",re1,re2)
    if negative_1 == True:
        re1 = -re1
    if negative_2 == True:
        re2 = -re2 

    if re1 < 0:
        angle_mean = re1
    elif re2 < 0:
        angle_mean = re2 
    elif re1 ==0:
        angle_mean = re2
    elif re2 == 0:
        angle_mean = re1 
    else:
        # angle_mean = max(re1,re2)
        angle_mean = int((re1+re2)/2)
        if re1 < re2 : 
            angle_mean = -1*angle_mean 
            
    print("re1,2:",re1,re2,angle_mean)
    # ===================
    h,w = image.shape[:2]
    center = (w//2,h//2)
    rotated = image.copy()
    M = cv2.getRotationMatrix2D(center,(angle_mean),1.0)
    rotated = cv2.warpAffine(rotated,M,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)

    dst = app.config["UPLOAD_FOLDER"]+'\\static\\output\\'+name
    cv2.imwrite(dst,rotated)

def calcuate_area(filename,maskname,name):
    #filename: detect ruler, maskname : segment after binary image
    img = cv2.imread(filename)
    msk = cv2.imread(maskname)
    
    img = cv2.resize(img,(600,600))
    msk =  cv2.resize(msk,(600,600))

    # msk = cv2.cvtColor(msk,cv2.COLOR_BGR2GRAY)
    img_ = img.copy()
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    img_mean = img.mean()
    lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    
    blue_low = np.array([94,54,54]) #100,50,70
    blue_high = np.array([147,255,255]) #blue

    green_low = np.array([43, 135, 100]) #43, 154, 100
    green_high = np.array([96, 255, 255]) #green 

    red_low = np.array([0,164,170])
    red_high = np.array([179,255,255]) #red
    print("mean",img_mean)
    # red,green,blue
    if img_mean > 148:
        low_black = np.array([0])
        high_black = np.array([107])
    else:
        low_black = np.array([0])
        high_black = np.array([115])

    low_white = np.array([180])
    high_white = np.array([255])
    
    centerpoint = []
    rgb_pixel = []
    score =[]

    avg_color_block = []
    print("detect color box")
    # contours,centerpoint = find_contour(b,low_black,high_black,img_,centerpoint)
    
    pre_area = 0
    contours ,centerpoint,pre_area= find_contour(hsv,blue_low,blue_high,img_,centerpoint,pre_area) #blue
    # contours ,centerpoint= find_contour(hsv,green_low,green_high,img_,centerpoint)
    contours ,centerpoint,pre_area= find_contour(a,low_black,high_black,img_,centerpoint,pre_area) #green
    # contours ,centerpoint = find_contour(hsv,red_low,red_high,img_,centerpoint)
    contours ,centerpoint,pre_area= find_contour(a,low_white,high_white,img_,centerpoint,pre_area) # red
    
    print("centerpoint",centerpoint)
    if len(centerpoint) >2:
        
        postBlue = findGreenBlock(centerpoint)
        print("postBlue",postBlue)

        if postBlue == 'down' or postBlue =='upper':
            #find red and green
            x=centerpoint[1][0]-(centerpoint[2][0]-centerpoint[1][0])
            y=centerpoint[1][1]-(centerpoint[2][1]-centerpoint[1][1])
        elif postBlue == 'right' or postBlue == 'left':
            x=centerpoint[1][0]-(centerpoint[0][0]-centerpoint[1][0])
            y=centerpoint[1][1]-(centerpoint[0][1]-centerpoint[1][1])

        side_len_x =abs(centerpoint[0][2][1] -  centerpoint[0][2][0]) 
        side_len_y =abs(centerpoint[0][2][2] -  centerpoint[0][2][1])
        max_side = max(side_len_x[0],side_len_y[1],side_len_x[1],side_len_y[0]) #找最大長寬
        
        cropp = img.copy()
        crop = cropp[x-int(max_side*(3/5)):x+int(max_side*(3/5)),y-int(max_side*(3/5)):y+int(max_side*(3/5))] #裁切
        res = sharpen(crop) #影像增強
        
        contrast = 50 #增強對比度
        brightness = 0
        res_contrast = res * (contrast/127 + 1) - contrast + brightness # 轉換公式
        # 轉換公式參考 https://stackoverflow.com/questions/50474302/how-do-i-adjust-brightness-contrast-and-vibrance-with-opencv-python

        # 調整後的數值大多為浮點數，且可能會小於 0 或大於 255
        # 為了保持像素色彩區間為 0～255 的整數，所以再使用 np.clip() 和 np.uint8() 進行轉換
        res_contrast = np.clip(res_contrast, 0, 255)
        res_contrast = np.uint8(res_contrast)

        # bin = equimage(res) #做二值化+均值強化:灰階
        bin = getEdgeline(res_contrast) #做二值化

        bin[bin==255]=1
        skeleton0=morphology.skeletonize(bin)
        bin = skeleton0.astype(np.uint8)*255 #做骨架化
        print("detect line")
        
        line_array,line_array0,most_angle = findLines(bin,res,minlen=8) #binary:equ or bin
        if line_array == -1 and line_array0 == -1 and most_angle == -1:
            score.append(['-1','-1','-1'])
            
        result_rotated_bin,result_rotated =bin,res
        
        tmp = 0 # 可以重複做幾次
        while((most_angle <=88 or most_angle >=92) and tmp < 5) :#重複找斜率為0的
            if most_angle >= 92:
                most_angle %=91
                most_angle *= -1
            else:
                most_angle %=91
            # print('1:',most_angle)
            result_rotated_bin,result_rotated = rotated_angle(result_rotated_bin,result_rotated,-most_angle)
            msk = rotated_rgb_angle(msk,-most_angle) #旋轉msk  
            result_rotated_bin = np.uint8(result_rotated_bin)
            result_rotated_bin = sharpen(result_rotated_bin,300)
            result_rotated_bin = skeleton(result_rotated_bin) 
            line_array,line_array0,most_angle = findLines(result_rotated_bin,result_rotated,minlen=4,maxgap=12)
            if line_array == -1 and line_array0 == -1 and most_angle == -1:
                score.append(['-1','-1','-1'])
                
            if most_angle == 0:
                break
            tmp +=1 
        
        if tmp <5:
            # vertical_array = findVertialLine(result_rotated,line_array0) #找中垂線
            # point_posit = getFrequencyLine(vertical_array,result_rotated) #找pixel/uint
            point_posit = getFrequencyLine_(result_rotated) #找pixel/uint
            ratio_pixel_cm = point_posit[0][1]* point_posit[0][1]
            real_area = round(int(point_posit[0][2])/10 * int(point_posit[0][2])/10,3)
            
            print('point_posit',point_posit,real_area/ratio_pixel_cm)
            #計算面積
            print("calculate area",point_posit)
            if point_posit :
                objectPixel,contour_area = getMask(msk)
                if point_posit[0][0] != -1:
                    realArea = calculateArea(point_posit,objectPixel)
                    realContourArea = calculateCountourArea(point_posit,contour_area)
                    
                    print("final area",objectPixel,realArea)
                    print("contour area",realContourArea,contour_area)
                    score.append([objectPixel,realArea,realContourArea])
                else:
                    score.append(['-1','-1','-1'])
            else:
                score.append(['-1','-1','-1'])
        else:
            score.append(['-1','-1','-1'])
        return score
    
    else:
        return [['nan']]
        

    
# @app.route('/upload/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config["UPLOAD_FOLDER"],filename)
if __name__ == "__main__":
    app.run(host="0.0.0.0",debug = True )
