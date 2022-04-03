# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 09:16:38 2021

@author: hrb12
"""

import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
# import pcl
# import pcl.pcl_visualization



cameraMatrix1 = np.array([   [915.583314922979, 0.0, 486.156216093482],
                                            [0.0, 913.176127266096, 317.345300339797],
                                            [0.0, 0.0, 1.0]
                                        ])
cameraMatrix2 = np.array([   [918.360165765321, 0.0, 491.577227643437],
                                            [0.0, 916.33382965125, 321.06530759765,],
                                            [0.0, 0.0, 1.0]
                                        ])

distCoeffs1 =  np.array([-0.319132360976366, 0.27814804841972, 0.0, 0.0, -0.272034910182435])


distCoeffs2 =  np.array([-0.300021351338835, 0.178425245713208, 0.0, 0.0, -0.145246493919359 ])
w, h = 960,604

P1 = np.array([   [914.754978458673, 0.0, 500.581356048584, 0.0],
                                            [0.0, 914.754978458673, 323.8738555908203, 0.0],
                                            [0.0, 0.0, 1.0, 0.0]
                                        ])

P2 = np.array([   [914.754978458673, 0.0, 500.581356048584, -457.6941777394],
                                            [0.0, 914.754978458673, 323.8738555908203, 0.0],
                                            [0.0, 0.0, 1.0, 0.0]
                                        ])
R1 = np.array([   [0.9996897304359291, -0.01599848225199777, -0.01909165856002074],
                                            [0.01595237875765353, 0.999869463407055, -0.002564714030888407],
                                            [0.01913019793186326, 0.002259360909722831, 0.9998144481929471]
                                        ])


R2 = np.array([   [0.9999619043475868, -0.006242203948765858, 0.006101208356551847],
                                            [0.006227026493007892, 0.9999774781767194, 0.002503453691135995],
                                            [-0.006116698014731802, -0.002465365934358731, 0.9999782537516534]
                                        ])
R = np.array([   [1, 4.90931e-15, -1.47775e-14],
                                            [-4.90931e-15, 1,-9.52108e-15],
                                            [1.47775e-14, 9.52108e-15, 1]
                                        ])
T = np.array([[-0.500346], [1.21295e-31], [0]])



def stereoMatchSGBM(left_image, right_image, down_scale=False):
    # SGBM匹配参数设置
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 5
    paraml = {'minDisparity': 1,
             'numDisparities': 128,
             'blockSize': blockSize,
             'P1': 8 * img_channels * blockSize ** 2,
             'P2': 32 * img_channels * blockSize ** 2,
             'disp12MaxDiff': -1,
             'preFilterCap': 64,
             'uniquenessRatio': 5,  # (5~15)
             'speckleWindowSize': 200,  #(50-200)
             'speckleRange':5, #(1,2)
              'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
              # 'mode':cv2.STEREO_SGBM_MODE_HH
             }

    # 构建SGBM对象
    left_matcher = cv2.StereoSGBM_create(**paraml)
    paramr = paraml
    paramr['minDisparity'] = -paraml['numDisparities']
    right_matcher = cv2.StereoSGBM_create(**paramr)

    # 计算视差图
    size = (left_image.shape[1], left_image.shape[0])
    if down_scale == False:
        disparity_left = left_matcher.compute(left_image, right_image)
        disparity_right = right_matcher.compute(right_image, left_image)

    else:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        factor = left_image.shape[1] / left_image_down.shape[1]

        disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
        disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
        disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
        disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left = factor * disparity_left
        disparity_right = factor * disparity_right

    # 真实视差（因为SGBM算法得到的视差是×16的）
    trueDisp_left = disparity_left.astype(np.float32) / 16.
    trueDisp_right = disparity_right.astype(np.float32) / 16.

    return trueDisp_left, trueDisp_right


def preprocess(img1, img2):
    
    # 彩色图->灰度图
    if(img1.ndim == 3):#判断为三维数组
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 通过OpenCV加载的图像通道顺序是BGR
    if(img2.ndim == 3):
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 直方图均衡
    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)

    return img1, img2



def depthTry(dismap,fx,baseline):
    (height,width) = dismap.shape
    depthmap = np.zeros((height,width))
    for y in range(height):
        for x in range(width):
            if dismap[y][x] != 0:
                depthmap[y][x] = fx * baseline/dismap[y][x]
    return depthmap
            
    
    
    
def getCorrectImage(img0,img1):

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = \
        cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, (w, h), R, T)

    map1_1, map1_2 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (w, h), cv2.CV_16SC2)
    map2_1, map2_2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (w, h), cv2.CV_16SC2)

   
    result1 = cv2.remap(img0, map1_1, map1_2, cv2.INTER_LINEAR)
    result2 = cv2.remap(img1, map2_1, map2_2, cv2.INTER_LINEAR)


    result = np.concatenate((result1, result2), axis=1)
    result[::20, :] = 0
    cv2.imwrite("rec.png", result)
    return result1,result2,Q
    

def discalculation(imgl,imgr):
    
    iml, imr ,Q = getCorrectImage(imgl,imgr)
    iml_, imr_ = preprocess(iml, imr)
    disp,_ = stereoMatchSGBM(iml, imr)
    depth = depthTry(disp,Q[2,3],0.500346)
    return depth


def blankImage(h,w):
    image = np.zeros((h,w))  # 两圆之间 20 px，
    # Ellipse parameters
    center = (w // 2, h)
    dis = h//20
    lenth = h//dis
    step = 255//lenth
    k = w
    for i in range(0,255,step):
        if k >0:
            cv2.circle(image, (center[0], center[1]),k, i, -1)
            k -= dis
    return image
  

def angleCalculate(x,y):
    x=np.array(x)
    y=np.array(y)
    # 两个向量
    Lx=np.sqrt(x.dot(x))
    Ly=np.sqrt(y.dot(y))
    #相当于勾股定理，求得斜线的长度
    cos_angle=x.dot(y)/(Lx*Ly)
    #求得cos_sita的值再反过来计算，绝对长度乘以cos角度为矢量长度，初中知识。。
    # return cos_angle
    angle=np.arccos(cos_angle)
    
    return angle



def viewplot(x,y,dis,view_h,view_w,img_view):
    # creatw view image   
    # shift point to center
    ang = angleCalculate([ x- 480,604 - y], [480,0])
    try:
        real_x = int(dis * np.cos(ang))
        real_y = int(dis * np.sin(ang))
    except ValueError:
        return
    if ang < np.pi/2:
        view_x = real_x + view_w//2
    else:
        view_x = view_w//2 - real_x
    view_y = view_h - real_y
    cv2.rectangle(img_view, (view_x-4,view_y-10), (view_x+4,view_y+10), (0,255,0), 2)



