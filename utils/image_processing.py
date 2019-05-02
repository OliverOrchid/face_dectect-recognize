#encoding:utf-8
"""
图像处理
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys

def show_image(win_name, rgb_image):
    plt.title(win_name)
    plt.imshow(rgb_image)
    plt.show()


def show_image_rect(win_name, rgb_image, rect):
    plt.figure()
    plt.title(win_name)
    plt.imshow(rgb_image)
    rect =plt.Rectangle((rect[0], rect[1]), rect[2], rect[3], linewidth=2, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
    plt.show()


def show_image_boxes(win_name, rgb_image, boxes):
    plt.title(win_name)
    for box in boxes:
        cv2.rectangle(rgb_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    plt.imshow(rgb_image)
    plt.show()

def show_image_box(win_name, rgb_image, box):
    plt.title(win_name)
    cv2.rectangle(rgb_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    plt.imshow(rgb_image)
    plt.show()

def cv_show_image_text(win_name,bgr_image, boxes,boxes_name):

    for name ,box in zip(boxes_name,boxes):
        cv2.rectangle(bgr_image, (box[0],box[1]),(box[2],box[3]), (249, 204, 226), 2, 8, 0)#粉色
        cv2.putText(bgr_image,name, (box[0],box[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 255), thickness=2)#青
    cv2.imshow(win_name, bgr_image)
    cv2.waitKey(1)


def read_image(image_path, resize_height=0, resize_width=0, normalization=False):
    bgr_image = cv2.imread(image_path)
    
    # 若是灰度图则转为三通道
    if len(bgr_image.shape)==2:
        print("哎呦！发现了非法的灰度图像！！！没关系正在转换>>>", image_path)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)# 将BGR转为RGB 格式转化

    if resize_height > 0 and resize_width > 0:
        rgb_image = cv2.resize(rgb_image, (resize_width, resize_height))
    rgb_image = np.asanyarray(rgb_image)
    if normalization:
        #分母应为float
        rgb_image = rgb_image / 255.0
    return rgb_image

def save_image(image_path, rgb_image):
    # try:
    plt.imsave(image_path, rgb_image)
    # except:
    #     print("当前图像帧数据格式非法，无法裁剪该图像，此帧即将被忽略！")

def crop_image(image, box):
    crop_img= image[box[1]:box[3], box[0]:box[2]]
    return crop_img

def crop_images(image, boxes, resize_height=0, resize_width=0):

    crops=[]
    for box in boxes:
        crop_img=crop_image(image, box)
        if resize_height > 0 and resize_width > 0:
            crop_img = cv2.resize(crop_img, (resize_width, resize_height))
        crops.append(crop_img)


    return crops

def get_crop_images(image, boxes, resize_height=0, resize_width=0, whiten=False):
 
    
    crops=[]
    for box in boxes:
        crop_img=crop_image(image, box)
        if resize_height > 0 and resize_width > 0:
            try:
                crop_img = cv2.resize(crop_img, (resize_width, resize_height))
                if whiten:
                    crop_img = prewhiten(crop_img)
            except:
                print("当前帧图片已损坏！")
         
        crops.append(crop_img)

    try:
        crops=np.stack(crops)
        print("摄像头运行正常，并返回当前视频帧,图像矩阵合法!")
        return crops
    except:
        print("\n系统将跳过当前视频帧，图像矩阵是非法空值 或 格式非法！\n")
        return True

    



def resize_image(image,resize_height,resize_width):

    # try:
    image = cv2.resize(image, (resize_width, resize_height))
    return image
    # except:
    #     print("当前图像帧数据格式非法，无法裁剪该图像，此帧即将被忽略！")

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def get_images(image_list,resize_height=0,resize_width=0,whiten=False):
    images = []
    for image_path in image_list:
        # img = misc.imread(os.path.join(images_dir, i), mode='RGB')
        image=read_image(image_path)
        if resize_height > 0 and resize_width > 0:
            image = cv2.resize(image, (resize_width, resize_height))
        if whiten:
            image = prewhiten(image)
        images.append(image)
    images = np.stack(images)
    return images