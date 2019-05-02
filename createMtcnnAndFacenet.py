"""
五步走战略之2nd与3rd: 
    2nd--Mtcnn人脸库
    3rd--FaceNet人脸库
另见face_recognition.py

"""
#系统库
import numpy as np
import cv2
import os

#自定义库,详情见当前文件所在目录
from utils import image_processing , file_processing
import face_recognition


resize_width = 160
resize_height = 160

def create_face(images_dir, out_face_dir):
    '''
    生成人脸数据图库，保存在out_face_dir中，这些数据库将用于生成embedding数据库
    '''
    #此处引用自定义函数 : file_processing.gen_files_labels(images_dir,postfix='jpg')
    #返回值:   图像数组:  images_dir下所有文件(是的,你没想错!包括子目录下的所有文件)的路径(Path)会被存储到image_list
    #         标签数组:  names_list就是相应的的"文件名或子目录名"，在本项目中,子目录名 将作为 作为样本的标签(即ID,e.g.姓名 或 学号之类的)

    print('#2nd--Mtcnn人脸库')

    image_list,names_list=file_processing.gen_files_labels(images_dir,postfix='jpg')
    
    face_detect=face_recognition.Facedetection()
    
    for image_path ,name in zip(image_list,names_list):
        
        # try:
        image=image_processing.read_image(image_path, resize_height=0, resize_width=0, normalization=False)
        
        # 获取 判断标识 bounding_box crop_image
        bounding_box, points = face_detect.detect_face(image)
        bounding_box = bounding_box[:,0:4].astype(int)
        
        # try:
        bounding_box=bounding_box[0,:]
        # except:
        # print("跳过当前图片")
            # continue
        
        print("矩形框坐标为:{}".format(bounding_box))
        
        face_image = image_processing.crop_image(image,bounding_box)
        # except:
            # print("当前图像帧格式非法，将被忽略")
        
        out_path=os.path.join(out_face_dir,name)
        
        face_image=image_processing.resize_image(face_image, resize_height, resize_width)
        
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        
        basename=os.path.basename(image_path)

        out_path=os.path.join(out_path,basename)

        image_processing.save_image(out_path,face_image)
        
        # cv2.waitKey(0)

def create_embedding(model_path, emb_face_dir, out_emb_path, out_filename):
    '''
    产生embedding数据库，这些embedding其实就是人脸特征
    '''
    print('#3rd--FaceNet人脸库')

    face_net = face_recognition.facenetEmbedding(model_path)

    image_list,names_list=file_processing.gen_files_labels(emb_face_dir,postfix='jpg')
    
    images= image_processing.get_images(image_list,resize_height,resize_width,whiten=True)
    
    compare_emb = face_net.get_embedding(images)
    
    np.save(out_emb_path, compare_emb)


    file_processing.write_data(out_filename, names_list, model='w')


if __name__ == '__main__':
    
    #2nd--Mtcnn人脸库
    images_dir='dataset/images'#待处理数据集
    out_face_dir='dataset/emb_face'
    create_face(images_dir,out_face_dir)
    
    #3rd--FaceNet人脸库
    model_path = 'models/20180408-102900'# 调用 GoogleFaceNet系统库模型 ,类似于 OpencvHaarCascade系统库分类器
    emb_face_dir = './dataset/emb_face'#输入MTCNN处理后所得到的数据集
    out_emb_path = 'dataset/emb/faceEmbedding.npy'
    out_filename = 'dataset/emb/name.txt'

    #传入参数，开始运行
    create_embedding(model_path, emb_face_dir, out_emb_path, out_filename)