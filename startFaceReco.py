import cv2
from scipy import misc
import tensorflow as tf
import numpy as np
import os
from utils import file_processing,kclasser,image_processing#导入自定义模块
import face_recognition#导入自定义模块

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.externals import joblib


#识别静态图像(Still Image)中的人
def face_reco(model_path, dataset_path, filename):
    ##############################################################
    # my_knn_model = KNeighborsClassifier()
    # my_knn_model = joblib.load('./models/knn.model')#######################################################
    #TIPS
    print("人脸识别系统正在启动>>>\n")

    # 先传入 "Embedding特征向量数据集" ,再 传入标签数组
    dataset_emb,names_list=load_dataset(dataset_path, filename)

    # 构造 MTCNN人脸检测器
    face_detect=face_recognition.Facedetection()
 
    print("加载Google-Facenet模型>>>")
    # 传入系统库模型 从而 构造 "Facenet特征提取器"
    face_net=face_recognition.facenetEmbedding(model_path)
    #打开摄像头，传入待检测的 VideoFrame
    camera = cv2.VideoCapture(0)
    #见L87
    # counter = 1

    #############################################################################################################
    #############################################################################################################
    #############################################################################################################
    print("KNN>>>\n")

    print("\n加载预训练模型，即将开始人脸检测>>>\n")

    while True:
    
        read, image = camera.read()#每一张视频帧都是一个三维矩阵
        #传入待检测的 StillImage
        # image=cv2.imread("/home/oliver/test_images/05.png")#first image_path
        # image = cv2.imread('/home/oliver/test_images/001.png')

    # ########

        bgr_image = image
        resize_height=0
        resize_width=0
        # normalization=False

    # 若是灰度图（即二维矩阵）则转为三通道（即：三维矩阵）
        if len(bgr_image.shape)==2:
            print("非法的灰度图！正在转换为BGR图>>>", image)
            bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)# 将BGR转为RGB 格式转化

        if resize_height > 0 and resize_width > 0:
            rgb_image = cv2.resize(rgb_image, (resize_width, resize_height))

        rgb_image = np.asarray(rgb_image)#几何图像代数化，便于运算、分析
    # #########

    #刷新参数：宽度 和 高度
        resize_width = 160
        resize_height = 160

    # 对StillImage 进行 MTCNN人脸检测，获得bounding_box
        bounding_box, points = face_detect.detect_face(rgb_image)
        bounding_box = bounding_box[:,0:4].astype(int)


        if rgb_image is None:
            continue

    # 在StillImage上 ,基于bounding_box 进行Crop裁剪操作 
        face_images = image_processing.get_crop_images(rgb_image,bounding_box,resize_height,resize_width,whiten=True)

        if face_images is True:
            continue

    # 基于Facenet特征提取器 , 得到 摄像头实时返回的视频帧 的特征向量
        pred_emb=face_net.get_embedding(face_images)#FaceNET 特征向量
   
    # 把 StillImage特征向量  和 "Embedding特征向量数据集" 进行比较 ,最后返回一个 "数据库中的已有标签" 或者 "Unknown标签"
        pred_name=compare_embadding(pred_emb, dataset_emb, names_list)#FaceNet 标签数组

        if pred_name is True:
            continue    

    # 因为L37 把图片转化为RGB的格式 -  所以此处 还原 测试图片 为BGR图像
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    # bgr_image = cv2.resize(bgr_image,(450,550))

    # while True:



    # 调用函数,在StillImage上绘制人脸边框和 "相对应的标签"
        image_processing.cv_show_image_text("【@ZTJ-FaceReco人脸识别系统】", bgr_image, bounding_box, pred_name )

    #
        if cv2.waitKey(1) &  0xff == ord("q") :
            print("由于您按下了\"Q\"键，系统即将退出！")
            break

    #############################################################################################################
    #############################################################################################################
    #############################################################################################################
    camera.release()#释放占用的摄像头
    cv2.destroyAllWindows()#杀死窗口进程

def knn_model():

    dataset_emb,names_list=load_dataset(dataset_path, filename)
    
    #Create KNN Classifier
    knnModel= KNeighborsClassifier(n_neighbors=5)
    
    # X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)
    X_train, X_test, y_train, y_test = train_test_split(dataset_emb,names_list, test_size=0.3) # 70% training and 30% test
    
    knnModel.fit(X_train,y_train)
    predict = knnModel.predict(X_test)  
    accuracy = metrics.accuracy_score(y_test, predict)
    
    print ('accuracy: %.2f%%' % (100 * accuracy)  ) 

    #保存模型
    joblib.dump(knnModel, './models/knn.model')



# 加载人脸数据库
def load_dataset(dataset_path,filename):
    # dataset_path: 特征向量文件之路径（faceEmbedding.npy）
    # filename: 标签组文件之路径（name.txt）
    compare_emb=np.load(dataset_path)
    names_list=file_processing.read_data(filename)
    return compare_emb,names_list

def compare_embadding(pred_emb, dataset_emb, names_list):
    
    try:
        # 为bounding_box 匹配标签
        pred_num = len(pred_emb)
        dataset_num = len(dataset_emb)
        pred_name = []
        for i in range(pred_num):
            dist_list = []
            for j in range(dataset_num):
            #dist即是向量空间距离
                dist = np.sqrt(np.sum(np.square(np.subtract(pred_emb[i, :], dataset_emb[j, :]))))
                dist_list.append(dist)
            min_value = min(dist_list)
            if (min_value > 0.65):#阈值判定
                pred_name.append('OH MY GOD!!!')
            else:
                pred_name.append(names_list[dist_list.index(min_value)])
        return pred_name
    except:
        print("帧图像噪声过大，系统无法处理，将跳过当前帧！")
        return True


#建立主函数,基于主函数,调用启动函数!
if __name__=='__main__':
    
    model_path='models/20180408-102900'#传入系统库模型
    dataset_path='dataset/emb/faceEmbedding.npy'#传入 "Embedding特征向量数据集" , 
    filename='dataset/emb/name.txt'#传入标签数组
    
    #参见L21 , 调用 "启动函数" , "人脸识别程序"开始启动
    face_reco(model_path, dataset_path, filename)
