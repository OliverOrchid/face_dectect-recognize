import tensorflow as tf

#@David Sandberg  导入FaceNet模型 ， 该模型 基于MIT协议发布 ； Copyright (c) 2016 David Sandberg
import facenet

#@David Sandberg 导入MTCNN模型 ， 该模型 基于MIT协议发布 ； Copyright (c) 2016 David Sandberg
import align.detect_face as detect_face

###############################################################################
# """
# MTCNN的关键参数：

# nms_threshold：非极大值抑制nms筛选人脸框时的IOU阈值，三个网络可单独设定阈值，值设置的过小
# ，nms合并的少，会产生较多冗余计算。示例nms_threshold[3] = { 0.5, 0.7, 0.7 };。

# threshold：人脸框阈值，三个网络可单独设定阈值，值设置的太小，会有很多框通过，也就增加
# 了计算量，还有可能导致最后不是人脸的框错认为人脸。示例threshold[3] = {0.8, 0.8, 0.8};

# minsize ：最小可检测图像，该值大小，可控制图像金字塔的阶层数的参数之一，越小，阶层越多，计
# 算越多。示例minsize = 40;

# factor ：生成图像金字塔时候的缩放系数, 范围(0,1)，可控制图像金字塔的阶层数的参数之一，
# 越大，阶层越多，计算越多。示例factor = 0.709;

# 输入图片的尺寸，minsize和factor共同影响了图像金字塔的阶层数。用户可根据自己的精度需求进行调控。
# """
#################################################################################
#facenet-API
class facenetEmbedding:

    def __init__(self,model_path):
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        #传入模型路径
        facenet.load_model(model_path)
        
        #填充数据到 graph中
        
        #待处理的 人脸数据
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")

        #保存特征向量
        self.tf_embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        
        #生成预训练模型
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    #传入MTCNN生成的人脸数据，获取对应特征向量
    def  get_embedding(self,images):
        
        try:
        #喂食数据到神经网络
            feed_dict = {self.images_placeholder: images, 
                            self.phase_train_placeholder: False}
        #启动Google-FaceNet模型
            embedding = self.sess.run(self.tf_embeddings, 
                            feed_dict=feed_dict)
        #返回向量
            return embedding

        except:
            print("当前帧图像数据格式错误，FaceNet模型无法处理！")



#mtcnn-API
class Facedetection:
    def __init__(self):
        #minsize ：最小可检测图像
        self.minsize = 20

        #threshold：人脸框阈值
        self.threshold = [0.6, 0.7, 0.7]
        
        #factor ：生成图像金字塔时候的缩放系数
        self.factor = 0.709
        
        print('MTCNN人脸检测与定位正在运行>>>\n')

        #调用create_mtcnn-API，利用TensorFlow搭建神经网络
        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, None)#参数解包，依次赋值给左侧，第二个参数默认值为None
    
    #传入训练集数据，返回检测并定位人脸
    def detect_face(self,image):
        bounding_boxes, points = detect_face.detect_face(image, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        return  bounding_boxes, points
