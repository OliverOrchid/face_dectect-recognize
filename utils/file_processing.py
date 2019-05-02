# -*-coding: utf-8 -*-
"""
文件处理
"""
import os

#写入数据
def write_data(file, content_list, model):
    with open(file, mode=model) as f:  #参见Python学习笔记-P1 ,以指定模式(model)打开文件(file),同时创建了"文件对象"(file),并将file简记作"f"
                                       #这也意味着,file如果之前有数据,将会被擦除并被改写
        for line in content_list:  #for-in 遍历 目标content_list
            f.write(line + "\n") #f调用write函数,将content_list的所有内容(PS:包括文件 文件夹)  写入到 "f"(即:file)


#读取数据
def read_data(file):
    with open(file, mode="r") as f: #以只读模式打开文件(file)
        content_list = f.readlines()#逐行读取样本信息，存入到list
        content_list = [content.rstrip() for content in content_list]#除去每个样本的 首或尾的空白字符
    return content_list

    
#获取file_dir目录下，所有文本路径，包括子目录文件
def getFilePathList(file_dir):

    filePath_list = []
    for walk in os.walk(file_dir):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_list.extend(part_filePath_list)
    return filePath_list



#获得file_dir目录下，后缀名为postfix所有文件列表，包括子目录
def get_files_list(file_dir, postfix='ALL'):

    postfix = postfix.split('.')[-1]
    file_list = []
    filePath_list = getFilePathList(file_dir)
    if postfix == 'ALL':
        file_list = filePath_list
    else:
        for file in filePath_list:
            basename = os.path.basename(file)  # 获得路径下的文件名
            postfix_name = basename.split('.')[-1]
            if postfix_name == postfix:
                file_list.append(file)
    file_list.sort()
    return file_list



# 获取files_dir路径下所有文件路径，以及labels,其中labels用子级文件名表示
# files_dir目录下，同一类别的文件放一个文件夹，其labels即为文件的名
# filePath_list所有文件的路径,label_list对应的labels

def gen_files_labels(files_dir,postfix='ALL'):

    # 文件路径
    filePath_list=get_files_list(files_dir, postfix=postfix)
    print("训练集照片数量:{}".format(len(filePath_list)))
    
    # 获取所有样本标签Label
    label_list = []
    for filePath in filePath_list:
        label = filePath.split(os.sep)[-2]
        label_list.append(label)

    labels_set = list(set(label_list))
    print("人物标签:{}".format(labels_set))

    return filePath_list, label_list