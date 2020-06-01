import cv2
import numpy as np
import os
import struct

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

images,labels = load_mnist('MNIST_data')

p = 0.01 #学习率
errorcount = 0
count = 0
w = np.random.rand(1,785)
nums = [2,8] #进行二分类的数据

for i in range(60000):
    data = np.append([1],images[i])
    if(labels[i]==nums[0]):
        if np.dot(w,data.T)<0:
            w = w + p*data
    elif(labels[i]==nums[1]):
        if np.dot(w,data.T)>0:
            w = w - p*data

testimages,testlabels = load_mnist('MNIST_data','t10k')
for i in range(10000):
    data = np.append([1],testimages[i])
    if(testlabels[i]==nums[0]):
        count = count +1
        if np.dot(w,data.T)<0:
            errorcount=errorcount+1
    elif(testlabels[i]==nums[1]):
        count = count +1
        if np.dot(w,data.T)>0:
            errorcount=errorcount+1


rate = errorcount/count
print("error rate: %f" % rate)
