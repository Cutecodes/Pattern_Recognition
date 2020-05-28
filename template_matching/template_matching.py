import cv2
import numpy as np


class DataObj:
    def __init__(self,data,label):
        self._data = data
        self._label = label

template = []
for i in range(10):
    src = cv2.imread("./"+str(i)+"/126.bmp")
    gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    translate = cv2.resize(gray,None,fx=0.1,fy=0.1,interpolation=cv2.INTER_CUBIC)
    data = np.array(translate)
    data.resize(1,625)
    template.append(DataObj(data,i))

TestDataObj=template[0]
Dist = 0.0
MatchLabel = 0
count = 0
wrongCount = 0

for i in range(10):
    for j in range(125):
        count=count+1
        src = cv2.imread("./"+str(i)+"/"+str(j+1)+".bmp")
        gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
        translate = cv2.resize(gray,None,fx=0.1,fy=0.1,interpolation=cv2.INTER_CUBIC)
        data = np.array(translate)
        data.resize(1,625)
        TestDataObj = DataObj(data,i)
        Dist = np.linalg.norm(TestDataObj._data - template[0]._data)
        for k in range(9):
            dist = np.linalg.norm(TestDataObj._data - template[k+1]._data)
            if dist<Dist:
                Dist = dist
                MatchLabel = k+1
        if(TestDataObj._label!=MatchLabel):
            wrongCount=wrongCount+1
            print("wrong match %d -> %d " %(TestDataObj._label,MatchLabel))
rate = wrongCount/count
print("error rate: %f" % rate)
                
                
        
        
        

