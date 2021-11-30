import numpy as np
import sys

class kmean:
    def __init__(self, data) -> None:
        self.data = self.normalize(np.array(data))
        self.p1set = []
        self.p2set = []
    
    def normalize(self, data):
        norm = np.linalg.norm(data)
        if norm == 0: 
           return data
        return data

    def giveVAlue(self, data):
        self.data = data
    
    def initalizeRandom(self, kn):
        self.point = np.random.random((kn, len(self.data[0])))
        for i in range(len(self.data[0])):
            self.point[:,i] = self.point[:,i] * np.max(self.data[:,i])

    def assignk(self, first, data):
        va = False
        for i in range(len(data)):
            pt = np.array(data[i])
            if first:
                min = sys.maxsize
            else:
                min = np.linalg.norm(self.point[self.assig[i]] - pt)
            for y in range(len(self.point)):
                k = self.point[y]
                if np.linalg.norm(k - pt) < min:
                    va = True
                    min = np.linalg.norm(k - pt)
                    self.assig[i] = y
        return va


    def fit(self, kn):
        self.initalizeRandom(kn)
        move = True
        self.assig = np.zeros((len(self.data)), int)
        self.assignk(True, self.data)
        while move:
            for i in range(len(self.point)):
                sum = np.zeros((len(self.data[0])))
                num = 0
                for y in range(len(self.assig)):
                    if i == self.assig[y]:
                        sum += self.data[y]
                        num += 1
                if num > 0:
                    self.point[i] = sum/num
            move = self.assignk(False, self.data)
        return self.assig


    def predic(self, data):
        data = np.array(data)
        self.assig = np.zeros((len(data)), int)
        self.assignk(True, data)
        return self.assig

