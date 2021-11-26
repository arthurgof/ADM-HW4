import numpy as np
import random

class kmean:
    def __init__(self, data) -> None:
        self.data = self.normalize(np.array(data))
        self.p1set = []
        self.p2set = []
    
    def normalize(self, data):
        norm = np.linalg.norm(data)
        if norm == 0: 
           return data
        return data / norm

    def giveVAlue(self, data):
        self.data = data
    
    def initalizeRandom(self, kn):
        self.point = np.random.random((kn, len(self.data[0])))
    
    def assignk(self, first):
        va = True
        for i in range(len(self.data)):
            pt = np.array(self.data[i])
            if first:
                min = 2
            else:
                min = np.linalg.norm(self.point[self.assig[i]] - pt)
            for y in range(len(self.point)):
                k = self.point[y]
                if np.linalg.norm(k - pt) < min:
                    va = False
                    min = np.linalg.norm(k - pt)
                    self.assig[i] = y
        return va


    def fit(self, kn):
        self.initalizeRandom(kn)
        move = True
        self.assig = np.zeros((len(self.data)), int)
        self.assignk(True)
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
            move = self.assignk(False)
            print(self.assig)