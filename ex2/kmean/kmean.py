import numpy as np
import random

class kmean:
    def __init__(self, data) -> None:
        self.data = self.normalize(np.array(data))
    
    def normalize(self, data):
        norm = np.linalg.norm(data)
        if norm == 0: 
           return data
        return data / norm

    def giveVAlue(self, data):
        self.data = data
    
    def initalizeRandom(self,kn):
        shape = self.data.shape
        print(shape)
        self.point = np.zeros((kn, len(shape)))
        for i in range(kn):
            for y in range(len(shape)):
                self.point[i][y] = random.random()
            
        print(self.point)

    

if __name__ == "__main__":
    cc = kmean([[[2]]])
    cc.initalizeRandom(2)