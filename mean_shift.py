import numpy as np

class meanShiftSeg:
    def __init__(self, image, windowSize):
        self.image = image.copy()
        self.segmentedImage = image.copy()
        self.windowSize = windowSize
        self.colorSpace = np.zeros((256, 256))
        self.N_clusters = np.int(256 / self.windowSize) ** 2
        self.numOfWindPerDim = np.int(np.sqrt(self.N_clusters))
        self.clustersUV = np.zeros(shape=(self.N_clusters, 2))

    def findCenterMass(self, row, col):
        window = self.colorSpace[row:row + self.windowSize, col:col + self.windowSize]
        momntIdx = range(self.windowSize)
        totalMass = np.max(np.cumsum(window))
        if (totalMass == 0):
            new_Row, new_Col = self.windowSize / 2, self.windowSize / 2
            return row + new_Row, col + new_Col
        if (totalMass > 0):
            momentCol = np.max(np.cumsum(window.cumsum(axis=0)[self.windowSize - 1] * momntIdx))
            cntrCol = np.round(1.0 * momentCol / totalMass)
            momentRow = np.max(np.cumsum(window.cumsum(axis=1)[:, self.windowSize - 1] * momntIdx))
            cntrRow = np.round(1.0 * momentRow / totalMass)
            return row + cntrRow, col + cntrCol

    def apply_mean_shift(self):
        clustersTemp = []
        # creation of window with boundary((0,0),(0,180),(180,180),(180,0))
        for itrRow in range(self.numOfWindPerDim):
            for itrCol in range(self.numOfWindPerDim):
                cntrRow, cntrCol = self.findCenterMass(int(itrRow * self.windowSize), int(itrCol * self.windowSize))
                clustersTemp.append((cntrRow, cntrCol))
        self.clustersUV = np.array(clustersTemp)
        # classifyColors
        numOfWindPerDim = np.int(np.sqrt(self.N_clusters))
        for row in range(self.image.shape[0]):
            for col in range(self.image.shape[1]):
                pixelU = self.segmentedImage[row, col, 1]
                pixelV = self.segmentedImage[row, col, 2]
                windowIdx = np.int(np.int(pixelV / self.windowSize) + np.int(numOfWindPerDim * (pixelU / self.windowSize)))
                self.segmentedImage[row, col, 1] = self.clustersUV[windowIdx, 0]
                self.segmentedImage[row, col, 2] = self.clustersUV[windowIdx, 1]
        return self.segmentedImage

