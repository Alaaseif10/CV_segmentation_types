import numpy as np

#K-means
def KmeansSegmentation( image , nclusters , experiments = 3 , threshold = 0 ,
                        maxIterations = 4 ) :
    assert (image.shape[ 2 ] == 3) , "TODO: grayscale not supported yet!"

    imageCopy = np.array(image , copy = True , dtype = np.float)
    featureVector = imageCopy.reshape(-1 , 3  )

    # For each experiment, estimate the cluster labels and the corresponding
    # within-class sum of squares (wss).
    labels = \
        np.zeros((featureVector.shape[ 0 ] , experiments) ,
                 dtype = np.dtype(int))
    wss = \
        np.zeros(experiments)

    # Distance array ( featuresCount X CentroidsCount )
    distance = np.zeros((featureVector.shape[ 0 ] , nclusters))

    for experiment in range(experiments) :
        # Randomly sample from the feature vector initial points as centroids.
        centroids = \
            featureVector[ np.random.choice(featureVector.shape[ 0 ] ,
                                            nclusters , replace = False ) ]

        for iteration in range(maxIterations) :
            # Calculate eucledian dist( points , each centroid).
            for centroidIdx in range(centroids.shape[ 0 ]) :
                distance[ : , centroidIdx ] = \
                    np.linalg.norm(featureVector - centroids[ centroidIdx ] ,
                                   axis = 1 )

            labels[ : , experiment ] = np.argmin(distance , axis = 1 )

            for centroidIdx in range(centroids.shape[ 0 ]) :
                cluster = featureVector[
                    labels[ : , experiment ] == centroidIdx ]
                newCentroid = np.mean(cluster , axis = 0)
                wss[ experiment ] += \
                    np.sum(np.linalg.norm(cluster - newCentroid ,
                                          axis = 1 ), keepdims = True )

                centroids[ centroidIdx ] = newCentroid

    finalLabels = labels[ : , np.argmin(wss) ]
    finalCentroids = __getCentroids__(featureVector , nclusters , finalLabels)
    return finalLabels , finalCentroids , np.min(wss)


def __getCentroids__( population , nclusters , labels ) :
    centroids = np.empty((nclusters , population.shape[ 1 ]))

    for centroidIdx in range(nclusters) :
        cluster = population[ labels == centroidIdx ]
        centroid = np.mean(cluster , axis = 0)
        centroids[ centroidIdx ] = centroid

    return centroids

#Region Growing
np.random.seed(42)
class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y

def getGrayDiff(img, currentPoint, tmpPoint):
    return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))


def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1),
                    Point(1, 0), Point(1, 1), Point(0, 1),
                    Point(-1, 1), Point(-1, 0)]
    else:
        connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]

    return connects


def regionGrow(img, seeds, thresh, p = 1):
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    seedList = []

    for seed in seeds:
        seedList.append(seed)
    label = 1
    connects = selectConnects(p)

    while (len(seedList) > 0):
        currentPoint = seedList.pop(0)

        seedMark[currentPoint.x, currentPoint.y] = label

        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y

            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue

            grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))

            if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
                seedMark[tmpX, tmpY] = label
                seedList.append(Point(tmpX, tmpY))

    return seedMark

def apply_region_growing(source: np.ndarray):
    """
    :param source:
    :return:
    """

    src = np.copy(source)
    img_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    seeds = []
    for i in range(3):
        x = np.random.randint(0, img_gray.shape[0])
        y = np.random.randint(0, img_gray.shape[1])
        seeds.append(Point(x, y))

    # seeds = [Point(10, 10), Point(82, 150), Point(20, 300)]
    output_image = regionGrow(img_gray, seeds, 10)

    return output_image
