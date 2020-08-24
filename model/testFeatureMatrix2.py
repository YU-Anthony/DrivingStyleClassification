import pickle
import numpy as np
np.set_printoptions(suppress = True)
import math
from scipy import stats
import time
import progressbar
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--shape', type=int, nargs='+', default=[5, 5])
# args = parser.parse_args()
# shape = args.shape

class point:
    lat = 0
    lng = 0
    time = 0
    def __init__(self, time, lat, lng):
        self.lat = lat
        self.lng = lng
        self.time = time

class basicFeature:
    speedNorm = 0
    diffSpeedNorm = 0
    accelNorm = 0
    diffAccelNorm = 0
    angularSpeed = 0
    def __init__(self, speedNorm, diffSpeedNorm, accelNorm, diffAccelNorm, angularSpeed):
        self.speedNorm = speedNorm
        self.diffSpeedNorm= diffSpeedNorm
        self.accelNorm= accelNorm
        self.diffAccelNorm = diffAccelNorm
        self.angularSpeed = angularSpeed

        
def returnAngularDisplacement(fLat, fLon, sLat, sLon):
    #Inspired by: https://www.quora.com/How-do-I-convert-radians-per-second-to-meters-per-second
    
    fLat = np.radians(float(fLat))
    fLon = np.radians(float(fLon))
    sLat = np.radians(float(sLat))
    sLon = np.radians(float(sLon))
    
    dis = np.sqrt((fLat-sLat)**2 + (fLon-sLon)**2)
    return dis

def generateStatisticalFeatureMatrix(Ls=256, Lf=4):
    #load trajectories
    trajectories = {}
    filename = 'final.csv'
    with open('analyzerData/preprocess/finalData/'+filename, 'r') as f:
        lines = f.readlines()[1:]
        ct = ''
        cd = ''
        tj = []
        bar = progressbar.ProgressBar()        
        for ln in bar(lines):
            pts = ln.replace('\r\n','').split(',')
            if pts[1] != ct:
                if ct == "" and pts[0]=="Driver":
                    continue
                if len(tj) >0:
                    trajectories[cd+"|"+ct] = tj
                    # print(tj)

                tj = []
                tj.append(point(int(pts[2]), float(pts[3]), float(pts[5])))                
                ct = pts[1]
                cd = pts[0]
            else:
                tj.append(point(int(pts[2]), float(pts[3]), float(pts[5]))) 
        trajectories[cd+"|"+ct] = tj
#     cPickle.dump(trajectories, open('trajectories', 'w'))
    print('Raw Trajectory Data is loaded! |Trajectories|:' + str(len(trajectories)))
    #Generate Basic Features for each trajectory
    basicFeatures = {}
    bar = progressbar.ProgressBar()
    for t in bar(trajectories):
        points = trajectories[t]
        traj = []
        lastSpeedNorm = lastAccelNorm = -1
        lastLatSpeed = lastLngSpeed = 0
        for i in range(1, len(points)):
            speedNorm = np.sqrt((points[i].lat-points[i-1].lat)**2 + (points[i].lng-points[i-1].lng)**2) #Time difference is unit
            diffSpeedNorm = 0
            if lastSpeedNorm > -1:
                diffSpeedNorm = np.abs(speedNorm - lastSpeedNorm)
                
            latSpeed = np.abs(points[i].lat-points[i-1].lat)
            lngSpeed = np.abs(points[i].lng-points[i-1].lng)
            accelNorm = np.sqrt((latSpeed - lastLatSpeed)**2 + (lngSpeed - lastLngSpeed)**2) #Time difference is unit
            
            diffAccelNorm = 0
            if lastAccelNorm > -1:
                diffAccelNorm = np.abs(accelNorm - lastAccelNorm)
            
            angularSpeed = returnAngularDisplacement(points[i-1].lat, points[i-1].lng, points[i].lat, points[i].lng)
             
            lastSpeedNorm = speedNorm
            lastAccelNorm = accelNorm
            lastLatSpeed = latSpeed
            lastLngSpeed = lngSpeed
            
            traj.append([speedNorm, diffSpeedNorm, accelNorm, diffAccelNorm, angularSpeed])        
        basicFeatures[t] = traj

    del trajectories
    print('Basic Features are created!')
    #Generate Statistical Feature Matrix
    bar = progressbar.ProgressBar()
    start = time.time()
    statisticalFeatureMatrix = {}
    for t in bar(basicFeatures):
        #print 'processing', t      
        matricesForTrajectory = []
        traj= basicFeatures[t]
        # print("len(traj):",len(traj))
        ranges = returnSegmentIndexes(Ls, len(traj))       
        # print("ranges:",ranges) 
        for p in ranges:
            if p[1] - p[0] < 256:
                continue
            # 创建指定大小的数组
            matrixForSegment = np.empty((129, 35))
            # 对每一维都初始化为0
            matrixForSegment[0, :] = np.zeros((35,))
            # print(matrixForSegment)
            st = p[0]
            # print("pre st:",st)
            for timestep in range(1, 129):
                en = min(st+Lf, p[1])
                # print("en:",en)
                # print("p[1]:",p[1])
                column = []
                for fIdx in range(0, 5):
                    arr = []
                    mean = 0.0
                    for i in range(st, en):            
                        mean += traj[i][fIdx]
                        arr.append(traj[i][fIdx])      
                    arr.sort()
                    mean = mean//len(arr)
                    column.append(mean) #mean
                    column.append(arr[0]) #min
                    column.append(arr[len(arr)-1]) #max                    
                    column.append(stats.scoreatpercentile(arr, 25)) #25% percentile
                    if len(arr)%2 == 0:
                        column.append((arr[len(arr)//2] + arr[(len(arr)//2) -1])//2) #50% percentile
                    else:
                        column.append(arr[len(arr)//2]) #50% percentile
                    column.append(stats.scoreatpercentile(arr, 75)) #75% percentile
                    std = 0
                    for a in arr:
                        std += (a-mean)**2
                    column.append(math.sqrt(std)) #standard deviation
                matrixForSegment[timestep, :] = list(column)
                st += Lf//2
                # print("after st:",st)            
            matricesForTrajectory.append(matrixForSegment)
              
        statisticalFeatureMatrix[t] = normalizeStatFeatureMatrix(np.array(matricesForTrajectory))
        print("statisticalFeatureMatrix[t]",statisticalFeatureMatrix[t])
    
    del basicFeatures
    print("statistical features created")
    keys = [k.split("|") for k, v in statisticalFeatureMatrix.items() for i in range(v.shape[0])]
    print(keys)
    pickle.dump(keys, open("RNN/smallSample_keys.pkl", "wb"))
    del keys
    np.save('RNN/smallSample.npy', np.vstack(statisticalFeatureMatrix.values()), allow_pickle=False)

def returnSegmentIndexes(Ls, leng):
    ranges = []
    start = 0
    while True:        
        end = min(start+Ls, leng-1)
        ranges.append([start, end])
        start += Ls//2
        if end == leng-1:
            break        
    return ranges

def normalizeStatFeatureMatrix(statisticalFeatureMatrix, minimum=0, maximum=40):
    r = float(maximum-minimum)
    mins = statisticalFeatureMatrix.min((0, 1))
    maxs = statisticalFeatureMatrix.max((0, 1))    
    statisticalFeatureMatrix = np.nan_to_num(minimum + ((statisticalFeatureMatrix-mins)/(maxs-mins))*r)
    return statisticalFeatureMatrix
    
if __name__ == '__main__':
    generateStatisticalFeatureMatrix()    