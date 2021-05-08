#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculates Euclidean distances for points up to the specified number of nearby
neighbours.

Uses local sub-regions, memory-mapping, and chunking of data to avoid memory
blowouts.

Can be parallelized to speed up processing.

"""
import numpy as np
from scipy.spatial.distance import cdist


def RegionCropper(InputTable, RegionBounds, DataCols=[3, 4, 5]):
    """
    Crops x, xy, or xyz data in 1, 2, or 3 dimensions

    @params:
        InputTable    - Required  : coordinate data (numpy array)
        RegionBounds  - Required  : limits (List)
        DataCols      - Required  : location of x, y, z columns in the input
                                    array (List)
    """
    RegBndsSize = len(RegionBounds)

    if RegBndsSize == 6 and len(DataCols) == 3:
        # crop by 3 dim
        OutputTable = InputTable[
                                 (RegionBounds[1] > InputTable[:, DataCols[0]]) &
                                 (InputTable[:, DataCols[0]] > RegionBounds[0]) &
                                 (RegionBounds[3] > InputTable[:, DataCols[1]]) &
                                 (InputTable[:, DataCols[1]] > RegionBounds[2]) &
                                 (RegionBounds[5] > InputTable[:, DataCols[2]]) &
                                 (InputTable[:, DataCols[2]] > RegionBounds[4])
                                 ]
    else:
        # something amiss! error!
        print('Problem with size of RegionBounds or DataCols!')
        OutputTable = []
    return OutputTable


def dnns_v3(inputdata1, inputdata3, output_filename, i):
    """
    Compute the distances for the i-th inputdata1 point to all nearest-neighbour 
    points in inputdata2. Nearest neighbours are assesed from ps['ClosestFriend'] 
    up to 1000, inclusive.
    
    This functions returns an array N (points) by M (columns of the original input 
    data) by Z (total neighbours).
    
    Reading along Axis 0, e.g. output[n,:,:] gives the data for nth point. 
    
    Each NN is in each row as distance-to-NNth, NNth's row from the original data 
    table (with it's UID in the last column).
    
    """
    
    # Expects data as [Xcoord, Ycoord, UID]
    TotalColumns = np.shape(inputdata1)[1]
    inputdata1 = inputdata1[i, :].reshape(1, TotalColumns)
    ClosestFriend = 1
    inputdata2 = inputdata3[inputdata3[:, 3] == inputdata1[0, 3], :]
    t = 0
    for different in [False, True]:
        if different:
            t = 1
            ClosestFriend = 0
            inputdata2 = inputdata3[inputdata3[:, 3]
                                != inputdata1[0, 3], :]
            

        maxNeighbours = 1000
        TestRegionSize = 10

        UseableRegion = False

        while not UseableRegion:
            TestRegBounds = [np.floor(inputdata1[0, 0] - TestRegionSize),
                            np.ceil(inputdata1[0, 0] + TestRegionSize),
                            np.floor(inputdata1[0, 1] - TestRegionSize),
                            np.ceil(inputdata1[0, 1] + TestRegionSize),
                            np.floor(inputdata1[0, 2] - TestRegionSize),
                            np.ceil(inputdata1[0, 2] + TestRegionSize)]
            TestRegion = RegionCropper(inputdata2,
                                    TestRegBounds,
                                    [0, 1, 2])
            if np.shape(TestRegion)[0] >= maxNeighbours + 1:
                UseableRegion = True
                # Have a useable region but enlarge it by >= cbrt(3) to make sure
                # we are capturing all the true NNs and not just corner-dense
                # events.
                TestRegionSize = 1.5 * TestRegionSize
                TestRegBounds = [inputdata1[0, 0] - TestRegionSize,
                                inputdata1[0, 0] + TestRegionSize,
                                inputdata1[0, 1] - TestRegionSize,
                                inputdata1[0, 1] + TestRegionSize,
                                inputdata1[0, 2] - TestRegionSize,
                                inputdata1[0, 2] + TestRegionSize]
                TestRegion = RegionCropper(inputdata2,
                                        TestRegBounds,
                                        [0, 1, 2])
            else:
                # Enlarging TestRegionSize to get more points
                TestRegionSize = TestRegionSize + 2
        
        # calculate the distances
        dists = cdist(inputdata1[:, [0, 1, 2]],
                    TestRegion[:, [0, 1, 2]]).reshape(TestRegion.shape[0])
        # print(dists.shape)
        
        # sort by the distance column (column 0)
        dists = dists[dists.argsort()]
            
        # export this point's results for insertion into the main output
        output_filename[i, t, :] = dists[ClosestFriend : 1000 + ClosestFriend]
