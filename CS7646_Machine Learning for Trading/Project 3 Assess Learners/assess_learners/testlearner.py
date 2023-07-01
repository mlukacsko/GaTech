""""""  		  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		   	 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		   	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		   	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		   	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 		  		  		    	 		 		   		 		  
or edited.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		   	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		   	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import math  		  	   		   	 		  		  		    	 		 		   		 		  
import sys
import numpy as np
import time
# import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import util
  		  	   		   	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		   	 		  		  		    	 		 		   		 		  
    # if len(sys.argv) != 2:
        # print("Usage: python testlearner.py <filename>")
        # sys.exit(1)
    data = np.genfromtxt(util.get_learner_data_file('Istanbul.csv'), delimiter=',')
    data = data[1:, 1:]
  		  	   		   	 		  		  		    	 		 		   		 		  
    # compute how much of the data is training and testing  		  	   		   	 		  		  		    	 		 		   		 		  
    train_rows = int(0.6 * data.shape[0])  		  	   		   	 		  		  		    	 		 		   		 		  
    test_rows = data.shape[0] - train_rows  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    # separate out training and testing data  		  	   		   	 		  		  		    	 		 		   		 		  
    train_x = data[:train_rows, 0:-1]  		  	   		   	 		  		  		    	 		 		   		 		  
    train_y = data[:train_rows, -1]  		  	   		   	 		  		  		    	 		 		   		 		  
    test_x = data[train_rows:, 0:-1]  		  	   		   	 		  		  		    	 		 		   		 		  
    test_y = data[train_rows:, -1]  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    # print(f"{test_x.shape}")
    # print(f"{test_y.shape}")
  		  	   		   	 		  		  		    	 		 		   		 		  
    # create a learner and train it  		  	   		   	 		  		  		    	 		 		   		 		  
    #learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner
    #learner.add_evidence(train_x, train_y)  # train it

    ''' Experiment 1 '''

    inSamples = np.zeros((40, 1))
    outSamples = np.zeros((40, 1))
    for x in range(1, 41):
        learner = dt.DTLearner(leaf_size=x, verbose=False)  # create a DTLearner
        learner.addEvidence(train_x, train_y)  # train it
        # print(learner.author())

        # evaluate in sample
        pred_y = learner.query(train_x)  # get the predictions
        rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        inSamples[x - 1, 0] = rmse
        c = np.corrcoef(pred_y, y=train_y)
        # print()
        # print("In sample results")
        # print(f"RMSE: {rmse}")
        c = np.corrcoef(pred_y, y=train_y)
        # print(f"corr: {c[0,1]}")

        # evaluate out of sample
        pred_y = learner.query(test_x)  # get the predictions
        rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        outSamples[x - 1, 0] = rmse
        c = np.corrcoef(pred_y, y=test_y)
        # print()
        # print("Out of sample results")
        # print(f"RMSE: {rmse}")
        c = np.corrcoef(pred_y, y=test_y)
        # print(f"corr: {c[0,1]}")
    # plot data
    xaxis = np.arange(1, 41)
    plt.plot(xaxis, inSamples, label="In Sample RMSE")
    plt.plot(xaxis, outSamples, label="Out of Samples RMSE")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("Experiment 1: DTLearner RMSE")
    plt.savefig("DTLearner.png")
    plt.clf()

    ''' Experiment 2 '''

    inSamples = np.zeros((40, 1))
    outSamples = np.zeros((40, 1))
    for x in range(1, 41):
        learner = rt.RTLearner(leaf_size=x, verbose=False)  # create a RTLearner
        learner.addEvidence(train_x, train_y)  # train it
        # print(learner.author())

        # evaluate in sample
        pred_y = learner.query(train_x)  # get the predictions
        rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        inSamples[x - 1, 0] = rmse
        c = np.corrcoef(pred_y, y=train_y)
        # print()
        # print("In sample results")
        # print(f"RMSE: {rmse}")
        c = np.corrcoef(pred_y, y=train_y)
        # print(f"corr: {c[0, 1]}")

        # evaluate out of sample
        pred_y = learner.query(test_x)  # get the predictions
        rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        outSamples[x - 1, 0] = rmse
        c = np.corrcoef(pred_y, y=test_y)
        # print()
        # print("Out of sample results")
        # print(f"RMSE: {rmse}")
        c = np.corrcoef(pred_y, y=test_y)
        # print(f"corr: {c[0, 1]}")
    # plot data
    xaxis = np.arange(1, 41)
    plt.plot(xaxis, inSamples, label="In Sample RMSE")
    plt.plot(xaxis, outSamples, label="Out of Samples RMSE")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("Experiment 2: RTLearner RMSE")
    plt.savefig("RTLearner.png")
    plt.clf()

    ''' Experiment 3 '''

    rmseBagArr1 = []  # in sample
    rmseBagArr2 = []  # out of sample
    rmseRTArr1 = []
    rmseRTArr2 = []
    corrBag1 = []  # in sample
    corrBag2 = []  # out of sample
    corrRT1 = []
    corrRT2 = []
    axi = []
    for i in range(100):
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": i + 1}, bags=10, boost=False,
                                   verbose=False)
        learnerRT = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": i + 1}, bags=10, boost=False,
                                  verbose=False)  # constructor
        learner.addEvidence(train_x, train_y)  # training step
        learnerRT.addEvidence(train_x, train_y)
        predYBag = learner.query(train_x)  # get the predictions
        predYRT = learnerRT.query(train_x)
        rmseBag1 = math.sqrt(((train_y - predYBag) ** 2).sum() / train_y.shape[0])
        rmseRT1 = math.sqrt(((train_y - predYRT) ** 2).sum() / train_y.shape[0])
        cBag = np.corrcoef(predYBag, y=train_y)
        cRT = np.corrcoef(predYRT, y=train_y)
        corrBag1.append(cBag[0, 1])
        corrRT1.append(cRT[0, 1])
        rmseBagArr1.append(rmseBag1)
        rmseRTArr1.append(rmseRT1)
        axi.append(i + 1)
        # evaluate out of sample
        predYBag = learner.query(test_x)  # get the predictions
        predYDT = learnerRT.query(test_x)
        rmseBag2 = math.sqrt(((test_y - predYBag) ** 2).sum() / test_y.shape[0])
        rmseRT2 = math.sqrt(((test_y - predYDT) ** 2).sum() / test_y.shape[0])
        rmseBagArr2.append(rmseBag2)
        rmseRTArr2.append(rmseRT2)
        cBag = np.corrcoef(predYBag, y=test_y)
        cRT = np.corrcoef(predYDT, y=test_y)
        corrBag2.append(cBag[0, 1])
        corrRT2.append(cRT[0, 1])
        print("end", i + 1)

    line1, = plt.plot(axi, rmseBagArr1, label="In sample(DTBag)")
    line2, = plt.plot(axi, rmseBagArr2, label="Out of Sample(DTBag)")
    line3, = plt.plot(axi, rmseRTArr1, label="In sample(RTBag)")
    line4, = plt.plot(axi, rmseRTArr2, label="Out of Sample(RTBag)")
    plt.legend()
    plt.xlabel('Leaf size')
    plt.ylabel('RMSE')
    plt.title("DTBagLearner VS RTBagLearner RMSE Analysis")
    plt.grid(True)
    plt.savefig("RTBagAndDTBag.png")
    plt.clf()

    #
    # Experement 4
    #

    rmseBagArr1 = []  # in sample
    rmseBagArr2 = []  # out of sample
    rmseRTArr1 = []
    rmseRTArr2 = []
    corrBag1 = []  # in sample
    corrBag2 = []  # out of sample
    corrRT1 = []
    corrRT2 = []
    axi = []
    for i in range(40):
        learner = dt.DTLearner(leaf_size=i + 1, verbose=False)
        learnerRT = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": i + 1}, bags=60, boost=False,
                                  verbose=False)  # constructor
        learner.addEvidence(train_x, train_y)  # training step
        learnerRT.addEvidence(train_x, train_y)
        predYBag = learner.query(train_x)  # get the predictions
        predYRT = learnerRT.query(train_x)
        rmseBag1 = math.sqrt(((train_y - predYBag) ** 2).sum() / train_y.shape[0])
        rmseRT1 = math.sqrt(((train_y - predYRT) ** 2).sum() / train_y.shape[0])
        cBag = np.corrcoef(predYBag, y=train_y)
        cRT = np.corrcoef(predYRT, y=train_y)
        corrBag1.append(cBag[0, 1])
        corrRT1.append(cRT[0, 1])
        rmseBagArr1.append(rmseBag1)
        rmseRTArr1.append(rmseRT1)
        axi.append(i + 1)
        # evaluate out of sample
        predYBag = learner.query(test_x)  # get the predictions
        predYDT = learnerRT.query(test_x)
        rmseBag2 = math.sqrt(((test_y - predYBag) ** 2).sum() / test_y.shape[0])
        rmseRT2 = math.sqrt(((test_y - predYDT) ** 2).sum() / test_y.shape[0])
        rmseBagArr2.append(rmseBag2)
        rmseRTArr2.append(rmseRT2)
        cBag = np.corrcoef(predYBag, y=test_y)
        cRT = np.corrcoef(predYDT, y=test_y)
        corrBag2.append(cBag[0, 1])
        corrRT2.append(cRT[0, 1])
        print("end", i + 1)

    line1, = plt.plot(axi, rmseBagArr1, label="In sample(DT)")
    line2, = plt.plot(axi, rmseBagArr2, label="Out of Sample(DT)")
    line3, = plt.plot(axi, rmseRTArr1, label="In sample(RTBag)")
    line4, = plt.plot(axi, rmseRTArr2, label="Out of Sample(RTBag)")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=1)})
    plt.xlabel('Leaf size')
    plt.ylabel('Error')
    plt.title("DTLearner VS RTBagLearner RMSE Analysis")
    plt.grid(True)
    plt.savefig("RTBagandDTLeanrer.png")
    plt.clf()

    '''Experiment 5'''

    dtTimes = np.zeros((50, 1))
    for x in range(1, 51):
        start = time.time()
        learner = dt.DTLearner(leaf_size=x, verbose=False)  # constructor
        learner.addEvidence(train_x, train_y)  # training step

        # evaluate in sample
        predY = learner.query(train_x)  # get the predictions

        # evaluate out of sample
        predY = learner.query(test_x)  # get the predictions
        end = time.time()
        dtTimes[x - 1, 0] = end - start

    rtTimes = np.zeros((50, 1))
    for x in range(1, 51):
        start = time.time()
        learner = rt.RTLearner(leaf_size=x, verbose=False)  # constructor
        learner.addEvidence(train_x, train_y)  # training step

        # evaluate in sample
        predY = learner.query(train_x)  # get the predictions

        # evaluate out of sample
        predY = learner.query(test_x)  # get the predictions
        end = time.time()
        rtTimes[x - 1, 0] = end - start

    xaxis = np.arange(1, 51)
    plt.plot(xaxis, dtTimes, label="DTLearner Times")
    plt.plot(xaxis, rtTimes, label="RTLearner Times")
    plt.xlabel("Leaf Sizes")
    plt.ylabel("Times")
    plt.legend()
    plt.title("Experiment 3 - DT Learners vs. RT Learners wrt Time")
    plt.savefig("Exp5.png")
    plt.clf()


