
import math
import matplotlib.pyplot as plt
import numpy as np
import DTLearner as dtl
import BagLearner as bl
import RTLearner as rtl
import util

if __name__ == "__main__":
    # if len(sys.argv) != 2:
        # print("Usage: python testdt_learner.py <filename>")
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


    """Experiment 4"""

    dt_in_sample = np.zeros((75, 1))
    dt_out_sample = np.zeros((75, 1))
    rt_in_sample = np.zeros((75, 1))
    rt_out_sample = np.zeros((75, 1))
    for x in range(1, 76):
        dt_learner = dtl.DTLearner(leaf_size=x, verbose=False)  # create a DTLearner
        rt_learner = rtl.RTLearner(leaf_size=x, verbose=False)
        dt_learner.addEvidence(train_x, train_y)  # train it
        rt_learner.addEvidence(train_x, train_y)

        # evaluate in sample
        dt_prediction = dt_learner.query(train_x)  # get the predictions
        rt_prediction = rt_learner.query(train_x)
        dt_rmse = math.sqrt(((train_y - dt_prediction) ** 2).sum() / train_y.shape[0])
        rt_rmse = math.sqrt(((train_y - rt_prediction) ** 2).sum() / train_y.shape[0])
        dt_in_sample[x - 1, 0] = dt_rmse
        rt_in_sample[x - 1, 0] = rt_rmse
        c = np.corrcoef(dt_prediction, y=train_y)
        # print()
        # print("In sample results")
        # print(f"RMSE: {rmse}")
        c = np.corrcoef(dt_prediction, y=train_y)
        # print(f"corr: {c[0,1]}")

        # evaluate out of sample
        dt_prediction = dt_learner.query(test_x)  # get the predictions
        rt_prediction = rt_learner.query(test_x)
        dt_rmse = math.sqrt(((test_y - dt_prediction) ** 2).sum() / test_y.shape[0])
        rt_rmse = math.sqrt(((test_y - rt_prediction) ** 2).sum() / test_y.shape[0])
        dt_out_sample[x - 1, 0] = dt_rmse
        rt_out_sample[x - 1, 0] = rt_rmse
        c = np.corrcoef(dt_prediction, y=test_y)
        # print()
        # print("Out of sample results")
        # print(f"RMSE: {rmse}")
        c = np.corrcoef(dt_prediction, y=test_y)
        # print(f"corr: {c[0,1]}")

    # plot data
    x_axis = np.arange(1, 76)
    line1, = plt.plot(x_axis, dt_in_sample, label="In sample(DT)")
    line2, = plt.plot(x_axis, dt_out_sample, label="Out of Sample(DT)")
    line3, = plt.plot(x_axis, rt_in_sample, label="In sample(RT)")
    line4, = plt.plot(x_axis, rt_out_sample, label="Out of Sample(RT)")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("Exp 4: Single Learner RMSE Analysis")
    plt.grid(True)
    # plt.show()
    plt.savefig("Experiment 4.png")
    plt.clf()