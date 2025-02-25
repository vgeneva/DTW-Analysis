from dtw import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# bring in csv files for beef all train, beef1 test, EKG all train, EKG_A_test


beef1_test = pd.read_csv("/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_2018/Beef/Beef1_test_matrix.csv",skiprows = 1, header=None)
beef1_test = beef1_test.to_numpy()
print(beef1_test.shape)
print(beef1_test)
beef_train = pd.read_csv("/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_2018/Beef/Beef_train_matrix.csv", skiprows = 1, header=None)
beef_train = beef_train.to_numpy()
print(beef_train.shape)
print(beef_train)
beef_train_data = pd.read_csv("/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_2018/Beef/Beef_TRAIN.tsv", sep = '\t', header=None)   
beef_train_data = beef_train_data.to_numpy()
print(beef_train_data.shape) 
print(beef_train_data)

# I know that beef1_test[:,0] is beef 1
# I know that beef_train[:,0] is beef 1

# Obtain the dtw distance between beef1_test and beef_train
# obtain the path between beef1_test and beef_train

beef1beef_alignment = dtw(beef1_test[:,0], beef_train[:,0], keep_internals = True)
beef1beef_distance = beef1beef_alignment.distance
beef1beef_path1 = beef1beef_alignment.index1
print(beef1beef_path1)
beef1beef_path2 = beef1beef_alignment.index2
print(beef1beef_path2)


beef1beef_alignment.plot(type = 'twoway', offset = 2)
plt.figure(figsize=(10,5))
plt.plot(beef1beef_path2, beef1beef_path1, 'r')
plt.xlabel('Beef1_test')
plt.ylabel('Beef1_train')
plt.show()


beef_traj = pd.read_csv("/Users/vickyhaney/Documents/GAship/DrBruno/EKG/cdtw_UCR_data/Beef/Beef1Beef_x_traj_4.csv", header=None)
print(beef_traj.shape)
beef_traj = beef_traj.to_numpy()
plt.plot(beef_traj[:,0:1])