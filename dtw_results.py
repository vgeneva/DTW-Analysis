from dtw import *
import numpy as np
import pandas as pd


class DTWResults:
    def __init__(self, train_matrix_path, train_data_path):
        """
        Initializes the DTW Results with the training matrix and data path for 
        the original trainig.tsv file.

        Args:
            train_matrix: the training matrix, numpy array
            train_data_path: the path to the original training data file
        """

        self.train_matrix_path = train_matrix_path
        self.train_data_path = train_data_path

        # Load training matrix
        self.train_matrix = pd.read_csv(train_matrix_path, skiprows = 1, header=None)
        self.train_matrix = self.train_matrix.to_numpy()
        print(f"Loaded training matrix with shape: {self.train_matrix.shape}")
        print(f"Number of training signals: {self.train_matrix.shape[1]}")


        # Load training classifications from train.tsv file
        self.train_data = pd.read_csv(train_data_path, sep='\t', header=None)
        self.classifcations = self.train_data[0].unique()
        self.num_classes = len(self.classifcations)

        # Count occurances of each class in the training data
        self.count = self.train_data[0].value_counts().values.tolist()

        print(f" Loaded training data with {self.num_classes} classifications.")
        
        for num, cls in enumerate(self.classifcations):
            print(f"Class {cls} has {self.count[num]} signals")


    def compute_distance_matrix(self, test_matrix):
        """
        Compute the distance matrix between training and test samples using DTW.
        
        Args:
            test_matrix: the test matrix, numpy array

        Returns:
            Dist_matrix: the distance matrix, numpy array (num_train_samples, num_test_samples)
        """

        # Load test matrix
        #print(f"Loaded test matrix with shape: {test_matrix.shape}")
        print(f"Number of test signals: {test_matrix.shape[1]}")

        m, d = self.train_matrix.shape[1], test_matrix.shape[1]
        print(f"Computing distance matrix for {m} training signals and {d} test signals.")

        # Initialize distance matrix
        Dist_matrix = np.zeros((m, d))

        # Compute DTW distance for each pair of training and test samples
        for i in range(m):
            for j in range(d):
                Dist_matrix[i, j] = dtw(self.train_matrix[:, i], test_matrix[:, j]).distance
        return Dist_matrix
    
    def find_accuracy(self, class_index, test_matrix_path):
        """
        Compute the accuracy of the classification for a specific class using DTW.

        Args:
            class_index (integer): the index of the class to find the
                accuracy for (0-based)
            test_matrix: the test matrix, numpy array
        Returns:
            matrix_min: the minimum distance matrix, numpy array(# of class, # of test signals)
            matrix_ind: the index of the minimum distance, numpy array (# of test signals)
            acc: the accuracy of the classification, float
        """
        # Load test matrix
        test_matrix = pd.read_csv(test_matrix_path, skiprows = 1, header=None)
        test_matrix = test_matrix.to_numpy()
        print(f"Loaded test matrix with shape: {test_matrix.shape}")
        
        m, d = self.train_matrix.shape[1], test_matrix.shape[1] # number of training and test signals
        dist_matrix = self.compute_distance_matrix(test_matrix)
        print(f"Shape of Dist_matrix: {dist_matrix.shape}, it should be ({m}, {d})")

        # Split the distance matrix into submatrices for each class
        start = 0
        submatrices = []
        for size in self.count:  # for each class
            submatrices.append(dist_matrix[start:start+size, :]) # append the submatrix to the list
            start += size
        for sub in range(len(submatrices)):
            print(f"Submatrices sizes: {submatrices[sub].shape}")
        # Find the minimum value in each column of each submatrix
        matrix_min = []
        for sub in range(len(submatrices)):
            submatrices_min = np.reshape(np.min(submatrices[sub], axis=0), (1, d))
            print(f"submatrix shape: {submatrices_min.shape}, should be (1, {d})")
            matrix_min.append(submatrices_min)
        matrix_min = np.concatenate(matrix_min, axis=0)
        print(f"Shape of matrix_min: {matrix_min.shape}")
        # Find the index of the minimum value in each column
        matrix_ind = np.argmin(matrix_min, axis=0)
        # Calculate accuracy
        acc = np.sum(matrix_ind == class_index) / d
        print(f"Accuracy of classification for class {class_index} is {acc:.4f}")

        return matrix_min, matrix_ind, acc
    

"""
# Example usage
beef_train_data_path = "/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_2018/Beef/Beef_TRAIN.tsv"
beef_train_matrix_path = "/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_2018/Beef/Beef_train_matrix.csv"
beef1_test_path = "/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_2018/Beef/Beef1_test_matrix.csv"
classifier = DTWResults(beef_train_matrix_path, beef_train_data_path)

beef2_test_path = "/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_2018/Beef/Beef2_test_matrix.csv"
beef3_test_path = "/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_2018/Beef/Beef3_test_matrix.csv"
beef4_test_path = "/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_2018/Beef/Beef4_test_matrix.csv"
beef5_test_path = "/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_2018/Beef/Beef5_test_matrix.csv"



_, _, acc_beef1 = classifier.find_accuracy(0, beef1_test_path)
_, _, acc_beef2 = classifier.find_accuracy(1, beef2_test_path)
_, _, acc_beef3 = classifier.find_accuracy(2, beef3_test_path)
_, _, acc_beef4 = classifier.find_accuracy(3, beef4_test_path)
_, _, acc_beef5 = classifier.find_accuracy(4, beef5_test_path)

print(acc_beef1)
print(acc_beef2)
print(acc_beef3)
print(acc_beef4)
print(acc_beef5)
"""

