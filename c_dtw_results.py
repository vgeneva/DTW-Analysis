import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class CDTWResults:
    def __init__(self, file_path_train):#, file_results_path, file_path_test):
        """ 
        Initializes the CDTW Results with the training data .tsv file
        
        Args:
            file_results_path (str): Path to the results file.
            file_path_train (str): Path to the training data file.
            file_path_test (str): Path to the test data file.
        """
    
        #self.file_results_path = file_results_path
        self.file_path_train = file_path_train
        #self.file_path_test = file_path_test
        
        # Load training data
        self.train_data = pd.read_csv(file_path_train)
        # remove the '.' in label names, i.e. .1 and .2 and so on from column names
        self.train_data.columns = self.train_data.columns.str.split('.').str[0]
        self.classification_labels = self.train_data.columns.unique() # this gets used for dtw_results_EKG_only, so becareful not to change these
        self.num_classes = len(self.classification_labels)
        print(f"Number of classes: {self.num_classes}")
        print(f"Loaded training matrix with shape: {self.train_data.shape}")
        print("Number of training signals: ", self.train_data.shape[1])
        self.num_class_each = self.train_data.columns.value_counts().values
        print("Training labels: ", self.classification_labels)
        print("Number of training signals per class: ", self.num_class_each)
        """
        # Load test data
        self.test_data = pd.read_csv(file_path_test)
        self.test_data = self.test_data.to_numpy()
        print(f"Loaded test matrix with shape: {self.test_data.shape}")
        print("Number of test signals: ", self.test_data.shape[1])
        """




    def shape_change(self, file_path_test, file_results_path):
        """
        Changes the shape of the results array to (# classes * # of each class * # of test signals)
        , taking np.min for each class (depending on how many training in each class), then change
        to shape: (# classes, # of test signals).


        Args:
            file_path_test (str): Path to the test data file.
        Returns:
            NN_min (np.array): Nearest Neighbor. The minimum values for each class and test signal.
            NN_matrix (np.array): The matrix of minimum values for each class and test signal. 
                Size: (# classes, # of test signals)
        """
        # load resutls data path
        self.file_results_path = file_results_path
        # Load results data
        self.results_data = pd.read_csv(file_results_path)
        self.results_data = np.transpose(self.results_data.to_numpy()) # transpose to get the right shape
        print(f"Loaded results matrix with shape: {self.results_data.shape}")
        #print(f"first 15 results data: {self.results_data[0:15]}")


        self.file_path_test = file_path_test
        # Load test data
        self.test_data = pd.read_csv(file_path_test)
        self.test_data = self.test_data.to_numpy()
        print(f"Loaded test matrix with shape: {self.test_data.shape}")
        print("Number of test signals: ", self.test_data.shape[1])

        NN_min = []
        chunk = self.train_data.shape[1]    # number of training signals
        #print("chunk: ", chunk)
        test_length = self.test_data.shape[1] # number of test signals
        #print("test_length: ", test_length)

        for i in range(test_length):
            array = self.results_data[:, i * chunk:(i + 1) * chunk] # choose a chunk of the results data
            #print(f"array shape: {array.shape}")
            for ind, num in enumerate(self.num_class_each): #take min every num
                NN_min.append(np.min(array[:, ind * num:(ind + 1) * num]))

        print(f"Length of NN_min: {len(NN_min)}, should be {self.test_data.shape[1] * self.num_classes}")

        # take NN_min and collect the frist 8 (or however many classes) and make into a column vector,
        # then take the next 8 and make into a column vector, and so on, and stack them into 
        # a matrix of shape (# classes, # of test signals)
        NN_min = np.array(NN_min)
        NN_matrix = np.zeros((self.num_classes, test_length))
        for ind in range(test_length):
            NN_matrix[:, ind] = NN_min[ind*self.num_classes:(ind+1)*self.num_classes]
        print(f"NN_matrix shape: {NN_matrix.shape}")

        return NN_min, NN_matrix


    def accuracy(self, class_index, NN_matrix):
        """
        Takes the NN_matrix and returns the accuracy of the classification.
        class_index will be an iput to the function, which will be the index of the class in the
        classification_labels list.
        Args:
            class_index (int): The index of the class in the classification_labels list.
            NN_matrix (np.array): The matrix of minimum values for each class and test signal. 
                Size: (# classes, # of test signals)
        Returns:
            accuracy (float): The accuracy of the classification.
        """
        d = NN_matrix.shape[1]
        # get the index of the min value in each column
        #print(NN_matrix)
        matrix_index = np.argmin(NN_matrix, axis=0)
        #print(f"Index of min values: {matrix_index}")
        # get the number of correct classifications
        accuracy = float(np.sum(matrix_index == class_index) / d)
        accuracy = round(accuracy, 4)


        return accuracy



############## Example Usage #######################
"""
# Take in .csv file and return a numpy array
file_path_train = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/2_current_filtered_data/all_train_data.csv'
file_path_results = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/gat_alg_distmatrices/adam_alltrain_2/Aall_dist_obj_4.csv'
file_path_test_A = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/current_filtered_data/A_test_matrix.csv'


classifier = CDTWResults(file_path_train)#, file_path_test_A)
NN_min_A, NN_matrix_A = classifier.shape_change(file_path_test_A, file_path_results)
NN_min_A[0:15]
A_acc = classifier.accuracy(7, NN_matrix_A)
print(f"Accuracy for A: {A_acc}")
"""

beef_train_data_path = "/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_2018/Beef/Beef_TRAIN.tsv"
pd.read_csv(beef_train_data_path, sep='\t', header = None)
path_results = f"/Users/vickyhaney/Documents/GAship/DrBruno/EKG/cdtw_UCR_data/Beef/Beef1Beef_dist_obj_4.csv"
path_test = f"/Users/vickyhaney/Documents/GAship/DrBruno/EKG/cdtw_UCR_data/Beef/Beef1_test_matrix.csv"

classifier = CDTWResults(beef_train_data_path)
NN_min_A, NN_matrix_A = classifier.shape_change(path_test, path_results)
A_acc = classifier.accuracy(0, NN_matrix_A)
print(f"Accuracy for Beef1: {A_acc}")