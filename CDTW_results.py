import pandas as pd
import numpy as np
 


class CDTWResults:
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
        self.classifications = self.train_data[0].unique()
        self.num_classes = len(self.classifications)
        print(f"Classifications: {self.classifications}")

        self.num_class_each = self.train_data[0].value_counts().values
        print(f"What is {self.num_class_each}?")
        print(f"Number of classes: {self.num_classes}")
        # Count occurances of each class in the training data
        self.count = self.train_data[0].value_counts().values.tolist()
        print(f"Occurances of each class in the training data: {self.count}")
        print(f"Loaded training data with {self.num_classes} classifications.")
        
        for num, cls in enumerate(self.classifications):
            print(f"Class {cls} has {self.count[num]} signals")


        # Initialize elapsed time
        self.elapsed_time = 0




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
        self.results_data = pd.read_csv(file_results_path, header=None)
        self.results_data = np.transpose(self.results_data.to_numpy()) # transpose to get the right shape
        #self.results_data = self.results_data.to_numpy() 
        print(f"Loaded results matrix with shape: {self.results_data.shape}")
        #print(f"first 15 results data: {self.results_data[0:15]}")


        self.file_path_test = file_path_test
        # Load test data
        self.test_data = pd.read_csv(file_path_test)
        self.test_data = self.test_data.to_numpy()
        print(f"Loaded test matrix with shape: {self.test_data.shape}")
        print("Number of test signals: ", self.test_data.shape[1])

        NN_min = []
        chunk = self.train_matrix.shape[1]    # number of training signals
        #chunk = self.train_data.shape[1]    # number of training signals
        print("chunk: ", chunk)
        test_length = self.test_data.shape[1] # number of test signals
        print("test_length: ", test_length)

        for i in range(test_length):
            array = self.results_data[:, i * chunk:(i + 1) * chunk] # choose a chunk of the results data
            print(f"array shape: {array.shape}")
            #for ind, num in enumerate(self.count):
            for ind, num in enumerate(self.num_class_each): #take min every num
                print(f"ind: {ind}, num: {num}")
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
beef_train_data_path = "/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_2018/Beef/Beef_TRAIN.tsv"
beef_train_matrix_path = "/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_2018/Beef/Beef_train_matrix.csv"
 
path_results = f"/Users/vickyhaney/Documents/GAship/DrBruno/EKG/cdtw_UCR_data/Beef/Beef1Beef_dist_obj_4.csv"
beef1 = "/Users/vickyhaney/Documents/GAship/DrBruno/EKG/cdtw_UCR_data/Beef/Beef1Beef_dist_obj_4.csv"
pd.read_csv(beef1, header=None)

path_test = f"/Users/vickyhaney/Documents/GAship/DrBruno/EKG/cdtw_UCR_data/Beef/Beef1_test_matrix.csv"


classifier = CDTWResults(beef_train_matrix_path,beef_train_data_path)
NN_min_Beef1, NN_matrix_Beef1 = classifier.shape_change(path_test, path_results)
print(classifier.classifications)
A_acc = classifier.accuracy(0, NN_matrix_Beef1)
print(f"Accuracy for Beef1: {A_acc}")
"""