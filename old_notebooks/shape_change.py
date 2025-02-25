import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class CDTWResults:
    def __init__(self, file_results_path, file_path_train):#, file_path_test):
        """ 
        Initializes the CDTW Results with the training matrix with labels, the test_matrix,
        the results matrix.
        
        Args:
            file_results_path (str): Path to the results file.
            file_path_train (str): Path to the training data file.
            file_path_test (str): Path to the test data file.
        """
    
        self.file_results_path = file_results_path
        self.file_path_train = file_path_train
        #self.file_path_test = file_path_test
        
        # Load training data
        self.train_data = pd.read_csv(file_path_train)
        # remove the '.' in label names, i.e. .1 and .2 and so on from column names
        self.train_data.columns = self.train_data.columns.str.split('.').str[0]
        self.classification_labels = self.train_data.columns.unique()
        self.num_classes = len(self.classification_labels)
        print(f"Number of classes: {self.num_classes}")
        print(f"Loaded training matrix withe shape: {self.train_data.shape}")
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

        # Load results data
        self.results_data = pd.read_csv(file_results_path)
        self.results_data = np.transpose(self.results_data.to_numpy()) # transpose to get the right shape
        print(f"Loaded results matrix with shape: {self.results_data.shape}")
        print(f"first 15 results data: {self.results_data[0:15]}")


    def shape_change(self, file_path_test):
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

        print(f" {len(NN_min)} NN_min should be {self.test_data.shape[1] * self.num_classes}")

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
        """
        d = NN_matrix.shape[1]
        # get the index of the min value in each column
        #print(NN_matrix)
        matrix_index = np.argmin(NN_matrix, axis=0)
        #print(f"Index of min values: {matrix_index}")
        # get the number of correct classifications
        accuracy = np.sum(matrix_index == class_index) / d

        return accuracy



    
# Take in .csv file and return a numpy array
file_path_train = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/2_current_filtered_data/all_train_data.csv'
file_path_results = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/gat_alg_distmatrices/adam_alltrain_2/Aall_dist_obj_4.csv'
file_path_test_A = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/current_filtered_data/A_test_matrix.csv'


classifier = CDTWResults(file_path_results, file_path_train)#, file_path_test_A)
NN_min_A, NN_matrix_A = classifier.shape_change(file_path_test_A)
NN_min_A[0:15]
A_acc = classifier.accuracy(7, NN_matrix_A)
print(f"Accuracy for A: {A_acc}")


file_path_results_R = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/gat_alg_distmatrices/adam_alltrain_2/Rall_dist_obj_4.csv'
file_path_test_R = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/current_filtered_data/R_test_matrix.csv'
classifier_R = CDTWResults(file_path_results_R, file_path_train)#, file_path_test_R)
NN_min_R, NN_matrix_R = classifier_R.shape_change(file_path_test_R)
R_acc = classifier_R.accuracy(0, NN_matrix_R)
print(f"Accuracy for R: {R_acc}")

file_path_results_V = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/gat_alg_distmatrices/adam_alltrain_2/Vall_dist_obj_4.csv'
file_path_test_V = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/current_filtered_data/V_test_matrix.csv'
classifier_V = CDTWResults(file_path_results_V, file_path_train)#, file_path_test_V)
NN_min_V, NN_matrix_V = classifier_V.shape_change(file_path_test_V)
V_acc = classifier_V.accuracy(1, NN_matrix_V)
print(f"Accuracy for V: {V_acc}")

file_path_results_L = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/gat_alg_distmatrices/adam_alltrain_2/Lall_dist_obj_4.csv'
file_path_test_L = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/current_filtered_data/L_test_matrix.csv'
classifier_L = CDTWResults(file_path_results_L, file_path_train)#, file_path_test_L)
NN_min_L, NN_matrix_L = classifier_L.shape_change(file_path_test_L)
L_acc = classifier_L.accuracy(2, NN_matrix_L)
print(f"Accuracy for L: {L_acc}")

file_path_results_jj = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/gat_alg_distmatrices/adam_alltrain_2/gall_dist_obj_4.csv'
file_path_test_jj = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/current_filtered_data/jj_test_matrix.csv'
classifier_jj = CDTWResults(file_path_results_jj, file_path_train)#, file_path_test_jj)
NN_min_jj, NN_matrix_jj = classifier_jj.shape_change(file_path_test_jj)
jj_acc = classifier_jj.accuracy(3, NN_matrix_jj)
print(f"Accuracy for jj: {jj_acc}")


file_path_results_E = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/gat_alg_distmatrices/adam_alltrain_2/Eall_dist_obj_4.csv'
file_path_test_E = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/current_filtered_data/E_test_matrix.csv'
classifier_E = CDTWResults(file_path_results_E, file_path_train)#, file_path_test_E)
NN_min_E, NN_matrix_E = classifier_E.shape_change(file_path_test_E)
E_acc = classifier_E.accuracy(4, NN_matrix_E)
print(f"Accuracy for E: {E_acc}")


file_path_results_N = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/gat_alg_distmatrices/adam_alltrain_2/Nall_dist_obj_4.csv'
file_path_test_N = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/current_filtered_data/N_test_matrix.csv'
classifier_N = CDTWResults(file_path_results_N, file_path_train)#, file_path_test_N)
NN_min_N, NN_matrix_N = classifier_N.shape_change(file_path_test_N)
N_acc = classifier_N.accuracy(5, NN_matrix_N)
print(f"Accuracy for N: {N_acc}")

file_path_results_J = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/gat_alg_distmatrices/adam_alltrain_2/Jall_dist_obj_4.csv'
file_path_test_J = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/current_filtered_data/J_test_matrix.csv'
classifier_J = CDTWResults(file_path_results_J, file_path_train)#, file_path_test_J)
NN_min_J, NN_matrix_J = classifier_J.shape_change(file_path_test_J)
J_acc = classifier_J.accuracy(6, NN_matrix_J)
print(f"Accuracy for J: {J_acc}")


print(A_acc, R_acc, V_acc, L_acc, jj_acc, E_acc, N_acc, J_acc)