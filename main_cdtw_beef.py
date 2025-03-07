from CDTW_results import CDTWResults
import os
import csv



# make a function to collect all the accuracies and put into a table
def collect_accuracies(algorithm = "cdtw"):
    beef_train_data_path = "/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_2018/Beef/Beef_TRAIN.tsv"
    beef_train_matrix_path = "/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_2018/Beef/Beef_train_matrix.csv"
    classifier = CDTWResults(beef_train_matrix_path, beef_train_data_path)
    
    #def find_class_index(label_list): # helper function
    #    list_labels=[]
    #    for ind, num in enumerate(label_list):
    #        print(f"For class_index: {ind}, the label is {num} ")
    #        #list_labels.append(num.split("_train_matrix")[0])
    #    return list_labels
    
    #class_label = find_class_index(list(classifier.classifications))  #figure this out!!!!!
    #print(dir(classifier))
    class_label = classifier.classifications
    print(class_label)

    time_list = []
    accuracies = []
    for ind, num in enumerate(class_label):
        
        print(num)
        path_results = f"/Users/vickyhaney/Documents/GAship/DrBruno/EKG/cdtw_UCR_data/Beef/Beef{num}Beef_dist_obj_4.csv"
        path_test = f"/Users/vickyhaney/Documents/GAship/DrBruno/EKG/cdtw_UCR_data/Beef/Beef{num}_test_matrix.csv"
        #classifier = CDTWResults(path_train)

        _, NN_matrix = classifier.shape_change(path_test, path_results)
        acc = classifier.accuracy(ind, NN_matrix)
        acc = float(acc)
        accuracies.append(acc)

        # open a .txt file
        path_time = f"/Users/vickyhaney/Documents/GAship/DrBruno//EKG/cdtw_UCR_data/Beef/Beef{num}Beef_elapsed_time.txt"
        # extract the value in the txt file
        time_list.append(float(open(path_time).readline().strip())) 

        

    return accuracies, time_list, class_label


#accruracies_for_beet, time_beef, labels_beef = collect_accuracies(algorithm="cdtw")

accruracies_for_beef, time_beef, beef_label = collect_accuracies(algorithm="cdtw")

print(accruracies_for_beef)
print(time_beef)
print(beef_label)

 
def log_results(algorithm, label, accuracies, times, filename = "results_log.csv"):
    # Create a CSV file and write the results
    """
    Logs restults dynamically to a csv file.
    Args:
        algorithm (str): Algorithm used.
        labels (list): List of labels.
        accuracies (list): List of accuracies.
        times (list): List of times.
        filename (str): Name of the CSV file to save the results.
    """
    # Check if the file already exists
    file_exists = os.path.isfile(filename)

    # Open the file in append mode
    with open(filename, mode='a', newline="") as file: # mode='a' to append, mode='w' to write
        writer = csv.writer(file)
        # Write the header only if the file is new
        if not file_exists:
            writer.writerow(["Algorithm", "Label", "Accuracy", "Time (mins)"])

        # Append data, does the order matter?
        for label, acc, time in zip(label, accuracies, times):
            writer.writerow([algorithm, label, acc, time])
    
    print(f"Logged {len(accuracies)} results to {filename} for {algorithm}.")
    return None

log_results("cdtw_beef", beef_label, accruracies_for_beef, time_beef)

 