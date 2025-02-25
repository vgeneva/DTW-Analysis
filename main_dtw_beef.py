from dtw_results import DTWResults
import os
import csv


####### Example usage #############
"""
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
######################################

# make a function to collect all the accuracies and put into a table
def collect_accuracies(algorithm = "dtw"):
    beef_train_data_path = "/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_2018/Beef/Beef_TRAIN.tsv"
    beef_train_matrix_path = "/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_2018/Beef/Beef_train_matrix.csv"
    classifier = DTWResults(beef_train_matrix_path, beef_train_data_path)

    accuracies = []
    time_list = []
    labels =[]
    for i in range(5):
        _, _, acc, time, label = classifier.find_accuracy(i, f"/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_2018/Beef/Beef{i+1}_test_matrix.csv")
        acc = float(acc)
        accuracies.append(acc)
        time = float(time)
        time_list.append(time)
        label = int(label)
        labels.append(label)

    return labels, accuracies, time_list

labels_for_beef, accuracies_for_beef, times_for_beef  = collect_accuracies(algorithm = "dtw")
print(accuracies_for_beef)
print(labels_for_beef)
print(times_for_beef)

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

log_results("dtw_beef", labels_for_beef, accuracies_for_beef, times_for_beef)