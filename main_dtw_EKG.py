from dtw_results_EKG_only import DTWResults
from c_dtw_results import CDTWResults # just to obtain the labels for class_index
import csv
import os





# make a function to collect all the accuracies and put into a table
def collect_accuracies(algorithm = "dtw"):

    path_train = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/2_current_filtered_data/all_train_data.csv'
    classifier_for_labels = CDTWResults(path_train) # only need this to get the labels so i can get accuracies using DTWResults

    def find_class_index(label_list): # helper function
        list_labels=[]
        for ind, num in enumerate(label_list):
            print(f"For class_index: {ind}, the label is {num} ")
            list_labels.append(num.split("_train_matrix")[0])
        return list_labels
    
    class_label = find_class_index(list(classifier_for_labels.classification_labels))
    print(class_label)


    train_matrix_path = "/Users/vickyhaney/Documents/GAship/DrBruno/EKG/2_current_filtered_data/all_train_data.csv"
    classifier = DTWResults(train_matrix_path)  # Not sure if this will work.  EKG is different than beef 


    accuracies = []
    time_list = []
    labels =[]
    for ind, num in enumerate(class_label):
        #name variables dynamically using globals
        #globals()[f"acc_{num}"] = acc
        #globals()[f"time_{num}"] = time
        _, _, acc, time, label = classifier.find_accuracy(ind, f"/Users/vickyhaney/Documents/GAship/DrBruno/EKG/current_filtered_data/{num}_test_matrix.csv")
        print(acc, time)
        acc = float(acc)
        accuracies.append(acc)
        time = float(time)
        time_list.append(time)
        labels.append(label)
    
    return labels, accuracies, time_list # order matters for results log


labels_for_EKG, accuracies_for_EKG, times_for_EKG = collect_accuracies(algorithm = "dtw")
print(accuracies_for_EKG)
print(labels_for_EKG)
print(times_for_EKG)

new_list_labels = []
for label in labels_for_EKG:
    print(label)
    new_list_labels.append(label.split("_train_matrix")[0]) # remove _test_matrix part of the EKG data

print(new_list_labels)


 
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

log_results("dtw_EKG", new_list_labels, accuracies_for_EKG, times_for_EKG)



