from dtw_results import DTWResults

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

