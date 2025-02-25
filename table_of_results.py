import pandas as pd
import os

def display_results(filename = 'results_log.csv'):
    
    if not os.path.isfile(filename):
        print(f"File {filename} does not exist.")
        return
    
    # Read the CSV file
    df_original = pd.read_csv(filename)
    
    # Display the DataFrame
    print(df_original)

# Call the function to display results
display_results()


df_original = pd.read_csv("results_log.csv")
print(df_original)
df_original




def transform_results(df):
    # Sort by Label and Algorithm to maintain order
    df_sorted = df.sort_values(by=["Label", "Algorithm"])

    # Reorder columns so it appears as Label, Algorithm, Accuracy, Time (mins)
    df_final = df_sorted[["Label", "Algorithm", "Accuracy", "Time (mins)"]]

    return df_final

# Transform the DataFrame into the interleaved format
df_transformed = transform_results(df_original)

# Display the transformed DataFrame
print(df_transformed)

df_transformed
