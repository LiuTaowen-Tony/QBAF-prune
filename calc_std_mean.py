import os
import csv
import statistics

def calculate_avg_std(filename):
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file)
        # headers = ["connections", "test_acc", "precision", "recall", "f1"]
        headers = next(reader)  # Read the headers
        headers = headers[1:]  # Remove the first column header

        data = [[] for _ in range(len(headers))]  # Create empty lists for each column

        k = 0
        for row in reader:
            for i, value in enumerate(row[1:]):
                try:
                    data[i].append(float(value))  # Assuming all values are numeric, convert to float
                except:
                    data[i].append(0.0)
            k += 1

        print(f"number of lines: {k}")
        if k < 2:
            print("not enough data")
            return

        print("Column\t\tAverage\t\tStandard Deviation")
        print("--------------------------------------------")
        for i, column in enumerate(data):
            avg = statistics.mean(column)
            std = statistics.stdev(column)
            print(f"{headers[i]:<20}\t\t{avg:.4f}\t\t{std:.4f}")

# traverse files in current directory
for file in os.listdir("."):
    if file.endswith(".csv"):
        print(f"file: {file}")
        calculate_avg_std(file)

# calculate_avg_std(sys.argv[1])