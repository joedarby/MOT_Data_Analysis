import matplotlib.pyplot as plt
import numpy as np

def analyse_age(df):
    limit = 12000
    bins = list(range(0, limit, int(limit/40)))
    passes = df[df.TestResult == 0].as_matrix(['VehicleAge']).ravel()
    fails = df[df.TestResult == 1].as_matrix(['VehicleAge']).ravel()

    print("By Vehicle Age:")
    print("Passes: ", summary_stats(passes))
    print("Fails: ", summary_stats(fails))

    plt.hist(passes, bins, color='blue', label='Pass')
    plt.hist(fails, bins, color='red', label='Fail')
    plt.title("Histogram of passes and\n fails by vehicle age")
    plt.legend()
    plt.xlabel("Vehicle Age (days)")
    plt.show()

def summary_stats(array):
    return "Mean = ", int(np.mean(array)), " Median = ", int(np.median(array)), " StdDev = ", int(np.std(array))