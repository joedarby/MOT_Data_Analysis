# This file takes a data file and produces another file containing a subset of the data

filePath = '../../Documents/Data/test_result_2013'
targetPath = filePath + 'large_subset'

target = open(targetPath + '.txt', 'w')

# Every nth line is written to file
n = 50

with open(filePath + '.txt', 'r') as testResults:
    i = 0
    for line in testResults:
        if i % n == 0:
            target.write(line)
        i += 1

target.close()
