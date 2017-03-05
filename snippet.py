snippet = open('../../Documents/Data/test_result_2013_snippet.txt', 'w')

with open('../../Documents/Data/test_result_2013.txt', 'r') as testResults:
    i = 0
    for line in testResults:
        if i % 400 == 0:
            snippet.write(line)
        i += 1

snippet.close()
