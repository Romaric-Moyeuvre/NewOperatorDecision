## IMPORT ##########

import math
import numpy

####################

def filter(matrix) :
    number_of_features = len(matrix)
    number_of_measurements = len(matrix[0])
    mask = []
    for k in range(number_of_measurements) :
        measure_k_invalidity = False
        for i in range(number_of_features) :
            measure_k_invalidity = measure_k_invalidity or math.isnan(matrix[i][k])
        if measure_k_invalidity :
            mask.append(k)
    matrix = numpy.delete(matrix, mask, 1)
    return matrix

def sort(matrix) :
    matrix = matrix.transpose()
    matrix = matrix.tolist()
    matrix.sort()
    matrix = numpy.array(matrix).transpose()
    return matrix

def information(matrix) :
    number_of_measurements = len(matrix[0])
    information_values = numpy.array([[1 for k in range(number_of_measurements)]])
    matrix = numpy.append(matrix, information_values, 0)
    return matrix

def pad(matrix) :
    number_of_features = len(matrix)
    number_of_measurements = len(matrix[0])
    padding_values = numpy.array([[0 for j in range(3000 - number_of_measurements)] for i in range(number_of_features)])
    matrix = numpy.append(matrix, padding_values, 1)
    return matrix