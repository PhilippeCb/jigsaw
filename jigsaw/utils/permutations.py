import csv 
import itertools
import numpy as np

def read_permutations():
    return(0)

def write_permutations(max_number_of_permutation:int):
    permutations = create_permutations(max_number_of_permutation)
    np.savetxt("{}_permutations.csv".format(max_number_of_permutation), 
               permutations, 
               delimiter=",",
               fmt='% 4d')

    return(0)

def hamming_distance_matrix(array_a, matrix_b):
    assert len(array_a) == matrix_b.shape[1] 

    distance = np.sum((array_a != matrix_b), axis=1)
    
    return(distance)

def index_from_distance_matrix(distance_matrix):
    one_array = np.ones(distance_matrix.shape[0])
    index = np.argmax(np.dot(one_array.T, distance_matrix))
    return(index)

def create_permutations(max_number_of_permutation:int):
    """
    Creates max_number_of_permutation permutations from index 1 to 9
    slow to run this is why we save results to reuse later 
    """
    number_of_permutation = 362880
    permutations_set = list(itertools.permutations([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    
    #initialisation, pick a random permutation and remove it from the set
    permutations = np.zeros((max_number_of_permutation, 9), dtype=np.uint8)
    distance_matrix = np.zeros((max_number_of_permutation, number_of_permutation), dtype=np.uint8)
    index_permutation = np.random.randint(number_of_permutation)    

    # choose the most different permutation from the initial ones until
    # max number of permutation is reached
    print("computing permutations please wait ... ")
    for i in range(max_number_of_permutation):

        permutations[i] = permutations_set[index_permutation]
        #del permutations_set[index_permutation]
        distance_matrix[i] = hamming_distance_matrix(permutations[i], np.array(permutations_set))
        index_permutation = index_from_distance_matrix(distance_matrix[:i+1, :])

    return(permutations)


if __name__ == "__main__":
    write_permutations(100)