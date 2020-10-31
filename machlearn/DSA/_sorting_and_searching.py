# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

import random
import timeit

def identify_correct_partition_idx(array, begin_idx, end_idx):
    """
    This function looks at the element of array[end_idx] and then it identifies its correct position index in a sorted array such that everything to left is smaller or equal, and everything to right is larger.

    Reference:
        https://www.youtube.com/watch?v=PgBzjlCcFvc
    """

    smaller_element_idx = begin_idx-1   
    pivot_element = array[end_idx]    # pivot element

    for loop_idx in range(begin_idx, end_idx):

        if array[loop_idx] <= pivot_element:
            # the idea is that if we find an element smaller or equal to the pivot element, then we move it to the left of the pivot element
            smaller_element_idx += 1  # increment index of smaller element
            array[smaller_element_idx], array[loop_idx] = array[loop_idx], array[smaller_element_idx] # this changes the global variable of the array

    # place the pivot element in the position of (smaller_element_idx + 1), which is the correct position since everything to the left is smaller or equal to the pivot element
    array[smaller_element_idx+1], array[end_idx] = array[end_idx], array[smaller_element_idx+1]
    return smaller_element_idx+1


def quicksort_inplace(array, begin_idx, end_idx):
    """
    A divide-and-conquer algorithm
    """
    if len(array) == 1:
        return
    
    if begin_idx < end_idx:
        correct_partitioning_idx = identify_correct_partition_idx(array, begin_idx, end_idx)

        # Using the "correct_partitioning_idx" to divide and conquer in a recursive manner
        quicksort_inplace(array, begin_idx, correct_partitioning_idx-1)
        quicksort_inplace(array, correct_partitioning_idx+1, end_idx)


def quicksort(array):
    """
    Not Inplace, but Standard version
    """
    if array == []:
        return []
    else:
        pivot = array[-1]
        smaller = quicksort([x for x in array[0:-1] if x <= pivot])
        larger = quicksort([x for x in array[0:-1] if x > pivot])
        return smaller + [pivot] + larger


def profile_quicksort():
    
    def test_quicksort_inplace():
        test_array = random.sample(range(1, 1000), 300)
        quicksort_inplace(test_array, 0, len(test_array)-1)
        
    def test_quicksort():
        test_array = random.sample(range(1, 1000), 300)
        quicksort(test_array)
        
    def test_python_array_sort():
        test_array = random.sample(range(1, 1000), 300)
        test_array.sort()
        
    def test_python_sorted():
        test_array = random.sample(range(1, 1000), 300)
        sorted(test_array)    
        
    print(f"quicksort_inplace(): {timeit.timeit(test_quicksort_inplace, number=30000):.2f} sec")
    print(f"quicksort(): {        timeit.timeit(test_quicksort,         number=30000):.2f} sec")
    print(f"python [].sort(): {   timeit.timeit(test_python_array_sort, number=30000):.2f} sec")
    print(f"python sorted(): {    timeit.timeit(test_python_sorted,     number=30000):.2f} sec")


def quicksort_demo():
    test_array = random.sample(range(1, 1000), 10)
    print(f"before sorting: {test_array}")
    quicksort_inplace(test_array, 0, len(test_array)-1)
    print(f"after sorting: {test_array}")
    
    profile_quicksort()
