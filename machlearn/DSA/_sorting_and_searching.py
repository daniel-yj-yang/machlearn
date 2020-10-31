# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

def partition(array, begin_idx, end_idx):
    """
    This function looks at the element of array[end_idx] and identifies its correct position index in a sorted array such that everything to left is smaller or equal, and everything to right is larger.

    Reference:
        https://www.youtube.com/watch?v=PgBzjlCcFvc
    """

    small_element_idx = begin_idx-1   
    pivot_element = array[end_idx]    # pivot element

    for loop_idx in range(begin_idx, end_idx):

        # If current element is smaller than or equal to the pivot element
        if array[loop_idx] <= pivot_element:
            small_element_idx += 1  # increment index of smaller element
            array[small_element_idx], array[loop_idx] = array[loop_idx], array[small_element_idx] # this changes the global variable

    array[small_element_idx+1], array[end_idx] = array[end_idx], array[small_element_idx+1]
    return small_element_idx+1


def quicksort(array, begin_idx, end_idx):
    """
    A divide-and-conquer algorithm
    """
    if len(array) == 1:
        return
    
    if begin_idx < end_idx:

        # array[partitioning_idx] is now at the correct position
        partitioning_idx = partition(array, begin_idx, end_idx)

        # Separately sort elements before and after partition index
        quicksort(array, begin_idx, partitioning_idx-1)
        quicksort(array, partitioning_idx+1, end_idx)

def quicksort_demo():
    import random
    test_array = random.sample(range(1, 1000), 10)
    print(f"before sorting: {test_array}")
    quicksort(test_array, 0, len(test_array)-1)
    print(f"after sorting: {test_array}")
