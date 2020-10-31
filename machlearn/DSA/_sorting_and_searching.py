# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

import random
import timeit



def mergesort_inplace(array):
    """
    Reference: https://www.youtube.com/watch?v=JSceec-wEyw
    """
    if len(array) > 1:
        mid_idx = len(array)//2  # Finding the mid of the array
        L_half = array[:mid_idx] # Dividing the array elements into 2 halves
        R_half = array[mid_idx:] 
 
        mergesort_inplace(L_half) # Sorting the left half
        mergesort_inplace(R_half) # Sorting the right half
 
        left_idx = right_idx = k = 0
         
        # Copy data from temp arrays L_half[] and R_half[]
        while left_idx < len(L_half) and right_idx < len(R_half):
            if L_half[left_idx] < R_half[right_idx]:
                array[k] = L_half[left_idx]
                left_idx += 1
            else:
                array[k] = R_half[right_idx]
                right_idx += 1
            k+= 1
         
        # Checking if any element was left
        while left_idx < len(L_half):
            array[k] = L_half[left_idx]
            left_idx += 1
            k += 1
         
        while right_idx < len(R_half):
            array[k] = R_half[right_idx]
            right_idx += 1
            k += 1


def mergesort(array):

    if array == []:
        return []
    elif len(array) == 1:
        return array
    else:
        mid_idx = len(array)//2   # Finding the mid of the array
        L_half = array[:mid_idx]  # Dividing the array elements into 2 halves
        R_half = array[mid_idx:]
        L_half = mergesort(L_half)
        R_half = mergesort(R_half)

        sorted_array = []

        while len(L_half) > 0 and len(R_half) > 0:
            if L_half[0] < R_half[0]:
                sorted_array.append(L_half.pop(0))
            else:
                sorted_array.append(R_half.pop(0))

        for left_element in L_half:
            sorted_array.append(left_element)
        for right_element in R_half:
            sorted_array.append(right_element)

        return sorted_array



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


def sort_profile():

    def test_mergesort_inplace():
        test_array = random.sample(range(1, 1000000), 300)
        mergesort_inplace(test_array)

    def test_mergesort():
        test_array = random.sample(range(1, 1000000), 300)
        mergesort(test_array)
    
    def test_quicksort_inplace():
        test_array = random.sample(range(1, 1000000), 300)
        quicksort_inplace(test_array, 0, len(test_array)-1)
        
    def test_quicksort():
        test_array = random.sample(range(1, 1000000), 300)
        quicksort(test_array)
        
    def test_python_array_sort():
        test_array = random.sample(range(1, 1000000), 300)
        test_array.sort()
        
    def test_python_sorted():
        test_array = random.sample(range(1, 1000000), 300)
        sorted(test_array)    

    print("\nBenchmarking:")
    print(f"mergesort_inplace(): {timeit.timeit(test_mergesort_inplace, number=10000):.2f} sec")
    print(f"mergesort(): {        timeit.timeit(test_mergesort,         number=10000):.2f} sec")  
    print(f"quicksort_inplace(): {timeit.timeit(test_quicksort_inplace, number=10000):.2f} sec")
    print(f"quicksort(): {        timeit.timeit(test_quicksort,         number=10000):.2f} sec")
    print(f"python [].sort(): {   timeit.timeit(test_python_array_sort, number=10000):.2f} sec")
    print(f"python sorted(): {    timeit.timeit(test_python_sorted,     number=10000):.2f} sec")


def sort_demo():
    test_array = random.sample(range(1, 1000000), 10)
    print("\nmergesort():")
    print(f"before sorting: {test_array}")
    print(f"after sorting: {mergesort(test_array)}")

    test_array = random.sample(range(1, 1000000), 10)
    print("\nquicksort_inplace():")
    print(f"before sorting: {test_array}")
    quicksort_inplace(test_array, 0, len(test_array)-1)
    print(f"after sorting: {test_array}")
    
    sort_profile()
