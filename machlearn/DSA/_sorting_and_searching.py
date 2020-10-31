# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

import random
import timeit

# Time complexity of comparison sort:
# BubbleSort: n^2
# TreeSort: nlogn
# HeapSort: nlogn
# MergeSort: nlogn
# QuickSort: nlogn


def brute_force_sort(array):
    unsorted_array = array.copy()
    sorted_array = []
    while unsorted_array:
        sorted_array.append(unsorted_array.pop(unsorted_array.index(min(unsorted_array))))
    return sorted_array


def bubble_sort_inplace(array):
    n = len(array)
    for i in range(n):
        swap_occurred = False
        for j in range(0, n-i-1): # why n-i-1? because the last i elements are already sorted in place
            if array[j] > array[j+1]:
                array[j], array[j+1] = array[j+1], array[j]
                swap_occurred = True
        if swap_occurred == False:
            break


def tree_sort(array):
    if len(array) <= 1:
        return array
    from ._trees_and_graphs import binary_search_tree
    BST = binary_search_tree()
    BST.construct_BST(array)
    return_array = []
    BST.order(BST.root_node, return_array, type="Inorder")
    return return_array


def heap_sort_inplace(array):
    heap_sort(array)
    
class heap_sort():
    def __init__(self, array):
        self.array = array
        self.sort_inplace()
        
    def heapify(self, arr, n, i):
        # Find largest among root and children
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2

        if l < n and arr[i] < arr[l]:
            largest = l

        if r < n and arr[largest] < arr[r]:
            largest = r

        # If root is not largest, swap with largest and continue heapifying
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            self.heapify(arr, n, largest)

    def sort_inplace(self):
        n = len(self.array)

        # Build max heap
        for i in range(n//2, -1, -1):
            self.heapify(self.array, n, i)

        for i in range(n-1, 0, -1):
            self.array[i], self.array[0] = self.array[0], self.array[i] # swap
            self.heapify(self.array, i, 0)  # Heapify root element


def merge_sort_inplace(array):
    """
    Reference: https://www.youtube.com/watch?v=JSceec-wEyw
    """
    if len(array) > 1:
        mid_idx = len(array)//2  # Finding the mid of the array
        L_half = array[:mid_idx] # Dividing the array elements into 2 halves
        R_half = array[mid_idx:] 
 
        merge_sort_inplace(L_half) # Sorting the left half
        merge_sort_inplace(R_half) # Sorting the right half
 
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


def merge_sort(array):

    if array == []:
        return []
    elif len(array) == 1:
        return array
    else:
        mid_idx = len(array)//2   # Finding the mid of the array
        L_half = array[:mid_idx]  # Dividing the array elements into 2 halves
        R_half = array[mid_idx:]
        L_half = merge_sort(L_half)
        R_half = merge_sort(R_half)

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


def quick_sort_inplace(array, begin_idx, end_idx):
    """
    A divide-and-conquer algorithm
    """
    if len(array) == 1:
        return
    
    if begin_idx < end_idx:
        correct_partitioning_idx = identify_correct_partition_idx(array, begin_idx, end_idx)

        # Using the "correct_partitioning_idx" to divide and conquer in a recursive manner
        quick_sort_inplace(array, begin_idx, correct_partitioning_idx-1)
        quick_sort_inplace(array, correct_partitioning_idx+1, end_idx)


def quick_sort(array):
    """
    Not Inplace, but Standard version
    """
    if array == []:
        return []
    else:
        pivot = array[-1]
        smaller = quick_sort([x for x in array[0:-1] if x <= pivot])
        larger = quick_sort([x for x in array[0:-1] if x > pivot])
        return smaller + [pivot] + larger


def sort_profiling():

    n_samples = 500

    def test_brute_force_sort():
        test_array = random.sample(range(1, 1000000), n_samples)
        brute_force_sort(test_array)

    def test_bubble_sort_inplace():
        test_array = random.sample(range(1, 1000000), n_samples)
        bubble_sort_inplace(test_array)

    def test_tree_sort():
        test_array = random.sample(range(1, 1000000), n_samples)
        tree_sort(test_array)

    def test_heap_sort_inplace():
        test_array = random.sample(range(1, 1000000), n_samples)
        heap_sort_inplace(test_array)

    def test_merge_sort_inplace():
        test_array = random.sample(range(1, 1000000), n_samples)
        merge_sort_inplace(test_array)

    def test_merge_sort():
        test_array = random.sample(range(1, 1000000), n_samples)
        merge_sort(test_array)
    
    def test_quick_sort_inplace():
        test_array = random.sample(range(1, 1000000), n_samples)
        quick_sort_inplace(test_array, 0, len(test_array)-1)
        
    def test_quick_sort():
        test_array = random.sample(range(1, 1000000), n_samples)
        quick_sort(test_array)
        
    def test_python_array_sort():
        test_array = random.sample(range(1, 1000000), n_samples)
        test_array.sort()
        
    def test_python_sorted():
        test_array = random.sample(range(1, 1000000), n_samples)
        sorted(test_array)    

    print("\nBenchmarking (from slow to fast):")
    print(f"bubble_sort_inplace(): {timeit.timeit(test_bubble_sort_inplace, number=10000):.2f} sec")
    print(f"brute_force_sort(): {   timeit.timeit(test_brute_force_sort,    number=10000):.2f} sec")
    print(f"heap_sort_inplace(): {  timeit.timeit(test_heap_sort_inplace,   number=10000):.2f} sec")
    print(f"tree_sort(): {          timeit.timeit(test_tree_sort,           number=10000):.2f} sec")
    print(f"merge_sort(): {         timeit.timeit(test_merge_sort,          number=10000):.2f} sec")
    print(f"merge_sort_inplace(): { timeit.timeit(test_merge_sort_inplace,  number=10000):.2f} sec")
    print(f"quick_sort(): {         timeit.timeit(test_quick_sort,          number=10000):.2f} sec")  
    print(f"quick_sort_inplace(): { timeit.timeit(test_quick_sort_inplace,  number=10000):.2f} sec")
    print(f"python sorted(): {      timeit.timeit(test_python_sorted,       number=10000):.2f} sec")
    print(f"python [].sort(): {     timeit.timeit(test_python_array_sort,   number=10000):.2f} sec")



def sort_demo():
    test_array = random.sample(range(1, 1000000), 10)
    print("\nmerge_sort():")
    print(f"before sorting: {test_array}")
    print(f"after sorting: {merge_sort(test_array)}")

    test_array = random.sample(range(1, 1000000), 10)
    print("\nquick_sort_inplace():")
    print(f"before sorting: {test_array}")
    quick_sort_inplace(test_array, 0, len(test_array)-1)
    print(f"after sorting: {test_array}")
    
    sort_profiling()
