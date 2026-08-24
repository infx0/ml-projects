"""
This script implements and analyzes four sorting algorithms: bubble sort,
merge sort, selection sort, and heap sort. Each function takes a list, sorts it
in ascending order, and returns the sorted list. Bubble sort and selection sort
use nested comparisons and have quadratic time complexity. Merge sort splits
the list into smaller halves before merging them back together in sorted order,
giving it n log n complexity. Heap sort first builds a max heap, then repeatedly
moves the largest remaining value into its final position, also giving it
n log n complexity.

The inline comments explain the major steps of each algorithm and record the
Big O, Big Omega, and Big Theta runtime bounds for comparison.
"""


def bubble_sort(arr):
    """
    Sort a list in ascending order using bubble sort. Bubble sort repeatedly
    compares neighboring values and swaps them when they are out of order. The
    outer loop iterates over the list, and the inner loop moves the largest
    remaining value toward the end.

    Args:
        arr: The list of comparable values to sort.

    Returns:
        The same list object, sorted in ascending order.
    """
    n = len(arr)
    for i in range(n):  # O(n)
        for j in range(0, n - i - 1):  # O(n)
            if arr[j] > arr[j + 1]:  # O(1)
                arr[j], arr[j + 1] = arr[j + 1], arr[j]  # O(1)

    # Total Runtime Complexity:
    # Big O notation: O(n^2)
    # Big Omega notation: Ω(n)
    # Big Theta notation: Θ(n^2)
    return arr


def merge_sort(arr):
    """
    Sort a list in ascending order using merge sort. Merge sort splits the list
    into halves until each sublist has one or zero values, then merges those
    sublists back into sorted order.

    Args:
        arr: The list of comparable values to sort.

    Returns:
        The same list object, sorted in ascending order.
    """
    if len(arr) > 1:  # O(1)
        mid = len(arr) // 2  # O(1)
        left_half = arr[:mid]  # O(1)
        right_half = arr[mid:]  # O(1)

        merge_sort(left_half)  # T(n/2)
        merge_sort(right_half)  # T(n/2)

        i = j = k = 0

        while i < len(left_half) and j < len(right_half):  # O(n)
            if left_half[i] < right_half[j]:  # O(1)
                arr[k] = left_half[i]  # O(1)
                i += 1  # O(1)
            else:
                arr[k] = right_half[j]  # O(1)
                j += 1  # O(1)
            k += 1  # O(1)

        while i < len(left_half):  # O(n)
            arr[k] = left_half[i]  # O(1)
            i += 1  # O(1)
            k += 1  # O(1)

        while j < len(right_half):  # O(n)
            arr[k] = right_half[j]  # O(1)
            j += 1  # O(1)
            k += 1  # O(1)

    # 2T(n/2) + O(n), using master theorem with a=2, b=2, n^(log2(2)) = n -> T(n) = Θ(n log n)
    # Since theta is a tight bounds, all notations have the same result.

    # Total Runtime Complexity:
    # Big O notation: O(n log n)
    # Big Omega notation: Ω(n log n)
    # Big Theta notation: Θ(n log n)
    return arr


def selection_sort(arr):
    """
    Sort a list in ascending order using selection sort. Selection sort scans
    the unsorted portion of the list to find the smallest remaining value, then
    swaps it into the current position.

    Args:
        arr: The list of comparable values to sort.

    Returns:
        The same list object, sorted in ascending order.
    """
    n = len(arr)  # O(1)
    for i in range(n - 1):  # O(n)
        min_index = i  # O(1)
        for j in range(i + 1, n):  # O(n)
            if arr[j] < arr[min_index]:  # O(1)
                min_index = j  # O(1)
        arr[i], arr[min_index] = arr[min_index], arr[i]  # O(1)

    # Total Runtime Complexity:
    # Big O notation: O(n^2)
    # Big Omega notation: Ω(n^2)
    # Big Theta notation: Θ(n^2)
    return arr


def heapify(arr, n, i):
    """
    Restore the max-heap property for a subtree.

    Args:
        arr: The list containing the heap values.
        n: The number of heap elements to consider from the front of the list.
        i: The index of the subtree root to heapify.

    Returns:
        The same list object, with the subtree rooted at i adjusted as needed.
    """
    largest = i  # O(1)
    left = 2 * i + 1  # O(1)
    right = 2 * i + 2  # O(1)

    if left < n and arr[left] > arr[largest]:  # O(1)
        largest = left  # O(1)

    if right < n and arr[right] > arr[largest]:  # O(1)
        largest = right  # O(1)

    if largest != i:  # O(1)
        arr[i], arr[largest] = arr[largest], arr[i]  # O(1)
        heapify(arr, n, largest)
    return arr


def heap_sort(arr):
    """
    Sort a list in ascending order using heap sort. Heap sort builds a max
    heap, then repeatedly swaps the largest value into its final position at
    the end of the list.

    Args:
        arr: The list of comparable values to sort.

    Returns:
        The same list object, sorted in ascending order.
    """
    n = len(arr)  # O(1)

    for i in range(n // 2 - 1, -1, -1):  # O(n/2)
        heapify(arr, n, i)  # O(log n)

    for i in range(n - 1, 0, -1):  # O(n)
        arr[i], arr[0] = arr[0], arr[i]  # O(1)
        heapify(arr, i, 0)  # O(log n)

    # Total Runtime Complexity:
    # Big O notation: O(n log n)
    # Big Omega notation: Ω(n log n)
    # Big Theta notation: Θ(n log n)
    return arr
