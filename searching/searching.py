"""
This script implements two basic array search algorithms and documents their
runtime complexity. The binary_search function assumes the input array is
already sorted, repeatedly checks the midpoint of the current search range,
and narrows the range until it either finds the target value or returns -1.
The linear_search function makes no sorting assumption; it checks each element
from left to right and returns the first matching index, or -1 when the target
is not present.
"""


def binary_search(arr, target):
    """
    Searches a sorted array for a target value using binary search. The search
    starts at the midpoint of the current range, then repeatedly narrows the
    range to the lower or upper half until the target is found or the range is
    exhausted.

    Args:
        arr (list): The sorted array to search.
        target: The value to locate in the array.

    Returns:
        int: Returns the index of the target value if found, or -1 otherwise.
    """
    low = 0  # O(1)
    high = len(arr) - 1  # O(1)

    while low <= high:  # O(log n)
        mid = (low + high) // 2  # O(1)
        mid_val = arr[mid]  # O(1)

        if mid_val == target:  # O(1)
            return mid  # O(1)
        elif mid_val < target:  # O(1)
            low = mid + 1  # O(1)
        else:
            high = mid - 1  # O(1)

    return -1  # O(1)


# Total Runtime Complexity:
# Big O notation: O(log n)
# Big Omega notation:  Ω(1)
# Big Theta notation: Θ(log n)


def linear_search(arr, target):
    """
    Searches an array for a target value using linear search. The search checks
    each element from left to right and stops when it finds the first matching
    value.

    Args:
        arr (list): The array to search.
        target: The value to locate in the array.

    Returns:
        int: Returns the index of the first matching value if found, or -1
            otherwise.
    """
    for i in range(len(arr)):  # O(n)
        if arr[i] == target:  # O(1)
            return i  # O(1)
    return -1  # O(1)


# Total Runtime Complexity:
# Big O notation: O(n)
# Big Omega notation: Ω(1)
# Big Theta notation: Θ(n)
