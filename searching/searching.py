def binary_search(arr, target):
    """
    Explanation: Binary search locates the array index of "target" by starting at the array midpoint (assuming the array has been presorted), and iteratively selecting either the upper half of the array if the midpoint is higher than the target, or the lower half of the array if the midpoint is lower than the target. When the midpoint is equal to "target", the index of the array is returned, otherwise return a -1 if "target" can't be found. Best-case complexity is Ω(1) if "target" is the first midpoint, otherwise worst and average-case complexity is O(log n) and Θ(log n) because the search space is halved at each pass through the while loop.
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
    Explanation: Linear Search scans through an array, looking for a particular element, "target". If target is found, the array index at which it's located is returned, otherwise a -1 is returned. This algorithm has worst-case O(n), average-case Θ(n), and best-case Ω(1), because the contents of the array are scanned sequentially, and the target could be located in any element. The best-case time is when "target" is in the first element. All operations within the "for" loop execute in constant time.
    """
    for i in range(len(arr)):  # O(n)
        if arr[i] == target:  # O(1)
            return i  # O(1)
    return -1  # O(1)


# Total Runtime Complexity:
# Big O notation: O(n)
# Big Omega notation: Ω(1)
# Big Theta notation: Θ(n)
