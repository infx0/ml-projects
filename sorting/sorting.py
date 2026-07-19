def bubble_sort(arr):
    """
        - The outer loop iterates 'n' times, where 'n' is the length of the array.
    - The inner loop also iterates 'n' times in the worst case.
    - The comparisons and swapping operations inside the inner loop are constant time ('O(1)').

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
    Explanation: Merge sort successively splits an array in half until there is one or zero elements per array, and then successively merges the arrays back together, sorting at each step. In the comments above, I applied the master theorem to show n log n complexity for all notations.
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
    Explanation: Selection sort goes through each element in an array and compares all successive elements. If one of the successive elements is less than the current element, it's "selected" and swapped. Therefore, best, worst, and average case complexity are all n^2 because regardless of how sorted the array is, all elements are still scanned.
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
    Explanation: Heap sort builds a max heap, and then successively moves the largest element to the end of the array. The total time complexity will be O(n) + O(n log n) = O(n log n).
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
