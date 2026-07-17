from sorting import bubble_sort, merge_sort, selection_sort ,heap_sort

arr = [5, 3, 4, 2, 1]

def test_bubble_sort():
    sorted = bubble_sort(arr)
    assert sorted == [1, 2, 3, 4, 5]

def test_merge_sort():
    sorted = merge_sort(arr)
    assert sorted == [1, 2, 3, 4, 5]

def test_selection_sort():
    sorted = selection_sort(arr)
    assert sorted == [1, 2, 3, 4, 5]

def test_heap_sort():
    sorted = heap_sort(arr)
    assert sorted == [1, 2, 3, 4, 5]