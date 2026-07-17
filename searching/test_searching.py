from searching import binary_search, linear_search

arr = [1, 2, 3, 4, 5]
arr2 = []

def test_binary_search():
    target = binary_search(arr, 3)
    target2 = binary_search(arr2, 3)
    assert target == 2
    assert target2 == -1

def test_linear_search():
    target = linear_search(arr, 3)
    target2 = linear_search(arr2, 3)
    assert target == 2
    assert target2 == -1
