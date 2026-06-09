END = "_end_"


def expand_from(words, mid, prefix):
    """
    uses a starting index in the list of words provdided by the call from the binary search and expands around that index as appropriate until all words matching the prefix are found.
    """
    results = []
    i = mid
    while i >= 0 and words[i].startswith(prefix):
        results.append(words[i])
        i -= 1
    j = mid + 1
    while j < len(words) and words[j].startswith(prefix):
        results.append(words[j])
        j += 1
    return results


def binary_search_autocomplete(words, prefix):
    """
    performs a binary search algorithm for autocomplete on a sorted list of string words to find a starting point for matching the user-supplied prefix to a group of words with matching prefixes. Runs with O(N) complexity, where N is the length of the valid word dictionary.
    """
    left = 0
    right = len(words) - 1
    results = []

    while left <= right:
        mid = (left + right) // 2
        if words[mid].startswith(prefix):
            results = expand_from(words, mid, prefix)
            break
        elif words[mid] < prefix:
            left = mid + 1
        else:
            right = mid - 1
    return results


def insert_trie(trie, word):
    """
    adds to a trie data structure, mechanized with hierarchical dicts, based on the input word. Used for initializating trie structures in unit tests and benchmarks.
    """
    if not word:
        trie = {END: True}
        return trie
    else:
        first_char = word[0]
        rest = word[1:]
        if first_char not in trie:
            trie[first_char] = {}
        trie[first_char] = insert_trie(trie[first_char], rest)
        return


def search_prefix(trie, prefix):
    node = trie
    for char in prefix:
        if char not in node:
            return None
        node = node[char]
    return node


def collect_words(node, prefix):
    results = []
    if END in node:
        results.append(prefix)
    for char, child in node.items():
        if END not in node:
            results += collect_words(child, prefix + char)
    return results


def trie_autocomplete(trie, prefix):
    node = search_prefix(trie, prefix)
    if not node:
        return []
    return collect_words(node, prefix)
