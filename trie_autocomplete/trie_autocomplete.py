"""
This module compares two ways to implement autocomplete for a prefix.

The binary-search path works from a sorted list of words. It first uses
binary_search_autocomplete to find any word that starts with the requested
prefix, then expand_from walks left and right from that match to collect the
neighboring words that share the same prefix.

The trie path stores words in nested dictionaries, one character per level,
with END marking the end of a valid word. insert_trie builds that structure,
search_prefix walks down to the node represented by the requested prefix, and
collect_words recursively gathers all complete words below that node.
trie_autocomplete ties those pieces together by returning every stored word
that begins with the given prefix, or an empty list when the prefix is absent.
"""

END = "_end_"


def expand_from(words, mid, prefix):
    """
    Expands from a starting index in the sorted word list to collect all neighboring
    words that match a prefix. It checks the starting index, moves left while words
    continue matching, and then moves right from the starting index to find the rest of
    the contiguous matching group.

    Args:
        words (list[str]): The sorted list of words to search.
        mid (int): The starting index found by the binary search.
        prefix (str): The prefix to compare against each word.

    Returns:
        list[str]: The list of matching words found around the starting index.
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
    Performs autocomplete on a sorted list of words using binary search. It searches
    for one word that starts with the given prefix, and then uses expand_from to gather
    the adjacent words that share that prefix.

    Args:
        words (list[str]): The sorted list of words to search.
        prefix (str): The prefix to autocomplete.

    Returns:
        list[str]: The list of words that start with the prefix, or an empty list if
            no matches are found.
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
    Adds a word to a trie implemented with nested dictionaries. Each character in the
    word becomes a dictionary key leading to another node, and the END marker is used
    to show that a complete word ends at the current node.

    Args:
        trie (dict): The trie to add the word to.
        word (str): The word to insert.

    Returns:
        dict | None: The completed trie node when the recursion reaches the end of the
            word. Otherwise the trie is modified in-place.
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
    """
    Searches a trie for the node matching the requested prefix. It follows the trie one
    character at a time and stops early if any character in the prefix is not present.

    Args:
        trie (dict): The trie to search.
        prefix (str): The prefix to follow through the trie.

    Returns:
        dict | None: The trie node for the prefix, or None if the prefix is not
            present.
    """
    node = trie
    for char in prefix:
        if char not in node:
            return None
        node = node[char]
    return node


def collect_words(node, prefix):
    """
    Recursively collects all complete words below a trie node. If the current node
    contains the END marker, the current prefix is added as a complete word, and child
    nodes are used to build longer matching words.

    Args:
        node (dict): The trie node to collect words from.
        prefix (str): The word fragment represented by the current node.

    Returns:
        list[str]: The complete words that can be built from the current trie node.
    """
    results = []
    if END in node:
        results.append(prefix)
    for char, child in node.items():
        if END not in node:
            results += collect_words(child, prefix + char)
    return results


def trie_autocomplete(trie, prefix):
    """
    Performs autocomplete against a trie for a given prefix. It first finds the node
    matching the prefix, and then collects every complete word below that node.

    Args:
        trie (dict): The trie to search.
        prefix (str): The prefix to autocomplete.

    Returns:
        list[str]: The stored words that start with the prefix, or an empty list if the
            prefix is not present.
    """
    node = search_prefix(trie, prefix)
    if not node:
        return []
    return collect_words(node, prefix)
