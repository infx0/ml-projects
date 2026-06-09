from trie_autocomplete import expand_from, binary_search_autocomplete, insert_trie, search_prefix, collect_words, trie_autocomplete, END

def test_expand_from():
    w = ["car", "cat", "dog"]
    assert sorted(expand_from(w, 1, "ca")) == ["car", "cat"]
    assert expand_from(w, 2, "dog") == ["dog"]
    assert expand_from(w, 0, "z") == []

def test_binary_search_autocomplete():
    w = ["ant", "bat", "car", "cat", "dog"]
    assert sorted(binary_search_autocomplete(w, "ca")) == ["car", "cat"]
    assert binary_search_autocomplete(w, "dog") == ["dog"]
    assert binary_search_autocomplete(w, "z") == []

def test_insert_trie():
    assert insert_trie({}, "cat") == {'c': {'a': {'t': {END: True}}}}
    assert END in insert_trie({}, "")
    assert "c" in insert_trie({}, "car")

def test_trie_search():
    t = insert_trie({}, "dog")
    assert search_prefix(t, "do") == {'g': {END: True}}
    assert search_prefix(t, "dog") == {END: True}
    assert search_prefix(t, "cat") is None

def test_collect_words():
    t = {}
    insert_trie(t, "cat")
    insert_trie(t, "car")
    assert sorted(collect_words(t, "")) == ["car", "cat"]
    assert collect_words(search_prefix(t, "ca"), "ca") in [["car", "cat"], ["cat", "car"]]
    assert collect_words(search_prefix(t, "dog") or {}, "dog") == []

def test_trie_autocomplete():
    t = {}
    insert_trie(t, "cat")
    insert_trie(t, "car")
    insert_trie(t, "dog")
    assert sorted(trie_autocomplete(t, "ca")) == ["car", "cat"]
    assert trie_autocomplete(t, "dog") == ["dog"]
    assert trie_autocomplete(t, "z") == []