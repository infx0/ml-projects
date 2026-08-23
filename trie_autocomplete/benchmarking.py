"""Compare the runtime of binary-search and trie-based autocomplete strategies.

The script defines representative dictionaries and prefixes, builds trie structures,
runs each autocomplete function repeatedly with a high-resolution timer, and prints
average timings for both small and extended word collections.
"""

import time

from trie_autocomplete import binary_search_autocomplete, trie_autocomplete, insert_trie
from long_dictionary import long_dictionary


def benchmark(func, words, queries, repeat=10):
    start = time.perf_counter()
    for _ in range(repeat):
        for q in queries:
            func(words, q)
    end = time.perf_counter()
    return (end - start) / repeat


dictionary1 = [
    "apple",
    "apricot",
    "avocado",
    "antelope",
    "ant",
    "anchor",
    "banana",
    "band",
    "bamboo",
    "barn",
    "basil",
    "basket",
    "carrot",
    "cat",
    "castle",
    "cactus",
    "canoe",
    "camera",
    "dog",
    "dolphin",
    "dragon",
    "drone",
    "drum",
    "duck",
    "elephant",
    "eagle",
    "emerald",
    "engine",
    "ember",
    "echo",
    "frog",
    "forest",
    "flute",
    "flower",
    "fox",
    "fortress",
    "grape",
    "guitar",
    "goose",
    "gold",
    "garden",
    "galaxy",
    "hat",
    "hammer",
    "horse",
    "house",
    "honey",
    "horizon",
    "ice",
    "iguana",
    "iron",
    "island",
    "ivory",
    "idea",
    "jungle",
    "jaguar",
    "jigsaw",
    "jewel",
    "jasmine",
    "joystick",
    "kite",
    "kangaroo",
    "keyboard",
    "knight",
    "kitchen",
    "kingdom",
    "lion",
    "llama",
    "ladder",
    "lantern",
    "lake",
    "leaf",
    "mango",
    "mountain",
    "mouse",
    "mirror",
    "magnet",
    "meadow",
    "night",
    "notebook",
    "noodle",
    "narwhal",
    "nebula",
    "nectar",
    "orange",
    "octopus",
    "orbit",
    "onion",
    "olive",
    "otter",
    "peach",
    "piano",
    "panda",
    "parrot",
    "pyramid",
    "pearl",
    "queen",
    "quail",
    "quartz",
    "quest",
    "quiver",
    "quokka",
    "river",
    "rabbit",
    "robot",
    "rainbow",
    "rose",
    "rocket",
    "sun",
    "star",
    "stone",
    "storm",
    "snake",
    "spoon",
    "tree",
    "turtle",
    "trumpet",
    "tower",
    "tiger",
    "treasure",
]

## set up
sorted_dictionary1 = sorted(dictionary1)

queries = ["dr", "ca", "qu", "mo", "st", "mou"]

## benchmarking
mean_time = benchmark(
    binary_search_autocomplete, sorted_dictionary1, queries, repeat=1000000
)
print(mean_time)

t = {}
for word in dictionary1:
    insert_trie(t, word)

mean_time = benchmark(trie_autocomplete, t, queries, repeat=1000000)
print(mean_time)

sorted_dictionary2 = sorted(long_dictionary)

mean_time = benchmark(
    binary_search_autocomplete, sorted_dictionary2, queries, repeat=100000
)
print(mean_time)

t = {}
for word in long_dictionary:
    insert_trie(t, word)

mean_time = benchmark(trie_autocomplete, t, queries, repeat=100000)
print(mean_time)
