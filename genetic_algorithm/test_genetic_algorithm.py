from genetic_algorithm import (
    crossover,
    eval_random_draw,
    mutate,
    reproduce,
    get_ascii,
    evaluate,
    pick_parents,
    generate_random_population,
)
import random


def test_crossover():
    assert crossover(
        parent1=["A", "B", "C"], parent2=["D", "E", "F"], crossover_idx=0
    ) == (["D", "E", "F"], ["A", "B", "C"])
    assert crossover(
        parent1=["A", "B", "C"], parent2=["D", "E", "F"], crossover_idx=4
    ) == (["A", "B", "C"], ["D", "E", "F"])
    assert crossover(
        parent1=["A", "B", "C"], parent2=["D", "E", "F"], crossover_idx=2
    ) == (["A", "B", "F"], ["D", "E", "C"])


def test_eval_random_draw():
    assert eval_random_draw(draw_result=0.0, threshold=1.0)
    assert not eval_random_draw(draw_result=1.0, threshold=0.0)
    assert not eval_random_draw(draw_result=0.5, threshold=0.5)


def test_mutate():
    assert mutate(["A"], 0, "B") == ["B"]
    assert mutate(["A", "B", "C"], 2, "B") == ["A", "B", "B"]
    assert mutate(["A"], 0, "A") == ["A"]


def test_reproduce():
    assert reproduce(
        parent1=["A", "B", "C"],
        parent2=["D", "E", "F"],
        crossover_prob=0.0,
        mutate_prob=0.0,
        encoding="ABCDEF",
    ) == (["A", "B", "C"], ["D", "E", "F"])
    assert reproduce(
        parent1=["A", "B", "C"],
        parent2=["D", "E", "F"],
        crossover_prob=1.0,
        mutate_prob=0.0,
        encoding="ABCDEF",
    ) != (["A", "B", "C"], ["D", "E", "F"])
    assert reproduce(
        parent1=["A", "B", "C"],
        parent2=["D", "E", "F"],
        crossover_prob=1.0,
        mutate_prob=1.0,
        encoding="ABCDEF",
    ) != (["A", "B", "C"], ["D", "E", "F"])


def test_get_ascii():
    assert get_ascii(tgt_char="a", rotate=0) == 97
    assert get_ascii(tgt_char="z", rotate=0) == 122
    assert get_ascii(tgt_char="z", rotate=1) == 97


def test_evaluate():
    assert evaluate(population=[["a"]], target_str="a") == {
        "genotype": ["a"],
        "fitness": 1.0,
        "phenotype": "a",
    }
    assert evaluate(population=[["a"]], target_str="b") == {
        "genotype": ["a"],
        "fitness": 0.5,
        "phenotype": "a",
    }
    assert evaluate(population=[["b"]], target_str="a") == {
        "genotype": ["b"],
        "fitness": 0.5,
        "phenotype": "b",
    }
    assert evaluate(population=[["a", "b"], ["a", "c"]], target_str="bb") == {
        "genotype": ["a", "b"],
        "fitness": 0.5,
        "phenotype": "ab",
    }


def test_pick_parents():
    assert pick_parents(
        population=[["a"]],
        subsample_size=1,
        target_str="a",
        num_parents=1,
        flip=False,
        rotate=0,
    ) == (["a"],)
    assert pick_parents(
        population=[["a"]],
        subsample_size=1,
        target_str="a",
        num_parents=3,
        flip=False,
        rotate=0,
    ) == (
        ["a"],
        ["a"],
        ["a"],
    )
    assert pick_parents(
        population=[["a", "b", "c"]],
        subsample_size=2,
        target_str="abc",
        num_parents=2,
        flip=False,
        rotate=0,
    ) == (
        ["a", "b", "c"],
        ["a", "b", "c"],
    )


def test_generate_random_population():
    random.seed(42)
    assert generate_random_population(pop_size=3, genotype_len=1, encoding="A") == [
        ["A"],
        ["A"],
        ["A"],
    ]
    assert generate_random_population(pop_size=0, genotype_len=10, encoding="ABC") == []
    assert generate_random_population(pop_size=2, genotype_len=3, encoding="AB") == [
        ["A", "A", "A"],
        ["A", "A", "B"],
    ]
