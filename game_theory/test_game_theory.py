from game_theory import (
    successors,
    terminal,
    get_dominated_columns,
    get_dominated_rows,
    frontier_setup,
)


def test_successors():
    assert successors(
        current_state=([0, 1], [0, 1]), dominated_rows=[0], dominated_cols=[0]
    ) == [([1], [0, 1]), ([0, 1], [1])]
    assert successors(
        current_state=([0, 1], [0, 1]), dominated_rows=[0, 1], dominated_cols=[0]
    ) == [([1], [0, 1]), ([0], [0, 1]), ([0, 1], [1])]
    assert successors(
        current_state=([0, 1], [0, 1]), dominated_rows=[0], dominated_cols=[0, 1]
    ) == [([1], [0, 1]), ([0, 1], [1]), ([0, 1], [0])]
    assert successors(
        current_state=([0, 1], [0, 1]), dominated_rows=[0, 1], dominated_cols=[0, 1]
    ) == [([1], [0, 1]), ([0], [0, 1]), ([0, 1], [1]), ([0, 1], [0])]


def test_terminal():
    assert not terminal(current_state=([0, 1], [0, 1]))
    assert terminal(current_state=([1], [1]))
    assert not terminal(current_state=([0, 1], [2, 3, 4]))


def test_get_dominated_columns():
    game = [[(0, 0), (0, 1)], [(0, 0), (0, 1)]]
    assert get_dominated_columns(
        game=game, current_state=([0, 1], [0, 1]), weak=False
    ) == [0]
    game = [[(0, 1), (0, 0)], [(0, 1), (0, 0)]]
    assert get_dominated_columns(
        game=game, current_state=([0, 1], [0, 1]), weak=False
    ) == [1]
    game = [[(0, 0), (0, 0)], [(0, 0), (0, 0)]]
    assert (
        get_dominated_columns(game=game, current_state=([0, 1], [0, 1]), weak=False)
        == []
    )
    game = [[(0, 2), (0, 1), (0, 0)], [(0, 2), (0, 1), (0, 0)]]
    assert get_dominated_columns(
        game=game, current_state=([0, 1], [0, 1, 2]), weak=False
    ) == [1, 2]


def test_get_dominated_rows():
    game = [[(0, 0), (0, 0)], [(1, 0), (1, 0)]]
    assert get_dominated_rows(
        game=game, current_state=([0, 1], [0, 1]), weak=False
    ) == [0]
    game = [[(1, 0), (1, 0)], [(0, 0), (0, 0)]]
    assert get_dominated_rows(
        game=game, current_state=([0, 1], [0, 1]), weak=False
    ) == [1]
    game = [[(0, 0), (0, 0)], [(0, 0), (0, 0)]]
    assert (
        get_dominated_rows(game=game, current_state=([0, 1], [0, 1]), weak=False) == []
    )
    game = [[(2, 0), (2, 0)], [(1, 0), (1, 0)], [(0, 0), (0, 0)]]
    assert get_dominated_rows(
        game=game, current_state=([0, 1, 2], [0, 1]), weak=False
    ) == [1, 2]


def test_frontier_setup():
    assert frontier_setup(game=[[(0, 0), (0, 0)]]) == [([0], [0, 1])]
    assert frontier_setup(game=[[(0, 0), (0, 0)], [(0, 0), (0, 0)]]) == [
        ([0, 1], [0, 1])
    ]
    assert frontier_setup(game=[[(0, 0)], [(0, 0)]]) == [([0, 1], [0])]
