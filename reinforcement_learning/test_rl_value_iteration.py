from rl_value_iteration import (
    init_rewards,
    read_world,
    init_val_matrix,
    find_mountains,
    valid_pos,
    convergence,
    best_action_value,
)

costs = {".": -1, "*": -3, "^": -5, "~": -7}
cardinal_moves = [(0, -1), (1, 0), (0, 1), (-1, 0)]
small_world = read_world("small.txt")


def test_init_rewards():

    r = init_rewards(
        world=small_world,
        costs=costs,
        goal_pos=(len(small_world[0]) - 1, len(small_world) - 1),
        goal_reward=10,
    )
    assert (len(small_world[0]) - 1, len(small_world) - 1) in r
    assert r[(len(small_world[0]) - 1, len(small_world) - 1)] == 10
    assert max(r.values()) == 10
    assert min(r.values()) == -3


def test_init_val_matrix():
    value = init_val_matrix(small_world)
    assert [v == 0.0 for _, v in value]
    assert (len(small_world), len(small_world[0])) not in value
    assert (len(small_world[0]), len(small_world)) not in value
    assert (3, 3) not in value  # mountain


def test_find_mountains():
    m = find_mountains(small_world)
    assert len(m) == 1
    assert (3, 3) in m
    assert (0, 0) not in m


def test_valid_pos():
    mountains = find_mountains(small_world)
    assert valid_pos(small_world, mountains, (-1, 1)) == False
    assert valid_pos(small_world, mountains, (0, -1)) == False
    assert valid_pos(small_world, mountains, (0, 0)) == True
    assert valid_pos(small_world, mountains, (3, 3)) == False
    assert valid_pos(small_world, mountains, (7, 7)) == False
    assert valid_pos(small_world, mountains, (7, 5)) == False


def test_convergence():
    value1 = {(0, 0): 0.5}
    value2 = {(1, 1): 0.5}
    assert convergence(value1, value2, 0.1) == None
    value2 = {(0, 0): 0.5}
    assert convergence(value1, value2, 0.5) == True
    value2 = {(0, 0): 1.0}
    assert convergence(value1, value2, 0.1) == False
    value1 = {(0, 0): 0.5, (0, 1): 1.0}
    value2 = {(0, 0): 0.5, (0, 1): 1.0}
    assert convergence(value1, value2, 0.1) == True


def test_best_action_value():
    world = [[".", "."], [".", "."]]
    mountains = []
    rewards = {(0, 0): 0.0, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 0.0}
    gamma = 0.9
    V_last = {(0, 0): 0.0, (0, 1): 10.0, (1, 0): 1.0, (1, 1): 0.0}
    curr_pos = (0, 0)
    assert best_action_value(
        curr_pos, cardinal_moves, world, mountains, rewards, gamma, V_last
    )[1] == (0, 1)
    V_last = {(0, 0): 0.0, (0, 1): 1.0, (1, 0): 10.0, (1, 1): 0.0}
    assert best_action_value(
        curr_pos, cardinal_moves, world, mountains, rewards, gamma, V_last
    )[1] == (1, 0)
    curr_pos = (1, 1)
    assert best_action_value(
        curr_pos, cardinal_moves, world, mountains, rewards, gamma, V_last
    )[1] == (0, -1)
