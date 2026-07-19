from a_star_search import (
    off_world,
    get_terrain,
    get_grid_squares,
    heuristic,
    COSTS,
    update_path_cost,
    get_moves,
    get_path_coords,
)

nano_world = [["🌾", "🌲", "⛰"], ["🐊", "🌋", "🌾"]]

micro_world = [["🌾", "🌲", "🌲"], ["🌾", "🌾", "🌾"], ["🌲", "🌲", "🌾"]]


def test_off_world():
    y = 0
    x = 0
    assert not off_world(position=(y, x), world=nano_world)
    y = 0
    x = -1
    assert off_world(position=(y, x), world=nano_world)
    y = -1
    x = 0
    assert off_world(position=(y, x), world=nano_world)
    y = 0
    x = 3
    assert off_world(position=(y, x), world=nano_world)
    y = 2
    x = 0
    assert off_world(position=(y, x), world=nano_world)


def test_get_terrain():
    y = 0
    x = 0
    assert get_terrain(position=(y, x), world=nano_world) == "🌾"
    y = 0
    x = 1
    assert get_terrain(position=(y, x), world=nano_world) == "🌲"
    y = 1
    x = 0
    assert get_terrain(position=(y, x), world=nano_world) == "🐊"


def test_get_grid_squares():
    assert get_grid_squares(position=(0, 0), goal=(2, 2)) == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 2),
        (2, 2),
    ]
    assert get_grid_squares(position=(2, 0), goal=(2, 2)) == [(2, 0), (2, 1), (2, 2)]
    assert get_grid_squares(position=(0, 0), goal=(2, 0)) == [(0, 0), (1, 0), (2, 0)]
    assert get_grid_squares(position=(2, 2), goal=(0, 0)) == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 2),
        (2, 2),
    ]


def test_heuristic():
    assert heuristic(position=(0, 0), goal=(0, 0), world=micro_world, costs=COSTS) == 1
    assert heuristic(position=(0, 0), goal=(2, 2), world=micro_world, costs=COSTS) == 9
    assert heuristic(position=(0, 0), goal=(0, 2), world=micro_world, costs=COSTS) == 7
    assert heuristic(position=(2, 0), goal=(0, 0), world=micro_world, costs=COSTS) == 5
    assert (
        heuristic(position=(1, 0), goal=(1, 2), world=nano_world, costs=COSTS) == 107
    )  # test with mountain


def test_update_path_cost():
    assert (
        update_path_cost(
            explored=[(0, 0)], current_state=(1, 1), costs=COSTS, world=micro_world
        )
        == 1
    )
    assert (
        update_path_cost(
            explored=[(0, 0), (0, 1)],
            current_state=(1, 1),
            costs=COSTS,
            world=micro_world,
        )
        == 4
    )
    assert (
        update_path_cost(
            explored=[(0, 0), (1, 0)],
            current_state=(2, 0),
            costs=COSTS,
            world=micro_world,
        )
        == 4
    )


def test_get_moves():
    assert get_moves(explored=[(0, (2, 2)), (0, (3, 2))]) == [(0, 1)]
    assert get_moves(explored=[(0, (2, 2)), (0, (1, 2))]) == [(0, -1)]
    assert get_moves(explored=[(0, (2, 2)), (0, (2, 3))]) == [(1, 0)]
    assert get_moves(explored=[(0, (2, 2)), (0, (2, 1))]) == [(-1, 0)]
    assert get_moves(
        explored=[
            (0, (0, 0)),
            (2, (1, 0)),
            (3, (2, 0)),
            (4, (3, 0)),
            (5, (3, 1)),
            (6, (3, 2)),
            (7, (3, 3)),
            (8, (3, 4)),
            (9, (3, 5)),
            (10, (3, 6)),
            (11, (4, 6)),
            (12, (5, 6)),
            (13, (6, 6)),
        ]
    ) == [
        (0, 1),
        (0, 1),
        (0, 1),
        (1, 0),
        (1, 0),
        (1, 0),
        (1, 0),
        (1, 0),
        (1, 0),
        (0, 1),
        (0, 1),
        (0, 1),
    ]


def test_get_path_coordS():
    assert get_path_coords(start=(2, 2), path=[(1, 0)]) == [(2, 2), (3, 2)]
    assert get_path_coords(start=(2, 2), path=[(-1, 0)]) == [(2, 2), (1, 2)]
    assert get_path_coords(start=(2, 2), path=[(0, 1)]) == [(2, 2), (2, 3)]
    assert get_path_coords(start=(2, 2), path=[(0, -1)]) == [(2, 2), (2, 1)]
    assert get_path_coords(
        start=(0, 0),
        path=[
            (0, 1),
            (0, 1),
            (0, 1),
            (1, 0),
            (1, 0),
            (1, 0),
            (1, 0),
            (1, 0),
            (1, 0),
            (0, 1),
            (0, 1),
            (0, 1),
        ],
    ) == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 3),
        (2, 3),
        (3, 3),
        (4, 3),
        (5, 3),
        (6, 3),
        (6, 4),
        (6, 5),
        (6, 6),
    ]
