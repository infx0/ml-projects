from a_star_search import off_world, get_terrain

nano_world = [
    ['🌾', '🌲', '⛰'],
    ['🐊', '🌋', '🌾']
]

micro_world = [
    ['🌾', '🌲', '🌲'],
    ['🌾', '🌾', '🌾'],
    ['🌲', '🌲', '🌾']
]

def test_off_world():
    y = 0; x = 0; 
    assert off_world(position=(y, x), world=nano_world) == False
    y = 0; x = -1
    assert off_world(position=(y, x), world=nano_world) == True
    y = -1; x = 0; 
    assert off_world(position=(y, x), world=nano_world) == True
    y = 0; x = 3; 
    assert off_world(position=(y, x), world=nano_world) == True
    y = 2; x = 0
    assert off_world(position=(y, x), world=nano_world) == True

def test_get_terrain():
    y = 0; x = 0
    assert get_terrain(position=(y, x), world=nano_world) == "🌾"
    y = 0; x = 1
    assert get_terrain(position=(y, x), world=nano_world) == "🌲"
    y = 1; x = 0
    assert get_terrain(position=(y, x), world=nano_world) == "🐊"