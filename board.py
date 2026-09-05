import random

import matplotlib.pyplot as plt
import networkx as nx

from actors import Neutral_Bot, Player, Random_Bot
from atomic_actions import fortify_destinations

pos = {
    "Alaska": (0, 10),
    "Northwest Territory": (1, 9),
    "Greenland": (4, 10),
    "Alberta": (1, 8),
    "Ontario": (2, 8),
    "Quebec": (3, 8),
    "Western US": (1, 7),
    "Eastern US": (2, 7),
    "Central America": (2, 6),
    "Iceland": (5, 10),
    "Scandinavia": (6, 10),
    "Ukraine": (7, 10),
    "Great Britain": (5, 9),
    "Northern Europe": (6, 9),
    "Western Europe": (5, 8),
    "Southern Europe": (6, 8),
    "Ural": (8, 9),
    "Siberia": (9, 9),
    "Yakutsk": (10, 9),
    "Kamchatka": (11, 9),
    "Irkutsk": (10, 8),
    "Mongolia": (9, 8),
    "Japan": (11, 8),
    "Afghanistan": (8, 8),
    "China": (9, 7),
    "Middle East": (7, 7),
    "India": (8, 7),
    "Siam": (9, 6),
    "Venezuela": (2, 5),
    "Peru": (2, 4),
    "Brazil": (3, 4),
    "Argentina": (2, 3),
    "North Africa": (5, 5),
    "Egypt": (6, 5),
    "East Africa": (6, 4),
    "Congo": (5, 4),
    "South Africa": (5, 3),
    "Madagascar": (7, 3),
    "Indonesia": (10, 5),
    "New Guinea": (11, 5),
    "West Australia": (10, 4),
    "East Australia": (11, 4),
}


class Territory:
    def __init__(self, name, continent, owner=None, owner_color=None):
        self.name = name
        self.continent = continent
        self.owner = owner
        self.owner_color = owner_color
        self.troop_count = 0
        self.key = -1
        self.neighbors = []
        self.neighbor_names = []

    def add_neighbor(self, neighbor):
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)
            self.neighbor_names.append(neighbor.name)


class Continent:
    def __init__(self, name, bonus_troop_count):
        self.name = name
        self.bonus_troop_count = bonus_troop_count
        self.territories = []

    def add_territory(self, territory):
        if territory not in self.territories:
            self.territories.append(territory)


def initialize(territories, players, player_colors, rng=None):
    """Assign territories round-robin and spend each owner's troop budget."""
    rng = rng or random
    num_players = len(players)
    if not 2 <= num_players <= 6 or num_players > len(territories):
        raise ValueError("Boards require 2-6 players and at least one territory each")
    initial_troops = 50 - 5 * num_players
    for player in players:
        player.total_troops = initial_troops
        player.placeable_troops = 0
        player.territories = [0] * len(territories)
        player.territory_count = 0

    shuffled_players = players.copy()
    rng.shuffle(shuffled_players)
    for index, territory in enumerate(territories):
        owner = shuffled_players[index % num_players]
        territory.key = index
        territory.owner = owner
        territory.owner_color = owner.name
        territory.troop_count = 1
        owner.territories[index] = 1
        owner.territory_count += 1

    remaining = {player: initial_troops - player.territory_count for player in players}
    while any(remaining.values()):
        for territory in territories:
            owner = territory.owner
            probability = 0.25 if territory.troop_count == 1 else 0.50
            if remaining[owner] > 0 and rng.random() < probability:
                territory.troop_count += 1
                remaining[owner] -= 1


def _create_players(num_players, bot_types, colors, num_territories, rng, dice_rng):
    if not isinstance(num_players, int) or not 2 <= num_players <= 6:
        raise ValueError("num_players must be between 2 and 6")
    if num_players > num_territories:
        raise ValueError("Each player needs at least one territory")
    if len(bot_types) != num_players - 1:
        raise ValueError("Provide one bot type for each opponent")
    colors = (
        list(colors)
        if colors is not None
        else ["red", "blue", "green", "yellow", "purple", "pink"][:num_players]
    )
    if len(colors) != num_players or len(set(colors)) != num_players:
        raise ValueError("Provide one distinct color per player")
    player_types = {None: Player, "Neutral": Neutral_Bot, "Random": Random_Bot}
    players = []
    for index, bot_type in enumerate([None] + list(bot_types)):
        if bot_type not in player_types:
            raise ValueError(f"Unsupported bot type: {bot_type!r}")
        player = player_types[bot_type](colors[index], index, num_territories)
        player.rng = rng or random
        player.dice_rng = dice_rng
        players.append(player)
    return players, colors


# colors must be passed with the same ordering as player turns
def create_board(num_players, bot_types, colors=None, rng=None, dice_rng=None):
    continents = [
        Continent("North America", 5),
        Continent("Europe", 5),
        Continent("Asia", 7),
        Continent("South America", 2),
        Continent("Africa", 3),
        Continent("Australia", 2),
    ]

    continent_index = {
        "North America": 0,
        "Europe": 1,
        "Asia": 2,
        "South America": 3,
        "Africa": 4,
        "Australia": 5,
    }

    territories = [
        Territory("Alaska", continents[0]),
        Territory("Northwest Territory", continents[0]),
        Territory("Greenland", continents[0]),
        Territory("Alberta", continents[0]),
        Territory("Ontario", continents[0]),
        Territory("Quebec", continents[0]),
        Territory("Western US", continents[0]),
        Territory("Eastern US", continents[0]),
        Territory("Central America", continents[0]),
        Territory("Iceland", continents[1]),
        Territory("Scandinavia", continents[1]),
        Territory("Ukraine", continents[1]),
        Territory("Great Britain", continents[1]),
        Territory("Northern Europe", continents[1]),
        Territory("Western Europe", continents[1]),
        Territory("Southern Europe", continents[1]),
        Territory("Ural", continents[2]),
        Territory("Siberia", continents[2]),
        Territory("Yakutsk", continents[2]),
        Territory("Kamchatka", continents[2]),
        Territory("Irkutsk", continents[2]),
        Territory("Mongolia", continents[2]),
        Territory("Japan", continents[2]),
        Territory("Afghanistan", continents[2]),
        Territory("China", continents[2]),
        Territory("Middle East", continents[2]),
        Territory("India", continents[2]),
        Territory("Siam", continents[2]),
        Territory("Venezuela", continents[3]),
        Territory("Peru", continents[3]),
        Territory("Brazil", continents[3]),
        Territory("Argentina", continents[3]),
        Territory("North Africa", continents[4]),
        Territory("Egypt", continents[4]),
        Territory("East Africa", continents[4]),
        Territory("Congo", continents[4]),
        Territory("South Africa", continents[4]),
        Territory("Madagascar", continents[4]),
        Territory("Indonesia", continents[5]),
        Territory("New Guinea", continents[5]),
        Territory("West Australia", continents[5]),
        Territory("East Australia", continents[5]),
    ]

    territory_dict = {territory.name: territory for territory in territories}
    connections = [
        ("Alaska", "Northwest Territory"),
        ("Alaska", "Kamchatka"),
        ("Northwest Territory", "Greenland"),
        ("Northwest Territory", "Alberta"),
        ("Northwest Territory", "Ontario"),
        ("Greenland", "Quebec"),
        ("Greenland", "Ontario"),
        ("Greenland", "Iceland"),
        ("Alberta", "Ontario"),
        ("Alberta", "Western US"),
        ("Alberta", "Alaska"),
        ("Ontario", "Quebec"),
        ("Ontario", "Western US"),
        ("Ontario", "Eastern US"),
        ("Quebec", "Eastern US"),
        ("Western US", "Eastern US"),
        ("Western US", "Central America"),
        ("Eastern US", "Central America"),
        ("Central America", "Venezuela"),
        ("Iceland", "Scandinavia"),
        ("Iceland", "Great Britain"),
        ("Scandinavia", "Ukraine"),
        ("Scandinavia", "Northern Europe"),
        ("Scandinavia", "Great Britain"),
        ("Ukraine", "Northern Europe"),
        ("Ukraine", "Southern Europe"),
        ("Ukraine", "Ural"),
        ("Ukraine", "Middle East"),
        ("Ukraine", "Afghanistan"),
        ("Great Britain", "Northern Europe"),
        ("Great Britain", "Western Europe"),
        ("Northern Europe", "Western Europe"),
        ("Northern Europe", "Southern Europe"),
        ("Western Europe", "Southern Europe"),
        ("Southern Europe", "Middle East"),
        ("Ural", "Siberia"),
        ("Ural", "Afghanistan"),
        ("Ural", "China"),
        ("Siberia", "Yakutsk"),
        ("Siberia", "Irkutsk"),
        ("Yakutsk", "Kamchatka"),
        ("Yakutsk", "Irkutsk"),
        ("Kamchatka", "Mongolia"),
        ("Kamchatka", "Japan"),
        ("Irkutsk", "Mongolia"),
        ("Mongolia", "China"),
        ("Mongolia", "Japan"),
        ("Afghanistan", "China"),
        ("Afghanistan", "Middle East"),
        ("Afghanistan", "India"),
        ("China", "India"),
        ("China", "Siam"),
        ("China", "Siberia"),
        ("India", "Siam"),
        ("Middle East", "India"),
        ("Middle East", "Egypt"),
        ("Middle East", "East Africa"),
        ("Venezuela", "Brazil"),
        ("Venezuela", "Peru"),
        ("Brazil", "Peru"),
        ("Brazil", "Argentina"),
        ("Brazil", "North Africa"),
        ("Peru", "Argentina"),
        ("North Africa", "Egypt"),
        ("North Africa", "East Africa"),
        ("North Africa", "Congo"),
        ("North Africa", "Western Europe"),
        ("North Africa", "Southern Europe"),
        ("Egypt", "East Africa"),
        ("Egypt", "Southern Europe"),
        ("East Africa", "Congo"),
        ("East Africa", "South Africa"),
        ("East Africa", "Madagascar"),
        ("Congo", "South Africa"),
        ("South Africa", "Madagascar"),
        ("Indonesia", "New Guinea"),
        ("Indonesia", "West Australia"),
        ("Indonesia", "Siam"),
        ("New Guinea", "West Australia"),
        ("New Guinea", "East Australia"),
        ("West Australia", "East Australia"),
    ]

    for territory1, territory2 in connections:
        territory_dict[territory1].add_neighbor(territory_dict[territory2])
        territory_dict[territory2].add_neighbor(
            territory_dict[territory1]
        )  # Make the connection bidirectional

    for territory in territories:
        continents[continent_index[territory.continent.name]].add_territory(territory)

    players, player_colors = _create_players(
        num_players, bot_types, colors, len(territories), rng, dice_rng
    )
    initialize(territories, players, player_colors, rng)
    return continents, territories, players


def create_board_test(
    num_players, bot_types, colors=None, size=0, rng=None, dice_rng=None
):
    if size not in (0, 1, 2):
        raise ValueError("size must be 0 (small), 1 (medium), or 2 (large)")
    continents = [
        Continent("North America", 5),
        Continent("South America", 2),
    ]

    continent_index = {
        "North America": 0,
        "South America": 1,
    }

    territories = [
        Territory("Alaska", continents[0]),
        Territory("Northwest Territory", continents[0]),
        Territory("Alberta", continents[0]),
        Territory("Ontario", continents[0]),
    ]
    if size >= 1:
        territories += [
            Territory("Greenland", continents[0]),
            Territory("Quebec", continents[0]),
            Territory("Western US", continents[0]),
            Territory("Eastern US", continents[0]),
            Territory("Central America", continents[0]),
        ]
    if size == 2:
        territories += [
            Territory("Venezuela", continents[1]),
            Territory("Peru", continents[1]),
            Territory("Brazil", continents[1]),
            Territory("Argentina", continents[1]),
        ]

    territory_dict = {territory.name: territory for territory in territories}
    connections = [
        ("Alaska", "Northwest Territory"),
        ("Northwest Territory", "Alberta"),
        ("Northwest Territory", "Ontario"),
        ("Alberta", "Ontario"),
        ("Alberta", "Alaska"),
    ]

    if size >= 1:
        connections += [
            ("Greenland", "Quebec"),
            ("Greenland", "Ontario"),
            ("Alberta", "Western US"),
            ("Ontario", "Quebec"),
            ("Ontario", "Western US"),
            ("Ontario", "Eastern US"),
            ("Quebec", "Eastern US"),
            ("Western US", "Eastern US"),
            ("Western US", "Central America"),
            ("Eastern US", "Central America"),
        ]
    if size == 2:
        connections += [
            ("Central America", "Venezuela"),
            ("Venezuela", "Brazil"),
            ("Venezuela", "Peru"),
            ("Brazil", "Peru"),
            ("Brazil", "Argentina"),
            ("Peru", "Argentina"),
        ]

    for territory1, territory2 in connections:
        territory_dict[territory1].add_neighbor(territory_dict[territory2])
        territory_dict[territory2].add_neighbor(
            territory_dict[territory1]
        )  # Make the connection bidirectional

    for territory in territories:
        continents[continent_index[territory.continent.name]].add_territory(territory)

    players, player_colors = _create_players(
        num_players, bot_types, colors, len(territories), rng, dice_rng
    )
    initialize(territories, players, player_colors, rng)
    return continents, territories, players


# returns the board graph
def create_graph(territories, display=False):
    G = nx.Graph()

    for territory in territories:
        G.add_node(territory)

    for territory in territories:
        for neighbor in territory.neighbors:
            G.add_edge(territory, neighbor)

    if display:
        display_graph(G, territories)

    return G


def find_shortest_path(
    graph, from_teritory, to_teritory, player_color, attack_turn=True, display=False
):
    # if attack turn, we cannot attack our own color
    # if fortify turn, we can only fortify through our territory
    def attribute_check(node):
        if node == from_teritory:
            return True
        if attack_turn:
            return node.owner_color != player_color
        else:
            return node.owner_color == player_color

    # Creating a subgraph of only possible legal paths
    nodes_subgraph = [node for node in graph.nodes() if attribute_check(node)]

    subgraph = graph.subgraph(nodes_subgraph)

    path = None
    try:
        path = nx.shortest_path(
            subgraph, from_teritory, to_teritory, weight=None, method="dijkstra"
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass

    if display:
        # Draw the graph with the generated path
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(graph)

        # Create labels for the nodes
        labels = {territory: territory.name for territory in graph.nodes()}

        # Define a color map, where different owner_colors are assigned different colors
        # Here we assume the 'owner_color' attribute is a string and use it directly for coloring
        color_map = {territory: territory.owner_color for territory in graph.nodes()}

        nx.draw_networkx_nodes(graph, pos, node_color=list(color_map.values()))
        nx.draw_networkx_labels(graph, pos, labels=labels)  # add the labels here
        nx.draw_networkx_edges(graph, pos, alpha=0.2)

        if path is not None:
            edges_in_path = [(path[i - 1], path[i]) for i in range(1, len(path))]
            nx.draw_networkx_edges(
                graph, pos, edgelist=edges_in_path, edge_color="red", width=2
            )

        plt.show()

    return path


# Find all possible territories to fortify to given the source territory
def fortify_bfs(territory):
    return fortify_destinations(territory)


def display_graph(
    graph, territories, title="Figure", save=False, blocking_display=True
):
    # TODO
    # better align graph to world map if the input graph isnt a world map

    node_colors = []
    for territory in graph.nodes():
        node_colors.append(territory.owner_color)

    plt.figure(title, figsize=(12, 6))

    # this could be done faster by defining the dict as a dict of objects in the first place
    # but there are bigger fish
    pos_with_objects = {}
    for territory in graph.nodes():
        pos_with_objects[territory] = pos[territory.name]

    labels = {
        territory: territory.name for territory in graph.nodes()
    }  # map the nodes to their names

    nx.draw(
        graph,
        pos_with_objects,
        node_color=node_colors,
        labels=labels,
        with_labels=True,
        node_size=2000,
        font_size=10,
    )

    # Add troop count labels
    for territory in territories:
        x, y = pos[territory.name]
        plt.text(
            x,
            y - 0.1,
            "Troops: " + str(territory.troop_count),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=10,
            color="black",
        )

    if save:
        plt.savefig(title)

    plt.show(block=blocking_display)


if __name__ == "__main__":
    continents, territories, players = create_board(2, ["Random"])
    display_graph(create_graph(territories), territories)
