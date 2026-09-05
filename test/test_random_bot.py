"""Random-bot regressions for preserving attack sources and border armies."""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from actors import Player, Random_Bot
from board import Continent, Territory
from test.test_regressions import assert_consistent, assign


class RandomBotTests(unittest.TestCase):
    def board(self, owners, troops, edges):
        continent = Continent("test", 0)
        territories = [Territory(str(i), continent) for i in range(len(owners))]
        continent.territories = territories
        for left, right in edges:
            territories[left].add_neighbor(territories[right])
            territories[right].add_neighbor(territories[left])
        players = [Random_Bot("red", 0, len(owners)), Player("blue", 1, len(owners))]
        assign(territories, players, owners, troops)
        players[0].rng = Mock()
        players[0].rng.choice.return_value = territories[0]
        players[0].rng.randint.return_value = 1
        return territories, players

    def test_declined_attack_still_fortifies_inland_source(self):
        territories, players = self.board([0, 0, 1], [2, 2, 1], [(0, 1), (1, 2)])
        players[0].make_move(players, territories)
        self.assertEqual([t.troop_count for t in territories], [1, 8, 1])
        self.assertEqual(players[0].hand.count, 0)
        assert_consistent(self, territories, players)

    def test_zero_attack_budget_also_moves_inland_army_forward(self):
        territories, players = self.board([0, 0, 1], [2, 2, 1], [(0, 1), (1, 2)])
        players[0].rng.randint.return_value = 0
        players[0].make_move(players, territories)
        self.assertEqual([t.troop_count for t in territories], [1, 8, 1])

    def test_fortification_can_cross_multiple_friendly_territories(self):
        territories, players = self.board(
            [0, 0, 0, 1], [7, 20, 2, 1], [(0, 1), (1, 2), (2, 3)]
        )
        players[0]._fortify(territories[0])
        self.assertEqual([t.troop_count for t in territories], [1, 20, 8, 1])
        assert_consistent(self, territories, players)

    def test_conquered_border_is_not_stripped_at_attack_limit(self):
        territories, players = self.board([0, 1, 1], [10, 1, 3], [(0, 1), (1, 2)])
        with patch(
            "atomic_actions.get_dice_bag",
            return_value=(np.array([6, 6, 6]), np.array([1]), 4),
        ):
            players[0].make_move(players, territories)
        self.assertIs(territories[1].owner, players[0])
        self.assertEqual(territories[0].troop_count, 1)
        self.assertEqual(territories[1].troop_count, 13)
        self.assertEqual(players[0].hand.count, 1)
        assert_consistent(self, territories, players)

    def test_refused_attack_holds_exposed_border(self):
        territories, players = self.board([0, 0, 1], [5, 40, 100], [(0, 1), (0, 2)])
        players[0].make_move(players, territories)
        self.assertEqual([t.troop_count for t in territories], [10, 40, 100])
        assert_consistent(self, territories, players)

    def test_lost_attack_leaves_source_garrison_untouched(self):
        territories, players = self.board([0, 1, 0], [4, 1, 2], [(0, 1), (0, 2)])
        with patch(
            "atomic_actions.get_dice_bag",
            return_value=(np.array([1]), np.array([6]), 2),
        ):
            players[0].make_move(players, territories)
        self.assertEqual([t.troop_count for t in territories], [1, 1, 2])
        self.assertEqual(players[0].hand.count, 0)
        assert_consistent(self, territories, players)

    def test_cannot_fortify_through_enemy_to_disconnected_border(self):
        territories, players = self.board(
            [0, 0, 1, 0], [7, 2, 1, 40], [(0, 1), (1, 2), (2, 3)]
        )
        players[0]._fortify(territories[0])
        self.assertEqual([t.troop_count for t in territories], [1, 8, 1, 40])

    def test_no_border_means_no_fortification(self):
        territories, players = self.board([0, 0, 0], [7, 2, 1], [(0, 1), (1, 2)])
        players[0]._fortify(territories[0])
        self.assertEqual([t.troop_count for t in territories], [7, 2, 1])


if __name__ == "__main__":
    unittest.main()
