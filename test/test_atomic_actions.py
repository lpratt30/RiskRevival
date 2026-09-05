import unittest
from unittest.mock import Mock, patch

import numpy as np

from atomic_actions import (
    attack_territory,
    generate_troops,
    get_card,
    get_dice_bag,
    place_troops,
    take_cards,
)


class TestGetDiceBag(unittest.TestCase):
    @patch("atomic_actions.np.random.randint")
    def test_dice_bag_size(self, mock_randint):
        mock_randint.return_value = np.array([1, 2, 3, 4, 5, 6])
        attacker_troops = 3
        defender_troops = 3
        attacker_rolls, defender_rolls, bag_size = get_dice_bag(
            attacker_troops, defender_troops
        )
        self.assertEqual(bag_size, attacker_troops + defender_troops)
        self.assertEqual(len(attacker_rolls), attacker_troops)
        self.assertEqual(len(defender_rolls), defender_troops)

    @patch("atomic_actions.np.random.randint")
    def test_attacker_rolls_sorted(self, mock_randint):
        mock_randint.return_value = np.array([6, 2, 3, 4, 5, 1])
        attacker_troops = 3
        defender_troops = 3
        attacker_rolls, defender_rolls, bag_size = get_dice_bag(
            attacker_troops, defender_troops
        )
        self.assertEqual(list(attacker_rolls), [6, 3, 2])
        self.assertEqual(list(defender_rolls), [5, 4, 1])

    @patch("atomic_actions.np.random.randint")
    # This is the test case where I found the code failed. I don't think the last 2 attacker dice should be sorted together with the first defender dice.
    def test_attacker_rolls_sorted_complicated(self, mock_randint):
        mock_randint.return_value = np.array([6, 6, 5, 2, 4, 1, 3, 1, 5, 2, 1, 4])
        attacker_troops = 8
        defender_troops = 4
        attacker_rolls, defender_rolls, bag_size = get_dice_bag(
            attacker_troops, defender_troops
        )
        self.assertEqual(list(attacker_rolls), [6, 6, 5, 4, 2, 1, 3, 1])
        self.assertEqual(list(defender_rolls), [5, 2, 4, 1])

    @patch("atomic_actions.np.random.randint")
    def test_defender_rolls_sorted(self, mock_randint):
        mock_randint.return_value = np.array([1, 2, 3, 4, 5, 6])
        attacker_troops = 3
        defender_troops = 3
        attacker_rolls, defender_rolls, bag_size = get_dice_bag(
            attacker_troops, defender_troops
        )
        self.assertEqual(list(defender_rolls), [5, 4, 6])

    @patch("atomic_actions.np.random.randint")
    def test_return_structure(self, mock_randint):
        mock_randint.return_value = np.array([1, 2, 3, 4, 5, 6])
        attacker_troops = 3
        defender_troops = 3
        attacker_rolls, defender_rolls, bag_size = get_dice_bag(
            attacker_troops, defender_troops
        )
        self.assertIsInstance(attacker_rolls, np.ndarray)
        self.assertIsInstance(defender_rolls, np.ndarray)
        self.assertIsInstance(bag_size, int)


class TestAttackTerritory(unittest.TestCase):
    def setUp(self):
        self.from_territory = Mock()
        self.to_territory = Mock()

        self.from_territory.name = "A"
        self.to_territory.name = "B"

        self.from_territory.neighbor_names = ["B"]
        self.from_territory.owner.key = 1
        self.to_territory.owner.key = 2

        self.from_territory.troop_count = 4
        self.to_territory.troop_count = 2

        self.from_territory.owner.territories = {self.from_territory.name: 1}
        self.to_territory.owner.territories = {self.to_territory.name: 1}
        self.from_territory.owner.territory_count = 1
        self.to_territory.owner.territory_count = 1

        self.from_territory.owner.damage_dealt = {2: 0}
        self.to_territory.owner.damage_received = {1: 0}
        self.from_territory.owner.total_troops = 5
        self.to_territory.owner.total_troops = 3

    @patch("atomic_actions.get_dice_bag")
    def test_non_adjacent_territory(self, mock_get_dice_bag):
        self.from_territory.neighbor_names = ["C"]  # B is not a neighbor
        troops_lost_attacker, troops_lost_defender, attacker_won, is_legal = (
            attack_territory(self.from_territory, self.to_territory, 4)
        )
        self.assertFalse(is_legal)
        self.assertEqual(troops_lost_attacker, 0)
        self.assertEqual(troops_lost_defender, 0)
        self.assertFalse(attacker_won)

    @patch("atomic_actions.get_dice_bag")
    def test_attack_own_territory(self, mock_get_dice_bag):
        self.to_territory.owner.key = 1  # Same owner as from_territory
        troops_lost_attacker, troops_lost_defender, attacker_won, is_legal = (
            attack_territory(self.from_territory, self.to_territory, 4)
        )
        self.assertFalse(is_legal)
        self.assertEqual(troops_lost_attacker, 0)
        self.assertEqual(troops_lost_defender, 0)
        self.assertFalse(attacker_won)

    @patch("atomic_actions.get_dice_bag")
    def test_not_enough_troops_to_attack(self, mock_get_dice_bag):
        self.from_territory.troop_count = 1  # Not enough troops to attack
        troops_lost_attacker, troops_lost_defender, attacker_won, is_legal = (
            attack_territory(self.from_territory, self.to_territory, 1)
        )
        self.assertFalse(is_legal)
        self.assertEqual(troops_lost_attacker, 0)
        self.assertEqual(troops_lost_defender, 0)
        self.assertFalse(attacker_won)

    @patch("atomic_actions.get_dice_bag")
    def test_successful_attack(self, mock_get_dice_bag):
        mock_get_dice_bag.return_value = (np.array([6, 5, 4]), np.array([3, 2]), 5)
        troops_lost_attacker, troops_lost_defender, attacker_won, is_legal = (
            attack_territory(self.from_territory, self.to_territory, 3)
        )
        self.assertTrue(is_legal)
        self.assertEqual(troops_lost_attacker, 0)  # Attackers won
        self.assertEqual(troops_lost_defender, 2)  # All defender troops lost
        self.assertTrue(attacker_won)

    @patch("atomic_actions.get_dice_bag")
    def test_failed_attack_with_three(self, mock_get_dice_bag):
        mock_get_dice_bag.return_value = (np.array([1, 1, 1]), np.array([6, 5]), 5)
        troops_lost_attacker, troops_lost_defender, attacker_won, is_legal = (
            attack_territory(self.from_territory, self.to_territory, 3)
        )
        self.assertTrue(is_legal)
        self.assertEqual(troops_lost_attacker, 3)  # All attacker troops lost
        self.assertEqual(troops_lost_defender, 0)  # No defender troops lost
        self.assertFalse(attacker_won)

    @patch("atomic_actions.get_dice_bag")
    def test_failed_attack_with_two(self, mock_get_dice_bag):
        self.from_territory.troop_count = 3
        mock_get_dice_bag.return_value = (np.array([1, 1]), np.array([6, 5]), 5)
        troops_lost_attacker, troops_lost_defender, attacker_won, is_legal = (
            attack_territory(self.from_territory, self.to_territory, 2)
        )
        self.assertTrue(is_legal)
        self.assertEqual(troops_lost_attacker, 2)  # All attacker troops lost
        self.assertEqual(troops_lost_defender, 0)  # No defender troops lost
        self.assertFalse(attacker_won)


class TestPlaceTroops(unittest.TestCase):
    def setUp(self):
        self.player = Mock()
        self.player.total_troops = 10
        self.player.placeable_troops = 5

        self.territory_owned = Mock()
        self.territory_owned.owner = self.player
        self.territory_owned.troop_count = 2

        self.territory_not_owned = Mock()
        self.territory_not_owned.owner = None
        self.territory_not_owned.troop_count = 3

    @patch("atomic_actions.place_troops")
    def test_place_troops(self, mock_place_troops):
        result = place_troops(self.player, self.territory_owned, 3)
        self.assertTrue(result)
        self.assertEqual(self.player.total_troops, 13)
        self.assertEqual(self.territory_owned.troop_count, 5)
        self.assertEqual(self.player.placeable_troops, 2)

    @patch("atomic_actions.place_troops")
    def test_place_troops_not_enough_troops(self, mock_place_troops):
        result = place_troops(self.player, self.territory_owned, 6)
        self.assertFalse(result)
        self.assertEqual(self.player.total_troops, 10)
        self.assertEqual(self.territory_owned.troop_count, 2)
        self.assertEqual(self.player.placeable_troops, 5)

    @patch("atomic_actions.place_troops")
    def test_place_troops_not_owner(self, mock_place_troops):
        result = place_troops(self.player, self.territory_not_owned, 3)
        self.assertFalse(result)
        self.assertEqual(self.player.total_troops, 10)
        self.assertEqual(self.territory_not_owned.troop_count, 3)
        self.assertEqual(self.player.placeable_troops, 5)

    @patch("atomic_actions.place_troops")
    def test_place_troops_exact_amount(self, mock_place_troops):
        result = place_troops(self.player, self.territory_owned, 5)
        self.assertTrue(result)
        self.assertEqual(self.player.total_troops, 15)
        self.assertEqual(self.territory_owned.troop_count, 7)
        self.assertEqual(self.player.placeable_troops, 0)


class TestGenerateTroops(unittest.TestCase):
    # Test cases for a more aggressive version to encourage aggressive play
    def setUp(self):
        self.player = Mock()
        self.player.territories = [True, False, True, True, False, True]
        self.player.placeable_troops = 0

        self.territories = [Mock() for _ in range(len(self.player.territories))]
        self.territories[0].continent = "Continent1"
        self.territories[2].continent = "Continent2"
        self.territories[3].continent = "Continent1"
        self.territories[5].continent = "Continent2"

        self.continent1 = Mock()
        self.continent1.territories = [self.territories[0], self.territories[3]]
        self.continent1.bonus_troop_count = 5

        self.continent2 = Mock()
        self.continent2.territories = [self.territories[2], self.territories[5]]
        self.continent2.bonus_troop_count = 3

        for territory in self.territories:
            if territory.continent == "Continent1":
                territory.continent = self.continent1
            elif territory.continent == "Continent2":
                territory.continent = self.continent2

    @patch("atomic_actions.generate_troops")
    def test_generate_troops_with_continent_bonus(self, mock_generate_troops):
        generate_troops(self.player, self.territories)

        expected_troops = (
            3
            + sum(1 for item in [True, False, True, True, False, True] if item)
            + 5
            + 3
        )
        self.assertEqual(self.player.placeable_troops, expected_troops)

    @patch("atomic_actions.generate_troops")
    def test_generate_troops_with_partial_continent_bonus(self, mock_generate_troops):
        self.continent1.territories.append(Mock())

        generate_troops(self.player, self.territories)

        expected_troops = (
            3 + sum(1 for item in [True, False, True, True, False, True] if item) + 3
        )
        self.assertEqual(self.player.placeable_troops, expected_troops)

    @patch("atomic_actions.generate_troops")
    def test_generate_troops_without_continent_bonus(self, mock_generate_troops):
        self.continent1.territories.append(Mock())
        self.continent2.territories.extend([Mock(), Mock()])

        generate_troops(self.player, self.territories)

        expected_troops = 3 + sum(
            1 for item in [True, False, True, True, False, True] if item
        )
        self.assertEqual(self.player.placeable_troops, expected_troops)


class TestGetCard(unittest.TestCase):
    def setUp(self):
        self.hand = Mock()
        self.hand.count = 0
        self.hand.soldier = 0
        self.hand.cavalry = 0
        self.hand.artillery = 0
        self.hand.wild = 0

    @patch("random.randint")
    def test_get_card_soldier(self, mock_randint):
        mock_randint.return_value = 0  # Soldier range
        get_card(self.hand)
        self.assertEqual(self.hand.count, 1)
        self.assertEqual(self.hand.soldier, 1)
        self.assertEqual(self.hand.cavalry, 0)
        self.assertEqual(self.hand.artillery, 0)
        self.assertEqual(self.hand.wild, 0)

    @patch("random.randint")
    def test_get_card_cavalry(self, mock_randint):
        mock_randint.return_value = 14  # Cavalry range
        get_card(self.hand)
        self.assertEqual(self.hand.count, 1)
        self.assertEqual(self.hand.soldier, 0)
        self.assertEqual(self.hand.cavalry, 1)
        self.assertEqual(self.hand.artillery, 0)
        self.assertEqual(self.hand.wild, 0)

    @patch("random.randint")
    def test_get_card_artillery(self, mock_randint):
        mock_randint.return_value = 28  # Artillery range
        get_card(self.hand)
        self.assertEqual(self.hand.count, 1)
        self.assertEqual(self.hand.soldier, 0)
        self.assertEqual(self.hand.cavalry, 0)
        self.assertEqual(self.hand.artillery, 1)
        self.assertEqual(self.hand.wild, 0)

    @patch("random.randint")
    def test_get_card_wild(self, mock_randint):
        mock_randint.return_value = 43  # Wild card range
        get_card(self.hand)
        self.assertEqual(self.hand.count, 1)
        self.assertEqual(self.hand.soldier, 0)
        self.assertEqual(self.hand.cavalry, 0)
        self.assertEqual(self.hand.artillery, 0)
        self.assertEqual(self.hand.wild, 1)


class TestTakeCards(unittest.TestCase):
    def setUp(self):
        self.player1 = Mock()
        self.player2 = Mock()

        self.player1.hand = Mock()
        self.player1.hand.artillery = 1
        self.player1.hand.cavalry = 2
        self.player1.hand.soldier = 3
        self.player1.hand.wild = 4
        self.player1.hand.count = 5

        self.player2.hand = Mock()
        self.player2.hand.artillery = 6
        self.player2.hand.cavalry = 7
        self.player2.hand.soldier = 8
        self.player2.hand.wild = 9
        self.player2.hand.count = 10

    def test_take_cards(self):
        take_cards(self.player1, self.player2)

        self.assertEqual(self.player1.hand.artillery, 7)  # 1 + 6
        self.assertEqual(self.player1.hand.cavalry, 9)  # 2 + 7
        self.assertEqual(self.player1.hand.soldier, 11)  # 3 + 8
        self.assertEqual(self.player1.hand.wild, 13)  # 4 + 9
        self.assertEqual(self.player1.hand.count, 15)  # 5 + 10


if __name__ == "__main__":
    unittest.main()
