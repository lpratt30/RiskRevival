"""Regression coverage for state corruption and turn-progression failures."""

import itertools
import random
import unittest
from unittest.mock import patch

import numpy as np

from actors import Player
from atomic_actions import (
    attack_territory,
    check_cards,
    fortify,
    generate_troops,
    place_troops,
    take_cards,
    trade_cards,
)
from board import (
    Continent,
    Territory,
    create_board,
    create_board_test,
    create_graph,
    fortify_bfs,
)
from env import Phase, RiskEnvFlat


def assign(territories, players, owners, troops):
    for player in players:
        player.territories = [0] * len(territories)
        player.territory_count = 0
        player.total_troops = 0
        player.placeable_troops = 0
    for index, (territory, owner, count) in enumerate(zip(territories, owners, troops)):
        player = players[owner]
        territory.key = index
        territory.owner = player
        territory.owner_color = player.name
        territory.troop_count = count
        player.territories[index] = 1
        player.territory_count += 1
        player.total_troops += count


def assert_consistent(case, territories, players):
    case.assertTrue(all(t.troop_count >= 1 for t in territories))
    for player in players:
        owned = [t for t in territories if t.owner is player]
        case.assertEqual(player.territory_count, len(owned))
        case.assertEqual(player.total_troops, sum(t.troop_count for t in owned))
        case.assertEqual(
            player.territories, [int(t.owner is player) for t in territories]
        )
        case.assertEqual(
            player.hand.count,
            sum(
                getattr(player.hand, kind)
                for kind in ("soldier", "cavalry", "artillery", "wild")
            ),
        )
        case.assertGreaterEqual(player.placeable_troops, 0)


def legal_actions(env):
    skip = len(env.territories)
    if env.phase == Phase.PLACEMENT:
        return [t.key for t in env.territories if t.owner is env.agent]
    if env.phase == Phase.ATTACK_SOURCE:
        return [skip] + [
            t.key
            for t in env.territories
            if t.owner is env.agent
            and t.troop_count > 1
            and any(n.owner is not env.agent for n in t.neighbors)
        ]
    if env.phase == Phase.ATTACK_TARGET:
        return [t.key for t in env.from_terr.neighbors if t.owner is not env.agent]
    if env.phase == Phase.FORTIFY_SOURCE:
        return [skip] + [
            t.key
            for t in env.territories
            if t.owner is env.agent and t.troop_count > 1 and fortify_bfs(t)
        ]
    return [t.key for t in fortify_bfs(env.from_terr)]


class MechanicsRegressions(unittest.TestCase):
    def setUp(self):
        continent = Continent("test", 2)
        self.territories = [Territory(str(i), continent) for i in range(3)]
        for territory in self.territories:
            continent.add_territory(territory)
        for left, right in zip(self.territories, self.territories[1:]):
            left.add_neighbor(right)
            right.add_neighbor(left)
        self.players = [Player("red", 0, 3), Player("blue", 1, 3)]
        assign(self.territories, self.players, [0, 1, 0], [10, 2, 1])

    def test_partial_army_conquest_leaves_uncommitted_troops(self):
        with patch(
            "atomic_actions.np.random.randint",
            side_effect=lambda a, b, n: np.array([6, 6, 6, 1, 1])[:n],
        ):
            result = attack_territory(*self.territories[:2], 3)
        self.assertEqual(result, (0, 2, True, True))
        self.assertEqual(self.territories[0].troop_count, 7)
        self.assertEqual(self.territories[1].troop_count, 3)
        assert_consistent(self, self.territories, self.players)

    def test_partial_army_defeat_does_not_destroy_reserves(self):
        with patch(
            "atomic_actions.get_dice_bag",
            return_value=(np.array([1, 1]), np.array([6, 6]), 4),
        ):
            result = attack_territory(*self.territories[:2], 2)
        self.assertEqual(result, (2, 0, False, True))
        self.assertEqual(self.territories[0].troop_count, 8)
        assert_consistent(self, self.territories, self.players)

    def test_dice_counts_follow_survivors_and_ties_favor_defender(self):
        with patch(
            "atomic_actions.get_dice_bag",
            side_effect=[
                (np.array([4, 4, 4]), np.array([4, 4]), 5),
                (np.array([1]), np.array([2, 2]), 3),
            ],
        ) as roll:
            result = attack_territory(*self.territories[:2], 3)
        self.assertEqual(result, (3, 0, False, True))
        self.assertEqual(
            [call.args[:2] for call in roll.call_args_list], [(3, 2), (1, 2)]
        )

    def test_invalid_armies_and_placements_do_not_mutate(self):
        self.players[0].placeable_troops = 5
        for amount in (-1, 0, 1.5, True):
            self.assertFalse(attack_territory(*self.territories[:2], amount)[-1])
            self.assertFalse(place_troops(self.players[0], self.territories[0], amount))
        self.assertEqual(self.territories[0].troop_count, 10)
        self.assertEqual(self.players[0].placeable_troops, 5)

    def test_fortification_requires_friendly_path(self):
        with self.assertRaises(ValueError):
            fortify(self.territories[0], self.territories[2], 2)
        assign(self.territories, self.players, [0, 0, 0], [10, 2, 1])
        fortify(self.territories[0], self.territories[2], 9)
        self.assertEqual([t.troop_count for t in self.territories], [1, 2, 10])
        for count in (-1, 0, 2, 0.5):
            with self.assertRaises(ValueError):
                fortify(self.territories[0], self.territories[2], count)
        assert_consistent(self, self.territories, self.players)

    def test_eliminated_players_get_no_reinforcements(self):
        assign(self.territories, self.players, [0, 0, 0], [10, 2, 1])
        generate_troops(self.players[1], self.territories)
        self.assertEqual(self.players[1].placeable_troops, 0)


class CardRegressions(unittest.TestCase):
    def hand(self, counts):
        player = Player("red", 0, 1)
        for kind, count in zip(("soldier", "cavalry", "artillery", "wild"), counts):
            setattr(player.hand, kind, count)
        player.hand.count = sum(counts)
        return player

    def test_invalid_trade_is_noop(self):
        player = self.hand((2, 2, 0, 0))
        before = vars(player.hand).copy()
        self.assertEqual(trade_cards(player), 0)
        self.assertEqual(vars(player.hand), before)

    def test_natural_mixed_set_preserves_wild(self):
        player = self.hand((1, 1, 1, 1))
        self.assertEqual(check_cards(player.hand), (10, False))
        self.assertEqual(trade_cards(player), 10)
        self.assertEqual(player.hand.wild, 1)
        self.assertEqual(player.hand.count, 1)

    def test_wild_substitutions(self):
        for counts, value in (
            ((0, 0, 2, 1), 8),
            ((0, 0, 1, 2), 10),
            ((0, 0, 0, 3), 10),
        ):
            player = self.hand(counts)
            self.assertEqual(trade_cards(player), value)
            self.assertEqual(player.hand.count, 0)

    def test_all_five_card_hands_trade_without_corruption(self):
        for counts in itertools.product(range(6), repeat=4):
            if sum(counts) != 5:
                continue
            player = self.hand(counts)
            self.assertGreater(trade_cards(player), 0, counts)
            self.assertEqual(player.hand.count, 2)
            remaining = [
                getattr(player.hand, kind)
                for kind in ("soldier", "cavalry", "artillery", "wild")
            ]
            self.assertEqual(sum(remaining), 2)
            self.assertGreaterEqual(min(remaining), 0)

    def test_transfer_clears_defender_and_cannot_duplicate_cards(self):
        player, defender = self.hand((0, 0, 0, 0)), self.hand((1, 2, 3, 1))
        take_cards(player, defender)
        take_cards(player, defender)
        self.assertEqual(player.hand.count, 7)
        self.assertEqual(sum(vars(defender.hand).values()), 0)


class BoardRegressions(unittest.TestCase):
    def test_initial_troop_budgets_on_all_boards(self):
        for size, count in ((0, 4), (1, 9), (2, 13), ("classic", 42)):
            for players in range(2, min(6, count) + 1):
                for seed in range(5):
                    factory = create_board if size == "classic" else create_board_test
                    kwargs = {} if size == "classic" else {"size": size}
                    _, territories, owners = factory(
                        players,
                        ["Neutral"] * (players - 1),
                        rng=random.Random(seed),
                        **kwargs,
                    )
                    self.assertEqual(len(territories), count)
                    assert_consistent(self, territories, owners)
                    self.assertEqual(
                        [p.total_troops for p in owners], [50 - 5 * players] * players
                    )

    def test_invalid_configurations_fail_clearly(self):
        for args in ((5, ["Neutral"] * 4), (2, []), (2, ["TFT"])):
            with self.assertRaises(ValueError):
                create_board_test(*args)
        with self.assertRaises(ValueError):
            create_board_test(2, ["Neutral"], colors=["red", "red"])
        with self.assertRaises(ValueError):
            create_board_test(2, ["Neutral"], size=7)

    def test_display_uses_territory_objects(self):
        _, territories, _ = create_board_test(2, ["Neutral"])
        with patch("matplotlib.pyplot.show"):
            graph = create_graph(territories, display=True)
        self.assertEqual(len(graph), 4)
        import matplotlib.pyplot as plt

        plt.close("all")


class EnvironmentRegressions(unittest.TestCase):
    def test_reinforcements_once_per_turn(self):
        env = RiskEnvFlat({"skip_bots": True})
        env.reset(seed=7)
        reserve = env.agent.placeable_troops
        starting = env.agent.total_troops
        env.step(legal_actions(env)[0])
        self.assertEqual(env.agent.total_troops, starting + reserve)
        env.step(len(env.territories))
        env.step(len(env.territories))
        self.assertEqual(env.agent.placeable_troops, reserve)
        env.step(legal_actions(env)[0])
        self.assertEqual(env.agent.total_troops, starting + 2 * reserve)

    def test_illegal_placement_preserves_cards_reserves_and_observation(self):
        env = RiskEnvFlat()
        env.agent.hand.soldier = env.agent.hand.count = 5
        before = env.get_state().copy()
        reserve = env.agent.placeable_troops
        enemy = next(t.key for t in env.territories if t.owner is not env.agent)
        for action in (enemy, -1, 100, 0.5, len(env.territories)):
            observation, reward, _, _, info = env.step(action)
            np.testing.assert_array_equal(observation, before)
            self.assertTrue(info["illegal_action"])
            self.assertEqual(reward, -10)
            self.assertEqual(env.agent.hand.count, 5)
            self.assertEqual(env.agent.placeable_troops, reserve)
        self.assertEqual(env.agent.cumulative_reward, -50)

    def test_seeded_environments_remain_independent_during_combat(self):
        config = {"size": 1, "num_players": 3, "bot_types": ["Random", "Neutral"]}
        first, second = RiskEnvFlat(config), RiskEnvFlat(config)
        first.reset(seed=42)
        second.reset(seed=42)
        chooser = random.Random(2)
        for _ in range(150):
            action = chooser.choice(legal_actions(first))
            a, b = first.step(action), second.step(action)
            np.testing.assert_array_equal(a[0], b[0])
            self.assertEqual(a[1:], b[1:])
            if a[2]:
                break
        self.assertEqual(config["bot_types"], ["Random", "Neutral"])

    def test_invalid_target_preserves_selection(self):
        env = RiskEnvFlat()
        env.step(legal_actions(env)[0])
        source = next(
            action for action in legal_actions(env) if action != len(env.territories)
        )
        env.step(source)
        before = env.get_state().copy()
        observation, _, _, _, info = env.step(source)
        self.assertTrue(info["illegal_action"])
        np.testing.assert_array_equal(observation, before)
        self.assertEqual(env.phase, Phase.ATTACK_TARGET)

    def test_elimination_trade_does_not_generate_another_turn(self):
        env = RiskEnvFlat({"num_players": 3, "bot_types": ["Neutral", "Neutral"]})
        assign(env.territories, env.players, [0, 1, 2, 2], [10, 1, 2, 2])
        defender = env.players[1]
        defender.hand.count = defender.hand.soldier = 5
        env.phase = Phase.ATTACK_SOURCE
        env.step(0)
        with patch(
            "atomic_actions.get_dice_bag",
            return_value=(np.array([6, 6, 6]), np.array([1]), 4),
        ):
            env.step(1)
        self.assertEqual(env.phase, Phase.PLACEMENT)
        total = env.agent.total_troops
        enemy = 2
        env.step(enemy)
        self.assertEqual(env.agent.hand.count, 5)
        env.step(1)
        self.assertEqual(env.agent.total_troops, total + 4)
        self.assertEqual(env.agent.hand.count, 2)
        self.assertEqual(defender.hand.count, 0)

    def test_terminal_reward_not_repeated(self):
        env = RiskEnvFlat()
        assign(env.territories, env.players, [0, 1, 0, 0], [10, 1, 1, 1])
        env.phase = Phase.ATTACK_SOURCE
        env.step(0)
        with patch(
            "atomic_actions.get_dice_bag",
            return_value=(np.array([6, 6, 6]), np.array([1]), 4),
        ):
            _, reward, terminated, _, _ = env.step(1)
        self.assertTrue(terminated)
        self.assertGreater(reward, 200)
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_bot_elimination_returns_terminal_reward_without_phasic_credit(self):
        env = RiskEnvFlat({"bot_types": ["Random"], "phasic_credit_assignment": False})
        assign(env.territories, env.players, [0, 1, 1, 1], [1, 100, 1, 1])
        env.phase = Phase.FORTIFY_SOURCE
        with (
            patch.object(env._random, "randint", return_value=1),
            patch.object(env._random, "choice", return_value=env.territories[1]),
            patch(
                "atomic_actions.get_dice_bag",
                return_value=(np.array([6, 6, 6]), np.array([1]), 4),
            ),
        ):
            _, reward, terminated, _, _ = env.step(len(env.territories))
        self.assertTrue(terminated)
        self.assertLessEqual(reward, -200)

    def test_random_games_preserve_invariants_and_reward_totals(self):
        chooser = random.Random(18)
        for size, players in ((0, 2), (1, 3), (2, 6), ("classic", 6)):
            env = RiskEnvFlat(
                {
                    "size": size,
                    "num_players": players,
                    "bot_types": ["Random"] * (players - 1),
                }
            )
            env.reset(seed=19)
            total = 0
            for _ in range(250):
                actions = legal_actions(env)
                self.assertTrue(actions)
                state, reward, terminated, _, info = env.step(chooser.choice(actions))
                self.assertFalse(info["illegal_action"])
                self.assertTrue(env.observation_space.contains(state))
                total += reward
                assert_consistent(self, env.territories, env.players)
                self.assertAlmostEqual(total, env.agent.cumulative_reward)
                if terminated:
                    break


if __name__ == "__main__":
    unittest.main()
