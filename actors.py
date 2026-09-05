"""Players and the original neutral/randomized heuristic opponents."""

import random

from atomic_actions import (
    attack_territory,
    fortify,
    generate_troops,
    get_card,
    place_troops,
    take_cards,
    trade_cards,
)


class Hand:
    def __init__(self):
        self.artillery = 0
        self.cavalry = 0
        self.soldier = 0
        self.wild = 0
        self.count = 0


class Player:
    def __init__(self, color, turn_order, num_territories):
        self.name = color
        self.turn_order = turn_order
        self.key = turn_order
        self.total_troops = 0
        self.placeable_troops = 0
        self.territories = [0] * num_territories
        self.territory_count = 0
        self.hand = Hand()
        self.is_bot = False
        self.damage_received = [0] * 6
        self.damage_dealt = [0] * 6
        self.cumulative_reward = 0
        self.positive_reward_only = 0
        self.negative_reward_only = 0
        self.rng = random
        self.dice_rng = None


class Neutral_Bot(Player):
    """Reinforce a randomly selected territory, then end the turn."""

    def __init__(self, color, turn_order, num_territories):
        super().__init__(color, turn_order, num_territories)
        self.is_bot = True

    def _place(self, territories, destination=None, verbose=False):
        while self.hand.count >= 5:
            if not trade_cards(self):
                raise ValueError("Five-card hand contains no valid trade")
        if destination is None:
            destination = self.rng.choice([t for t in territories if t.owner is self])
        if verbose:
            print(f"{self.name} places {self.placeable_troops} in {destination.name}")
        if not place_troops(self, destination, self.placeable_troops):
            raise RuntimeError("Bot attempted an illegal placement")
        return destination

    def make_move(self, players, territories, verbose=False):
        if not self.territory_count:
            raise ValueError("An eliminated bot cannot move")
        generate_troops(self, territories)
        self._place(territories, verbose=verbose)
        return True


class Random_Bot(Neutral_Bot):
    """Reinforce randomly, attack weak neighbors up to three times, fortify."""

    def __init__(self, color, turn_order, num_territories, num_players=None):
        super().__init__(color, turn_order, num_territories)

    def _attack(self, source, verbose=False):
        enemies = [t for t in source.neighbors if t.owner is not self]
        if not enemies or source.troop_count < 4:
            return None
        target = min(enemies, key=lambda territory: territory.troop_count)
        if target.troop_count > source.troop_count - 2:
            return None
        defender = target.owner
        _, _, won, legal = attack_territory(source, target, source.troop_count - 1)
        if not legal:
            raise RuntimeError("Bot attempted an illegal attack")
        if verbose:
            print(f"{self.name} attacked {target.name} and {'won' if won else 'lost'}")
        if won:
            if not defender.territory_count:
                take_cards(self, defender)
            return target
        return None

    def make_move(self, players, territories, verbose=False):
        if not self.territory_count:
            raise ValueError("An eliminated bot cannot move")
        attack_limit = self.rng.randint(0, 3)
        generate_troops(self, territories)
        source = self._place(territories, verbose=verbose)
        attacks = 0
        won_card = False
        while attacks < attack_limit:
            source = self._attack(source, verbose)
            if source is None:
                break
            attacks += 1
            won_card = True
            if self.hand.count >= 5:
                self._place(territories, source, verbose)
                attacks = 0
        if source is not None and source.troop_count > 1:
            friendly = [t for t in source.neighbors if t.owner is self]
            if friendly:
                target = max(friendly, key=lambda territory: territory.troop_count)
                fortify(source, target, source.troop_count - 1)
        if won_card:
            get_card(self.hand, self.rng)
        return True
