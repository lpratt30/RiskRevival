"""Risk mechanics; turn ordering is the responsibility of the environment/bots."""

import random
from collections import Counter, deque
from numbers import Integral

import numpy as np


def _positive_integer(value):
    return isinstance(value, Integral) and not isinstance(value, bool) and value > 0


def get_dice_bag(attacker_troops, defender_troops, verbose=False, rng=None):
    """Roll independent armies, sorting attacker triples and defender pairs.

    Combat requests a fresh bag per round so dice counts follow surviving armies.
    """
    size = attacker_troops + defender_troops
    rolls = np.random.randint(1, 7, size) if rng is None else rng.integers(1, 7, size)
    attacker = rolls[:attacker_troops].copy()
    defender = rolls[attacker_troops:].copy()
    for army, group_size in ((attacker, 3), (defender, 2)):
        for start in range(0, len(army), group_size):
            army[start : start + group_size] = sorted(
                army[start : start + group_size], reverse=True
            )
    if verbose:
        print(f"Attacker rolls: {attacker}; defender rolls: {defender}")
    return attacker, defender, size


def attack_territory(
    from_territory, to_territory, troops_to_attack_with, verbose=False
):
    """Commit an army until conquest or defeat, always leaving one behind.

    Returns (attacker losses, defender losses, conquered, legal). Illegal
    requests leave the board untouched. Oversized armies are capped for
    compatibility with the original API.
    """
    if (
        not _positive_integer(troops_to_attack_with)
        or to_territory.name not in from_territory.neighbor_names
        or from_territory.owner.key == to_territory.owner.key
        or from_territory.troop_count < 2
        or to_territory.troop_count < 1
    ):
        return 0, 0, False, False

    attacker = from_territory.owner
    defender = to_territory.owner
    committed = min(troops_to_attack_with, from_territory.troop_count - 1)
    remaining = committed
    defending = to_territory.troop_count
    rng = getattr(attacker, "dice_rng", None)
    if not isinstance(rng, np.random.Generator):
        rng = None
    while remaining and defending:
        attack_dice, defend_dice, _ = get_dice_bag(
            min(3, remaining), min(2, defending), verbose, rng=rng
        )
        for attack, defense in zip(attack_dice, defend_dice):
            if attack > defense:
                defending -= 1
            else:
                remaining -= 1
            if not remaining or not defending:
                break

    attacker_losses = committed - remaining
    defender_losses = to_territory.troop_count - defending
    conquered = defending == 0
    if conquered:
        from_territory.troop_count -= committed
        to_territory.troop_count = remaining
        defender.territories[to_territory.key] = 0
        defender.territory_count -= 1
        attacker.territories[to_territory.key] = 1
        attacker.territory_count += 1
        to_territory.owner = attacker
        to_territory.owner_color = from_territory.owner_color
    else:
        from_territory.troop_count -= attacker_losses
        to_territory.troop_count = defending

    attacker.total_troops -= attacker_losses
    defender.total_troops -= defender_losses
    attacker.damage_dealt[defender.key] += defender_losses
    defender.damage_received[attacker.key] += defender_losses
    return attacker_losses, defender_losses, conquered, True


def place_troops(player, territory, troops):
    if (
        not _positive_integer(troops)
        or territory.owner != player
        or troops > player.placeable_troops
    ):
        return False
    player.total_troops += troops
    territory.troop_count += troops
    player.placeable_troops -= troops
    return True


def generate_troops(player, territories):
    """Retain the aggressive rule: 3 + territories + continent bonuses."""
    owned = [territories[i] for i, flag in enumerate(player.territories) if flag]
    if not owned:
        return
    continents = Counter(territory.continent for territory in owned)
    bonus = sum(
        continent.bonus_troop_count
        for continent, count in continents.items()
        if count == len(continent.territories)
    )
    player.placeable_troops += 3 + len(owned) + bonus


def get_card(hand, rng=None):
    """Draw independently, retaining the original 14/14/14/2 distribution."""
    card = (rng or random).randint(0, 43)
    kind = (
        "soldier"
        if card < 14
        else "cavalry"
        if card < 28
        else "artillery"
        if card < 42
        else "wild"
    )
    setattr(hand, kind, getattr(hand, kind) + 1)
    hand.count += 1


def take_cards(player1, player2):
    if player1 is player2:
        raise ValueError("A player cannot take their own cards")
    for kind in ("soldier", "cavalry", "artillery", "wild", "count"):
        setattr(
            player1.hand,
            kind,
            getattr(player1.hand, kind) + getattr(player2.hand, kind),
        )
        setattr(player2.hand, kind, 0)


def _best_trade(hand):
    """Return the highest-value recipe, using as few wilds as possible."""
    if hand.count < 3:
        return 0, (0, 0, 0, 0)
    available = (hand.soldier, hand.cavalry, hand.artillery)
    for value, recipe in (
        (10, (1, 1, 1)),
        (8, (0, 0, 3)),
        (6, (0, 3, 0)),
        (4, (3, 0, 0)),
    ):
        used = tuple(min(have, need) for have, need in zip(available, recipe))
        wilds = 3 - sum(used)
        if wilds <= hand.wild:
            return value, used + (wilds,)
    return 0, (0, 0, 0, 0)


def check_cards(hand):
    value, used = _best_trade(hand)
    return value, bool(used[-1])


def trade_cards(player):
    """Trade one best set. An invalid hand is unchanged and returns zero."""
    value, used = _best_trade(player.hand)
    if value:
        for kind, count in zip(("soldier", "cavalry", "artillery", "wild"), used):
            setattr(player.hand, kind, getattr(player.hand, kind) - count)
        player.hand.count -= 3
        player.placeable_troops += value
    return value


def fortify_destinations(territory):
    """Find friendly territories reachable without crossing enemy territory."""
    visited = {territory}
    queue = deque([territory])
    reachable = []
    while queue:
        current = queue.popleft()
        for neighbor in current.neighbors:
            if neighbor.owner == territory.owner and neighbor not in visited:
                visited.add(neighbor)
                reachable.append(neighbor)
                queue.append(neighbor)
    return reachable


def fortify(from_territory, to_territory, troop_count):
    if not _positive_integer(troop_count) or troop_count >= from_territory.troop_count:
        raise ValueError(
            "Fortification must move positive whole troops and leave at least one"
        )
    if to_territory not in fortify_destinations(from_territory):
        raise ValueError("Fortification requires a path through friendly territories")
    from_territory.troop_count -= troop_count
    to_territory.troop_count += troop_count
