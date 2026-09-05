"""Five-phase Gym environment for the project's simplified Risk rules."""

import random
from enum import IntEnum

import gym
import numpy as np
from gym.spaces import Box, Discrete

from atomic_actions import (
    attack_territory,
    fortify,
    generate_troops,
    get_card,
    place_troops,
    take_cards,
    trade_cards,
)
from board import (
    create_board,
    create_board_test,
    create_graph,
    display_graph,
    fortify_bfs,
)


class Phase(IntEnum):
    PLACEMENT = 0
    ATTACK_SOURCE = 1
    ATTACK_TARGET = 2
    FORTIFY_SOURCE = 3
    FORTIFY_TARGET = 4


class RiskEnvFlat(gym.Env):
    """One territory/skip action per step; the agent always occupies player zero."""

    metadata = {"render_modes": []}
    CARD_TROOP_VALUE = 3
    EARLY_GAME = 6
    LATE_GAME = 10

    def __init__(self, env_config=None):
        config = dict(env_config or {})
        self.num_players = config.get("num_players", 2)
        self.colors = config.get("colors")
        self.board_size = config.get("size", 0)
        self.bot_types = list(
            config.get("bot_types", ["Neutral"] * (self.num_players - 1))
        )
        self.neutral_only = all(bot == "Neutral" for bot in self.bot_types)
        self.shuffle_bots = config.get("shuffle_bots", True)
        self.skip_bots = config.get("skip_bots", False)
        self.phasic_credit_assignment = config.get("phasic_credit_assignment", True)
        self.invalid_move_penalty = -10
        self._random = random.Random()
        self.reset()
        count = len(self.territories)
        self.action_space = Discrete(count + 1)
        self.observation_space = Box(
            low=np.array([0] * 5 + [-1] * count + [0], dtype=np.float32),
            high=np.array([1] * 5 + [1] * count + [count], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._random.seed(seed)
            if hasattr(self, "action_space"):
                self.action_space.seed(seed)
        bot_types = self.bot_types.copy()
        if self.shuffle_bots:
            self._random.shuffle(bot_types)
        kwargs = dict(colors=self.colors, rng=self._random, dice_rng=self.np_random)
        if self.board_size == "classic":
            self.continents, self.territories, self.players = create_board(
                self.num_players, bot_types, **kwargs
            )
        else:
            self.continents, self.territories, self.players = create_board_test(
                self.num_players, bot_types, size=self.board_size, **kwargs
            )
        self.agent = self.players[0]
        self.turns_passed = 0
        self.phase = Phase.PLACEMENT
        self.from_terr = None
        self.prev_move = len(self.territories)
        self.agent_game_ended = False
        self.agent_troop_gain = 0
        self.agent_gets_card = False
        self.players_agent_survived = 0
        self.players_agent_eliminated = 0
        self.recurrence = False
        self._prepare_turn()
        return self.get_state(), {}

    def _prepare_turn(self):
        generate_troops(self.agent, self.territories)
        self.troops = self.agent.placeable_troops

    def get_state(self, verbose=False):
        previous = (
            self.prev_move
            if self.phase in (Phase.ATTACK_TARGET, Phase.FORTIFY_TARGET)
            else len(self.territories)
        )
        state = get_state(
            self.players,
            self.territories,
            self.phase,
            self.troops,
            self.turns_passed,
            previous,
        )
        if verbose:
            print(f"Phase: {self.phase}; troops: {state[5:-1]}; source: {previous}")
        return state

    def _record_reward(self, reward):
        self.agent.cumulative_reward += reward
        if reward > 0:
            self.agent.positive_reward_only += reward
        else:
            self.agent.negative_reward_only += reward
        return reward

    def get_reward(self, illegal=False):
        """Preserve the original shaping coefficients and phase timing."""
        first_place_bonus = 200
        placement_bonus = 0.5
        elimination_bonus = 0.25
        survival_bonus = 1
        delay_penalty = max(0, self.turns_passed - self.LATE_GAME) * survival_bonus
        if illegal:
            reward = self.invalid_move_penalty - delay_penalty
            if self.agent.territory_count == 0:
                reward -= first_place_bonus
            return self._record_reward(reward)

        # This intentionally remains the original two-player comparison.
        reward = (
            self.agent.territory_count - self.players[1].territory_count
        ) * survival_bonus
        reward -= delay_penalty
        elimination_reward = (
            elimination_bonus + placement_bonus
        ) * self.players_agent_eliminated
        if self.agent_game_ended:
            reward += elimination_reward
            if self.agent.territory_count > 0:
                reward += first_place_bonus
            else:
                reward -= first_place_bonus
                reward += placement_bonus * self.players_agent_survived
        else:
            if self.phase == Phase.PLACEMENT:
                if self.neutral_only:
                    reward -= survival_bonus
                elif self.turns_passed < self.EARLY_GAME:
                    reward += survival_bonus
                elif self.turns_passed > self.LATE_GAME:
                    reward -= survival_bonus
            reward += placement_bonus * self.players_agent_survived + elimination_reward
            if self.agent_troop_gain > 0 and (
                self.turns_passed < self.EARLY_GAME or self.players_agent_eliminated > 0
            ):
                reward += survival_bonus

        self.players_agent_survived = 0
        self.players_agent_eliminated = 0
        self.agent_troop_gain = 0
        return self._record_reward(reward)

    def _result(self, reward=0, illegal=False):
        return (
            self.get_state(),
            float(reward),
            bool(self.agent_game_ended),
            False,
            {"illegal_action": illegal},
        )

    def _illegal(self):
        return self._result(self.get_reward(illegal=True), illegal=True)

    def _clear_source(self):
        self.from_terr = None
        self.prev_move = len(self.territories)

    def _phase_reward(self):
        return self.get_reward() if self.phasic_credit_assignment else 0

    def step(self, action, verbose=False):
        if self.agent_game_ended:
            raise RuntimeError("Episode has ended; call reset() before step()")
        if not self.action_space.contains(action):
            return self._illegal()
        handlers = {
            Phase.PLACEMENT: self._place,
            Phase.ATTACK_SOURCE: self._select_attack_source,
            Phase.ATTACK_TARGET: self._attack,
            Phase.FORTIFY_SOURCE: self._select_fortify_source,
            Phase.FORTIFY_TARGET: self._fortify,
        }
        return handlers[self.phase](int(action))

    def _place(self, action):
        if (
            action == len(self.territories)
            or self.territories[action].owner is not self.agent
        ):
            return self._illegal()
        # Validate the destination before consuming cards or reinforcements.
        while self.agent.hand.count >= 5:
            if not trade_cards(self.agent):
                raise ValueError("Five-card hand contains no valid trade")
        if not place_troops(
            self.agent, self.territories[action], self.agent.placeable_troops
        ):
            raise RuntimeError("Placement phase has no reinforcements")
        self.troops = 0
        self.phase = Phase.ATTACK_SOURCE
        return self._result(self.get_reward())

    def _select_attack_source(self, action):
        if action == len(self.territories):
            self.phase = Phase.FORTIFY_SOURCE
            if self.agent_gets_card:
                get_card(self.agent.hand, self._random)
                self.agent_gets_card = False
            return self._result(self._phase_reward())
        source = self.territories[action]
        if (
            source.owner is not self.agent
            or source.troop_count < 2
            or not any(t.owner is not self.agent for t in source.neighbors)
        ):
            return self._illegal()
        self.from_terr = source
        self.prev_move = action
        self.phase = Phase.ATTACK_TARGET
        return self._result(self._phase_reward())

    def _attack(self, action):
        if action == len(self.territories):
            return self._illegal()
        target = self.territories[action]
        if target not in self.from_terr.neighbors or target.owner is self.agent:
            return self._illegal()
        defender = target.owner
        lost, _, won, legal = attack_territory(
            self.from_terr, target, self.from_terr.troop_count - 1
        )
        if not legal:
            return self._illegal()
        self.agent_troop_gain -= lost
        self._clear_source()
        self.phase = Phase.ATTACK_SOURCE
        if won and not self.agent_gets_card:
            self.agent_gets_card = True
            self.agent_troop_gain += self.CARD_TROOP_VALUE
        if won and defender.territory_count == 0:
            self.agent_troop_gain += self.CARD_TROOP_VALUE * defender.hand.count
            self.players_agent_eliminated += 1
            take_cards(self.agent, defender)
            self.agent_game_ended = self.agent.territory_count == len(self.territories)
            if self.agent_game_ended:
                self.phase = Phase.PLACEMENT
                return self._result(self.get_reward())
            if self.agent.hand.count >= 5:
                self.phase = Phase.PLACEMENT
                self.recurrence = True
                # Trade during placement; never generate another turn's troops.
        return self._result(self._phase_reward())

    def _select_fortify_source(self, action):
        if action == len(self.territories):
            return self.handle_other_players()
        source = self.territories[action]
        if (
            source.owner is not self.agent
            or source.troop_count < 2
            or not fortify_bfs(source)
        ):
            return self._illegal()
        self.from_terr = source
        self.prev_move = action
        self.phase = Phase.FORTIFY_TARGET
        return self._result()

    def _fortify(self, action):
        if action == len(self.territories):
            return self._illegal()
        target = self.territories[action]
        if target not in fortify_bfs(self.from_terr):
            return self._illegal()
        fortify(self.from_terr, target, self.from_terr.troop_count - 1)
        return self.handle_other_players()

    def handle_other_players(self):
        self.turns_passed += 1
        self.phase = Phase.PLACEMENT
        self.recurrence = False
        self._clear_source()
        starting = sum(p.territory_count > 0 for p in self.players[1:])
        if not self.skip_bots:
            for player in self.players[1:]:
                if player.is_bot and player.territory_count > 0:
                    if not player.make_move(self.players, self.territories):
                        raise RuntimeError("Bot made an illegal move")
                    if self.agent.territory_count == 0:
                        self.agent_game_ended = True
                        break
            ending = sum(p.territory_count > 0 for p in self.players[1:])
            self.players_agent_survived += starting - ending
        if not self.agent_game_ended:
            self._prepare_turn()
        # Terminal rewards must also be delivered with non-phasic credit.
        reward = self.get_reward() if self.agent_game_ended else self._phase_reward()
        return self._result(reward)

    def show_board(self, blocking=True):
        display_graph(
            create_graph(self.territories),
            self.territories,
            title="Current board state",
            save=True,
            blocking_display=blocking,
        )


def get_state(players, board, agents_phase, troops, turns_passed, previous_move):
    """Keep the checkpoint-compatible layout: 5 phases, signed troops, source."""
    phases = [0] * 5
    phases[agents_phase] = 1
    signed = [t.troop_count if t.owner is players[0] else -t.troop_count for t in board]
    scale = max(1, max(abs(count) for count in signed))
    return np.asarray(
        phases + [count / scale for count in signed] + [previous_move], dtype=np.float32
    )


def __getattr__(name):
    # Existing notebooks can continue using "from env import DQN, DQNAgent".
    if name in ("DQN", "DQNAgent"):
        from dqn import DQN, DQNAgent

        return {"DQN": DQN, "DQNAgent": DQNAgent}[name]
    raise AttributeError(name)
