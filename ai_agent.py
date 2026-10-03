"""Hybrid adversarial agent for the mole, Zephyr.

Decision 1: Simple Reflex
    Once the restricted PIN is verified, activate the secondary security
    episode. No utility calculation is used.

Decision 2: Utility Based
    When Zephyr is questioned, score truth vs. lie from the current state
    and choose the higher-utility action. Ties are broken randomly.

    Utility = -100 * P(detective accuses Zephyr) - lying cost.
    P is built in odds form from published detection rates; see
    "Utility model" in README.md for the derivation and references.
"""

import random


# --- Utility model parameters ---------------------------------------------
# Odds are p / (1 - p). Each factor multiplies the odds of being accused.

# 1 mole among 5 suspects -> P = 0.20 -> odds 0.25.
PRIOR_ODDS = 0.25

# Odds multiplier for one incriminating fact that does not contradict his
# statement (admitting he was in Storage, or a clue that points at Supply).
# Design weight: no study measures it. Values 2-3 give the same 8 decisions.
INCRIMINATING_FACT = 2.0

# Demeanor leak of a lie with no evidence against it.
# Bond & DePaulo (2006): lies flagged 47%, truths flagged 39%.
# odds(0.47) / odds(0.39) = 1.39
LIE_LEAK = 1.39

# Multiplier per clue that contradicts the lie, by detective skill.
# Hartwig et al. (2006): liars caught 85.7% by interviewers trained in
# Strategic Use of Evidence, 55.0% by untrained ones; baseline 47%.
# Passing the security check = skilled detective; failing = unskilled.
CONTRADICTION_SKILLED = 6.76    # odds(0.857) / odds(0.47)
CONTRADICTION_UNSKILLED = 1.38  # odds(0.550) / odds(0.47)

# Fixed intrinsic cost of lying on the 0-100 scale. Small because a saboteur
# is among the ~20% who lie fully (Fischbacher & Foellmi-Heusi 2013), and
# fixed because lying barely responds to stake size (Abeler et al. 2019).
# 7.5 or more makes Zephyr tell the truth when no clue has been found.
LIE_COST = 5.0

# Value of escaping detection; utilities run from -VALUE (caught) to 0.
ESCAPE_VALUE = 100.0


def _odds_to_prob(odds):
    return odds / (1.0 + odds)


def accusation_probability(storage, cafe, security_failed, action):
    """P(the detective accuses Zephyr) after he gives `action`."""
    clues = int(storage) + int(cafe)
    if action == "truth":
        # Admitting Storage is one incriminating fact; each clue adds one.
        odds = PRIOR_ODDS * INCRIMINATING_FACT * INCRIMINATING_FACT ** clues
    else:
        # Both clues contradict "I was in the Cafeteria, never near Storage".
        k = CONTRADICTION_UNSKILLED if security_failed else CONTRADICTION_SKILLED
        odds = PRIOR_ODDS * LIE_LEAK * k ** clues
    return _odds_to_prob(odds)


def utility(storage, cafe, security_failed, action):
    """Expected utility of `action` ("truth" or "lie") in a given state."""
    p = accusation_probability(storage, cafe, security_failed, action)
    cost = LIE_COST if action == "lie" else 0.0
    return -ESCAPE_VALUE * p - cost


def utility_table():
    """All 8 states -> (truth utility, lie utility), rounded for display."""
    table = {}
    for storage in (False, True):
        for cafe in (False, True):
            for security_failed in (False, True):
                state = (storage, cafe, security_failed)
                table[state] = tuple(
                    round(utility(*state, action)) for action in ("truth", "lie")
                )
    return table


class MoleAI:

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.truth_count = 0
        self.lie_count = 0
        self.security_sabotage_count = 0
        self.security_skip_count = 0

    def _state_features(self, state):
        return (
            bool(getattr(state, "storage_evidence_found", False)),
            bool(getattr(state, "cafeteria_evidence_found", False)),
            bool(getattr(state, "security_failed", False)),
            bool(getattr(state, "contradiction_flagged", False)),
        )

    def _utility(self, state, action):
        """Utility for Zephyr's truth/lie decision.

        State: Storage found x Cafeteria found x Security failed (8 states).
        Computed from the parameters at the top of this module.
        """
        storage, cafe, security_failed, _contrad = self._state_features(state)
        return utility(storage, cafe, security_failed, action)

    def decide_security_sabotage(self, game_state):
        """Simple-reflex security mode decision.

        IF Storage evidence AND Cafeteria evidence -> Hard Mode (Wordle).
        ELSE -> Easy Mode (security completes without Wordle).
        No utilities are computed for this decision.
        """
        hard_mode = (
            bool(getattr(game_state, "storage_evidence_found", False))
            and bool(getattr(game_state, "cafeteria_evidence_found", False))
        )
        if hard_mode:
            self.security_sabotage_count += 1
        else:
            self.security_skip_count += 1
        return hard_mode

    def decide_truth_or_lie(self, game_state):
        """Utility-based choice between truth and lying."""
        candidates = [
            (self._utility(game_state, action), action)
            for action in ("truth", "lie")
        ]

        best_score = max(score for score, _ in candidates)
        best_actions = [
            action for score, action in candidates if score == best_score
        ]
        action = self.rng.choice(best_actions)

        if action == "truth":
            self.truth_count += 1
        else:
            self.lie_count += 1

        return action == "truth"

    def stats(self):
        return {
            "truth_count": self.truth_count,
            "lie_count": self.lie_count,
            "security_sabotage_count": self.security_sabotage_count,
            "security_skip_count": self.security_skip_count,
        }
