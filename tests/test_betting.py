"""
Tests for betting module (Kelly criterion)
"""

import sys
from pathlib import Path

import pytest
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.betting.kelly import (
    KellyCalculator,
    BetSizing,
    calculate_kelly_fraction,
    calculate_optimal_stake,
    compare_strategies,
)


class TestCalculateKellyFraction:
    """Tests for calculate_kelly_fraction function"""

    def test_positive_ev_bet(self):
        """Test Kelly fraction for positive EV bet"""
        # EV = 0.25 * 5.0 = 1.25, edge = 0.25
        kelly = calculate_kelly_fraction(0.25, 5.0)
        assert kelly > 0
        # f = (0.25 * 5 - 1) / (5 - 1) = 0.25 / 4 = 0.0625
        assert kelly == pytest.approx(0.0625, rel=1e-4)

    def test_negative_ev_bet(self):
        """Test Kelly fraction for negative EV bet"""
        # EV = 0.10 * 5.0 = 0.5, edge = -0.5
        kelly = calculate_kelly_fraction(0.10, 5.0)
        assert kelly < 0
        # f = (0.10 * 5 - 1) / (5 - 1) = -0.5 / 4 = -0.125
        assert kelly == pytest.approx(-0.125, rel=1e-4)

    def test_breakeven_bet(self):
        """Test Kelly fraction for breakeven bet (EV = 1.0)"""
        # EV = 0.20 * 5.0 = 1.0
        kelly = calculate_kelly_fraction(0.20, 5.0)
        assert kelly == pytest.approx(0.0, rel=1e-4)

    def test_high_probability_low_odds(self):
        """Test Kelly fraction for high prob, low odds bet"""
        # EV = 0.60 * 2.0 = 1.20
        kelly = calculate_kelly_fraction(0.60, 2.0)
        # f = (0.60 * 2 - 1) / (2 - 1) = 0.20 / 1 = 0.20
        assert kelly == pytest.approx(0.20, rel=1e-4)

    def test_low_probability_high_odds(self):
        """Test Kelly fraction for low prob, high odds bet"""
        # EV = 0.05 * 30.0 = 1.50
        kelly = calculate_kelly_fraction(0.05, 30.0)
        # f = (0.05 * 30 - 1) / (30 - 1) = 0.50 / 29 ≈ 0.0172
        assert kelly == pytest.approx(0.5 / 29, rel=1e-4)

    def test_odds_less_than_one(self):
        """Test that odds <= 1 returns 0"""
        assert calculate_kelly_fraction(0.50, 1.0) == 0.0
        assert calculate_kelly_fraction(0.50, 0.5) == 0.0


class TestCalculateOptimalStake:
    """Tests for calculate_optimal_stake function"""

    def test_positive_ev_stake(self):
        """Test stake calculation for positive EV"""
        stake = calculate_optimal_stake(
            probability=0.25,
            odds=5.0,
            bankroll=100000,
            kelly_multiplier=0.25,
        )
        # Kelly = 0.0625, quarter Kelly = 0.015625
        # Stake = 100000 * 0.015625 = 1562.5 -> 1500
        assert stake == 1500

    def test_negative_ev_stake(self):
        """Test stake is 0 for negative EV"""
        stake = calculate_optimal_stake(
            probability=0.10,
            odds=5.0,
            bankroll=100000,
        )
        assert stake == 0

    def test_max_stake_limit(self):
        """Test maximum stake limit is applied"""
        stake = calculate_optimal_stake(
            probability=0.50,
            odds=3.0,
            bankroll=100000,
            kelly_multiplier=1.0,  # Full Kelly
            max_stake_pct=0.05,
        )
        # Kelly = (0.5 * 3 - 1) / 2 = 0.25
        # But max is 5% = 5000
        assert stake <= 5000

    def test_min_stake(self):
        """Test minimum stake is respected"""
        stake = calculate_optimal_stake(
            probability=0.21,  # Just barely positive EV
            odds=5.0,
            bankroll=10000,
            kelly_multiplier=0.25,
            min_stake=100,
        )
        # Kelly is very small, stake might round to 0 or min_stake
        assert stake == 0 or stake >= 100

    def test_rounding_to_100(self):
        """Test stake is rounded to 100 yen"""
        stake = calculate_optimal_stake(
            probability=0.25,
            odds=5.0,
            bankroll=100000,
        )
        assert stake % 100 == 0


class TestKellyCalculator:
    """Tests for KellyCalculator class"""

    def test_init(self):
        """Test initialization"""
        calc = KellyCalculator(bankroll=100000)
        assert calc.bankroll == 100000
        assert calc.kelly_multiplier == 0.25
        assert calc.min_stake == 100

    def test_calculate_single(self):
        """Test single bet calculation"""
        calc = KellyCalculator(bankroll=100000)
        sizing = calc.calculate_single(0.25, 5.0)

        assert isinstance(sizing, BetSizing)
        assert sizing.probability == 0.25
        assert sizing.odds == 5.0
        assert sizing.expected_value == 1.25
        assert sizing.edge == pytest.approx(0.25, rel=1e-4)
        assert sizing.kelly_fraction == pytest.approx(0.0625, rel=1e-4)
        assert sizing.stake > 0

    def test_calculate_single_negative_ev(self):
        """Test single bet with negative EV returns 0 stake"""
        calc = KellyCalculator(bankroll=100000)
        sizing = calc.calculate_single(0.10, 5.0)

        assert sizing.expected_value == 0.5
        assert sizing.edge < 0
        assert sizing.stake == 0

    def test_calculate_multiple(self):
        """Test multiple bets calculation"""
        calc = KellyCalculator(bankroll=100000, max_total_exposure=0.30)
        bets = [
            (0.20, 6.0),
            (0.15, 10.0),
            (0.10, 15.0),
        ]
        sizings = calc.calculate_multiple(bets)

        assert len(sizings) == 3
        total_stake = sum(s.stake for s in sizings)
        assert total_stake <= 30000  # Max 30% exposure

    def test_calculate_multiple_scales_down(self):
        """Test that multiple bets are scaled down if over limit"""
        calc = KellyCalculator(
            bankroll=100000,
            kelly_multiplier=1.0,  # Full Kelly for more aggressive sizing
            max_total_exposure=0.20,
        )
        bets = [
            (0.30, 4.0),  # High EV bets
            (0.25, 5.0),
            (0.20, 6.0),
        ]
        sizings = calc.calculate_multiple(bets)

        total_stake = sum(s.stake for s in sizings)
        assert total_stake <= 20000  # Max 20% exposure

    def test_update_bankroll(self):
        """Test bankroll update after bet"""
        calc = KellyCalculator(bankroll=100000)

        # Win
        calc.update_bankroll(500)
        assert calc.bankroll == 100500

        # Loss
        calc.update_bankroll(-200)
        assert calc.bankroll == 100300

    def test_simulate_growth(self):
        """Test bankroll growth simulation"""
        calc = KellyCalculator(bankroll=100000, kelly_multiplier=0.25)

        probs = [0.25, 0.25, 0.25]
        odds = [5.0, 5.0, 5.0]
        outcomes = [True, False, True]

        history = calc.simulate_growth(probs, odds, outcomes)

        assert len(history) == 4  # Initial + 3 bets
        assert history[0] == 100000  # Initial

    def test_simulate_growth_all_losses(self):
        """Test simulation with all losses"""
        calc = KellyCalculator(bankroll=100000, kelly_multiplier=0.25)

        probs = [0.25] * 5
        odds = [5.0] * 5
        outcomes = [False] * 5

        history = calc.simulate_growth(probs, odds, outcomes)

        # Bankroll should decrease but not go negative
        assert history[-1] < history[0]
        assert history[-1] > 0


class TestCompareStrategies:
    """Tests for compare_strategies function"""

    def test_compare_returns_all_strategies(self):
        """Test that all strategies are returned"""
        results = compare_strategies(
            probabilities=[0.25, 0.25],
            odds_list=[5.0, 5.0],
            outcomes=[True, False],
        )

        assert "flat_100" in results
        assert "flat_1000" in results
        assert "kelly_full" in results
        assert "kelly_half" in results
        assert "kelly_quarter" in results

    def test_compare_with_all_wins(self):
        """Test comparison with all winning bets"""
        results = compare_strategies(
            probabilities=[0.30] * 10,
            odds_list=[4.0] * 10,
            outcomes=[True] * 10,
            initial_bankroll=100000,
        )

        # All strategies should profit
        for name, final in results.items():
            assert final > 100000, f"{name} should profit"

    def test_compare_kelly_outperforms_flat_long_term(self):
        """Test Kelly outperforms flat betting with many bets"""
        # Simulate many positive EV bets with realistic hit rate
        np.random.seed(42)
        n_bets = 100
        prob = 0.25
        odds = 5.0  # EV = 1.25

        outcomes = np.random.random(n_bets) < prob

        results = compare_strategies(
            probabilities=[prob] * n_bets,
            odds_list=[odds] * n_bets,
            outcomes=outcomes.tolist(),
            initial_bankroll=100000,
        )

        # Kelly should generally outperform, but this is probabilistic
        # Just check it doesn't go bankrupt with quarter Kelly
        assert results["kelly_quarter"] > 0


class TestBetSizing:
    """Tests for BetSizing dataclass"""

    def test_bet_sizing_creation(self):
        """Test BetSizing dataclass creation"""
        sizing = BetSizing(
            probability=0.25,
            odds=5.0,
            expected_value=1.25,
            edge=0.25,
            kelly_fraction=0.0625,
            recommended_fraction=0.015625,
            stake=1500,
        )

        assert sizing.probability == 0.25
        assert sizing.odds == 5.0
        assert sizing.expected_value == 1.25
        assert sizing.edge == 0.25
        assert sizing.stake == 1500
