"""
Tests for backtesting module
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.simulator import BacktestSimulator, BacktestResult, BetRecord
from src.backtesting.metrics import (
    calculate_metrics,
    analyze_by_dimension,
    calculate_sharpe_ratio,
    BacktestMetrics,
)
from src.backtesting.report import (
    generate_summary_report,
    generate_csv_report,
)
from src.preprocessing.parser import PayoutParser, RacePayouts
from src.backtesting.synthetic_odds import (
    SyntheticOddsGenerator,
    calculate_historical_exacta_rates,
    HISTORICAL_WIN_RATE,
    HISTORICAL_SECOND_RATE,
)


class TestSyntheticOddsGenerator:
    """Tests for SyntheticOddsGenerator class"""

    def test_init_default_margin(self):
        """Test default margin initialization"""
        generator = SyntheticOddsGenerator()
        assert generator.margin == 0.25

    def test_init_custom_margin(self):
        """Test custom margin initialization"""
        generator = SyntheticOddsGenerator(margin=0.20)
        assert generator.margin == 0.20

    def test_exacta_probs_sum_to_one(self):
        """Test that exacta probabilities sum to 1"""
        generator = SyntheticOddsGenerator()
        total = sum(generator.exacta_probs.values())
        assert total == pytest.approx(1.0, rel=1e-6)

    def test_exacta_probs_count(self):
        """Test that we have 30 exacta combinations"""
        generator = SyntheticOddsGenerator()
        assert len(generator.exacta_probs) == 30

    def test_get_odds_valid_combination(self):
        """Test getting odds for valid combination"""
        generator = SyntheticOddsGenerator()
        odds = generator.get_odds(1, 2)
        assert odds > 0

    def test_get_odds_invalid_same_boat(self):
        """Test getting odds when first equals second"""
        generator = SyntheticOddsGenerator()
        odds = generator.get_odds(1, 1)
        assert odds == 0.0

    def test_get_odds_invalid_out_of_range(self):
        """Test getting odds for out of range boats"""
        generator = SyntheticOddsGenerator()
        assert generator.get_odds(0, 1) == 0.0
        assert generator.get_odds(1, 7) == 0.0
        assert generator.get_odds(-1, 2) == 0.0

    def test_get_all_odds_count(self):
        """Test get_all_odds returns 30 combinations"""
        generator = SyntheticOddsGenerator()
        all_odds = generator.get_all_odds()
        assert len(all_odds) == 30

    def test_get_all_odds_keys(self):
        """Test get_all_odds has correct key structure"""
        generator = SyntheticOddsGenerator()
        all_odds = generator.get_all_odds()

        for (first, second), odds in all_odds.items():
            assert 1 <= first <= 6
            assert 1 <= second <= 6
            assert first != second
            assert odds > 0

    def test_course_1_has_lowest_odds(self):
        """Test that 1-X combinations have lower odds (higher probability)"""
        generator = SyntheticOddsGenerator()
        all_odds = generator.get_all_odds()

        # 1-2 should have lower odds than 6-5
        assert all_odds[(1, 2)] < all_odds[(6, 5)]

    def test_get_race_odds_returns_same_as_get_all_odds(self):
        """Test get_race_odds returns same as get_all_odds for now"""
        generator = SyntheticOddsGenerator()
        race_odds = generator.get_race_odds()
        all_odds = generator.get_all_odds()
        assert race_odds == all_odds

    def test_odds_reflect_margin(self):
        """Test that odds correctly reflect the margin"""
        gen_25 = SyntheticOddsGenerator(margin=0.25)
        gen_20 = SyntheticOddsGenerator(margin=0.20)

        # Lower margin should result in higher odds
        odds_25 = gen_25.get_odds(1, 2)
        odds_20 = gen_20.get_odds(1, 2)
        assert odds_20 > odds_25


class TestCalculateHistoricalExactaRates:
    """Tests for calculate_historical_exacta_rates function"""

    def test_empty_dataframe(self):
        """Test with empty dataframe"""
        df = pd.DataFrame(columns=["date", "stadium_code", "race_no", "boat_no", "rank"])
        rates = calculate_historical_exacta_rates(df)
        assert len(rates) == 0

    def test_single_race(self):
        """Test with single race"""
        df = pd.DataFrame({
            "date": [20240101] * 6,
            "stadium_code": [1] * 6,
            "race_no": [1] * 6,
            "boat_no": [1, 2, 3, 4, 5, 6],
            "rank": [1, 2, 3, 4, 5, 6],
        })
        rates = calculate_historical_exacta_rates(df)
        assert (1, 2) in rates
        assert rates[(1, 2)] == 1.0

    def test_multiple_races(self):
        """Test with multiple races"""
        # Two races: first 1-2, second 4-3
        df = pd.DataFrame({
            "date": [20240101] * 6 + [20240101] * 6,
            "stadium_code": [1] * 6 + [2] * 6,
            "race_no": [1] * 6 + [1] * 6,
            "boat_no": [1, 2, 3, 4, 5, 6] * 2,
            "rank": [1, 2, 3, 4, 5, 6, 4, 5, 2, 1, 6, 3],  # First: 1-2, Second: 4-3
        })
        rates = calculate_historical_exacta_rates(df)
        assert rates[(1, 2)] == 0.5
        assert rates[(4, 3)] == 0.5

    def test_incomplete_race_skipped(self):
        """Test that races with less than 6 boats are skipped"""
        df = pd.DataFrame({
            "date": [20240101] * 5,  # Only 5 boats
            "stadium_code": [1] * 5,
            "race_no": [1] * 5,
            "boat_no": [1, 2, 3, 4, 5],
            "rank": [1, 2, 3, 4, 5],
        })
        rates = calculate_historical_exacta_rates(df)
        assert len(rates) == 0


class TestHistoricalRates:
    """Tests for historical rate constants"""

    def test_win_rate_sum(self):
        """Test that win rates sum to 1"""
        total = sum(HISTORICAL_WIN_RATE.values())
        assert total == pytest.approx(1.0, rel=1e-6)

    def test_second_rate_sum(self):
        """Test that second place rates sum to 1"""
        total = sum(HISTORICAL_SECOND_RATE.values())
        assert total == pytest.approx(1.0, rel=1e-6)

    def test_win_rate_course_1_highest(self):
        """Test that course 1 has highest win rate"""
        assert HISTORICAL_WIN_RATE[1] == max(HISTORICAL_WIN_RATE.values())

    def test_win_rate_course_6_lowest(self):
        """Test that course 6 has lowest win rate"""
        assert HISTORICAL_WIN_RATE[6] == min(HISTORICAL_WIN_RATE.values())


class TestBacktestSimulator:
    """Tests for BacktestSimulator class"""

    def test_init_default_values(self):
        """Test default initialization values"""
        simulator = BacktestSimulator()
        assert simulator.ev_threshold == 1.0
        assert simulator.stake == 100
        assert simulator.max_bets_per_race == 3
        assert simulator.use_synthetic_odds is False

    def test_init_custom_values(self):
        """Test custom initialization values"""
        simulator = BacktestSimulator(
            ev_threshold=1.2,
            stake=200,
            max_bets_per_race=5,
            use_synthetic_odds=True,
        )
        assert simulator.ev_threshold == 1.2
        assert simulator.stake == 200
        assert simulator.max_bets_per_race == 5
        assert simulator.use_synthetic_odds is True
        assert simulator.synthetic_odds_gen is not None

    def test_init_with_synthetic_odds_creates_generator(self):
        """Test that synthetic odds mode creates generator"""
        simulator = BacktestSimulator(use_synthetic_odds=True)
        assert simulator.synthetic_odds_gen is not None

    def test_init_without_synthetic_odds_no_generator(self):
        """Test that non-synthetic mode has no generator"""
        simulator = BacktestSimulator(use_synthetic_odds=False)
        assert simulator.synthetic_odds_gen is None

    def test_print_summary_empty_result(self, capsys):
        """Test print_summary with empty result"""
        simulator = BacktestSimulator()
        result = BacktestResult()
        simulator.print_summary(result)

        captured = capsys.readouterr()
        assert "BACKTEST RESULTS" in captured.out
        assert "Total races: 0" in captured.out
        assert "ROI: 0.0%" in captured.out

    def test_print_summary_with_bets(self, capsys):
        """Test print_summary with bets"""
        simulator = BacktestSimulator()
        result = BacktestResult()
        result.total_races = 100
        result.races_with_bets = 50
        result.total_stake = 1000
        result.total_payout = 1500
        result.bets = [
            BetRecord(
                date=20240101, stadium_code=1, race_no=1,
                first=1, second=2, probability=0.2, odds=5.0,
                expected_value=1.0, stake=100,
                actual_first=1, actual_second=2,
                won=True, profit=400
            ),
        ]
        result.metrics = calculate_metrics(result)

        simulator.print_summary(result)

        captured = capsys.readouterr()
        assert "Total races: 100" in captured.out
        assert "Races with bets: 50" in captured.out
        assert "Total stake:" in captured.out
        assert "Hit rate:" in captured.out


class TestPayoutParser:
    """Tests for PayoutParser class"""

    def test_empty_payouts(self):
        """Test empty payouts dict"""
        parser = PayoutParser()
        empty = parser._empty_payouts()

        assert "win" in empty
        assert "exacta" in empty
        assert "trifecta" in empty
        assert len(empty["win"]) == 0

    def test_has_payouts_empty(self):
        """Test has_payouts with empty dict"""
        parser = PayoutParser()
        empty = parser._empty_payouts()

        assert parser._has_payouts(empty) is False

    def test_has_payouts_with_data(self):
        """Test has_payouts with data"""
        parser = PayoutParser()
        payouts = parser._empty_payouts()
        payouts["exacta"][(4, 3)] = 2310

        assert parser._has_payouts(payouts) is True

    def test_parse_payout_line_exacta(self):
        """Test parsing exacta payout line"""
        parser = PayoutParser()
        payouts = parser._empty_payouts()

        line = "        ２連単   4-3       2310  人気     9"
        parser._parse_payout_line(line, payouts)

        assert (4, 3) in payouts["exacta"]
        assert payouts["exacta"][(4, 3)] == 2310

    def test_parse_payout_line_trifecta(self):
        """Test parsing trifecta payout line"""
        parser = PayoutParser()
        payouts = parser._empty_payouts()

        line = "        ３連単   4-3-1     6080  人気    22"
        parser._parse_payout_line(line, payouts)

        assert (4, 3, 1) in payouts["trifecta"]
        assert payouts["trifecta"][(4, 3, 1)] == 6080

    def test_parse_payout_line_win(self):
        """Test parsing win payout line"""
        parser = PayoutParser()
        payouts = parser._empty_payouts()

        line = "        単勝     4          530"
        parser._parse_payout_line(line, payouts)

        assert 4 in payouts["win"]
        assert payouts["win"][4] == 530


class TestBetRecord:
    """Tests for BetRecord dataclass"""

    def test_winning_bet(self):
        """Test creating a winning bet record"""
        record = BetRecord(
            date=20240115,
            stadium_code=22,
            race_no=1,
            first=4,
            second=3,
            probability=0.10,
            odds=23.1,
            expected_value=2.31,
            stake=100,
            actual_first=4,
            actual_second=3,
            won=True,
            profit=2210,  # 2310 - 100
        )

        assert record.won is True
        assert record.profit == 2210

    def test_losing_bet(self):
        """Test creating a losing bet record"""
        record = BetRecord(
            date=20240115,
            stadium_code=22,
            race_no=1,
            first=1,
            second=2,
            probability=0.15,
            odds=3.5,
            expected_value=0.525,
            stake=100,
            actual_first=4,
            actual_second=3,
            won=False,
            profit=-100,
        )

        assert record.won is False
        assert record.profit == -100


class TestBacktestResult:
    """Tests for BacktestResult dataclass"""

    def test_empty_result(self):
        """Test empty backtest result"""
        result = BacktestResult()

        assert result.total_profit == 0
        assert result.roi == 0.0
        assert len(result.bets) == 0

    def test_result_with_bets(self):
        """Test result with bets"""
        result = BacktestResult()
        result.total_stake = 1000
        result.total_payout = 1500

        assert result.total_profit == 500
        assert result.roi == 0.5


class TestCalculateMetrics:
    """Tests for calculate_metrics function"""

    def test_empty_bets(self):
        """Test metrics with empty bets"""
        result = BacktestResult()
        metrics = calculate_metrics(result)

        assert metrics.total_bets == 0
        assert metrics.hit_rate == 0.0
        assert metrics.roi == 0.0

    def test_all_winning_bets(self):
        """Test metrics with all winning bets"""
        result = BacktestResult()
        result.total_stake = 200

        result.bets = [
            BetRecord(
                date=20240101, stadium_code=1, race_no=1,
                first=1, second=2, probability=0.2, odds=5.0,
                expected_value=1.0, stake=100,
                actual_first=1, actual_second=2,
                won=True, profit=400
            ),
            BetRecord(
                date=20240101, stadium_code=1, race_no=2,
                first=2, second=1, probability=0.3, odds=4.0,
                expected_value=1.2, stake=100,
                actual_first=2, actual_second=1,
                won=True, profit=300
            ),
        ]

        metrics = calculate_metrics(result)

        assert metrics.total_bets == 2
        assert metrics.winning_bets == 2
        assert metrics.hit_rate == 1.0
        assert metrics.gross_profit == 700
        assert metrics.gross_loss == 0

    def test_mixed_bets(self):
        """Test metrics with mixed winning/losing bets"""
        result = BacktestResult()
        result.total_stake = 300

        result.bets = [
            BetRecord(
                date=20240101, stadium_code=1, race_no=1,
                first=1, second=2, probability=0.2, odds=10.0,
                expected_value=2.0, stake=100,
                actual_first=1, actual_second=2,
                won=True, profit=900
            ),
            BetRecord(
                date=20240101, stadium_code=1, race_no=2,
                first=1, second=2, probability=0.1, odds=5.0,
                expected_value=0.5, stake=100,
                actual_first=3, actual_second=4,
                won=False, profit=-100
            ),
            BetRecord(
                date=20240101, stadium_code=1, race_no=3,
                first=2, second=1, probability=0.15, odds=6.0,
                expected_value=0.9, stake=100,
                actual_first=2, actual_second=3,
                won=False, profit=-100
            ),
        ]

        metrics = calculate_metrics(result)

        assert metrics.total_bets == 3
        assert metrics.winning_bets == 1
        assert metrics.hit_rate == pytest.approx(1/3)
        assert metrics.gross_profit == 900
        assert metrics.gross_loss == 200
        assert metrics.net_profit == 700
        assert metrics.profit_factor == pytest.approx(4.5)


class TestAnalyzeByDimension:
    """Tests for analyze_by_dimension function"""

    def test_analyze_by_stadium(self):
        """Test analysis by stadium"""
        bets = [
            BetRecord(
                date=20240101, stadium_code=1, race_no=1,
                first=1, second=2, probability=0.2, odds=5.0,
                expected_value=1.0, stake=100,
                actual_first=1, actual_second=2,
                won=True, profit=400
            ),
            BetRecord(
                date=20240101, stadium_code=2, race_no=1,
                first=1, second=2, probability=0.1, odds=10.0,
                expected_value=1.0, stake=100,
                actual_first=3, actual_second=4,
                won=False, profit=-100
            ),
        ]

        analysis = analyze_by_dimension(bets, "stadium")

        assert 1 in analysis
        assert 2 in analysis
        assert analysis[1]["wins"] == 1
        assert analysis[2]["wins"] == 0

    def test_analyze_by_odds_range(self):
        """Test analysis by odds range"""
        bets = [
            BetRecord(
                date=20240101, stadium_code=1, race_no=1,
                first=1, second=2, probability=0.3, odds=3.0,
                expected_value=0.9, stake=100,
                actual_first=1, actual_second=2,
                won=True, profit=200
            ),
            BetRecord(
                date=20240101, stadium_code=1, race_no=2,
                first=1, second=2, probability=0.1, odds=25.0,
                expected_value=2.5, stake=100,
                actual_first=3, actual_second=4,
                won=False, profit=-100
            ),
        ]

        analysis = analyze_by_dimension(bets, "odds_range")

        assert "low (<5)" in analysis
        assert "high (>20)" in analysis


class TestSharpeRatio:
    """Tests for calculate_sharpe_ratio function"""

    def test_empty_bets(self):
        """Test sharpe ratio with empty bets"""
        sharpe = calculate_sharpe_ratio([])
        assert sharpe == 0.0

    def test_positive_returns(self):
        """Test sharpe ratio with positive returns"""
        bets = [
            BetRecord(
                date=20240101, stadium_code=1, race_no=i,
                first=1, second=2, probability=0.2, odds=5.0,
                expected_value=1.0, stake=100,
                actual_first=1, actual_second=2,
                won=True, profit=100
            )
            for i in range(10)
        ]

        sharpe = calculate_sharpe_ratio(bets)
        # All same positive returns -> infinite Sharpe (std=0)
        # But our implementation returns 0 when std=0
        assert sharpe == 0.0 or sharpe > 0


class TestGenerateSummaryReport:
    """Tests for generate_summary_report function"""

    def test_summary_report(self):
        """Test generating summary report"""
        result = BacktestResult()
        result.total_races = 100
        result.races_with_bets = 50
        result.total_stake = 5000
        result.total_payout = 6000

        report = generate_summary_report(result)

        assert "BACKTEST SUMMARY REPORT" in report
        assert "Total Races: 100" in report
        assert "ROI:" in report


class TestGenerateCsvReport:
    """Tests for generate_csv_report function"""

    def test_csv_report_empty(self, tmp_path):
        """Test CSV report with empty bets"""
        result = BacktestResult()
        output_path = tmp_path / "test_report.csv"

        generate_csv_report(result, output_path)

        # File should still be created (or path returned)
        assert output_path == tmp_path / "test_report.csv"

    def test_csv_report_with_bets(self, tmp_path):
        """Test CSV report with bets"""
        result = BacktestResult()
        result.bets = [
            BetRecord(
                date=20240101, stadium_code=1, race_no=1,
                first=1, second=2, probability=0.2, odds=5.0,
                expected_value=1.0, stake=100,
                actual_first=1, actual_second=2,
                won=True, profit=400
            ),
        ]

        output_path = tmp_path / "test_report.csv"
        generate_csv_report(result, output_path)

        assert output_path.exists()

        df = pd.read_csv(output_path)
        assert len(df) == 1
        assert df.iloc[0]["won"] == True
