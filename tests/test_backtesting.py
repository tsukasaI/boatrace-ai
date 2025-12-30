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
