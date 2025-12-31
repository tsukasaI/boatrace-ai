"""
Tests for preprocessing module (parser)
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.parser import (
    ProgramParser,
    ResultParser,
    RaceInfo,
    RacerEntry,
    RaceResult,
    convert_to_csv,
    normalize_fullwidth_numbers,
)


class TestProgramParser:
    """Tests for ProgramParser class"""

    def test_normalize_number(self):
        """Test full-width to half-width number conversion"""
        assert normalize_fullwidth_numbers("１２３") == "123"
        assert normalize_fullwidth_numbers("０９") == "09"
        assert normalize_fullwidth_numbers("abc123") == "abc123"

    def test_parse_racer_line_regex(self):
        """Test parsing racer line with regex pattern"""
        parser = ProgramParser()
        # Format matching RACER_PATTERN
        line = "1 3527中嶋誠一53長崎51A2 5.10 30.40 5.91 43.18 55 15.87 78 33.33"

        result = parser._parse_racer_line(line)

        assert result is not None
        assert result.boat_no == 1
        assert result.racer_id == 3527
        assert result.racer_name == "中嶋誠一"
        assert result.age == 53
        assert result.branch == "長崎"
        assert result.weight == 51
        assert result.racer_class == "A2"
        assert result.national_win_rate == 5.10
        assert result.national_in2_rate == 30.40
        assert result.local_win_rate == 5.91
        assert result.local_in2_rate == 43.18
        assert result.motor_no == 55
        assert result.motor_in2_rate == 15.87
        assert result.boat_no_equip == 78
        assert result.boat_in2_rate == 33.33

    def test_parse_racer_line_fallback(self):
        """Test fallback parsing for non-standard format"""
        parser = ProgramParser()
        # Line that doesn't match regex but has correct fixed positions
        line = "1 3527後藤浩之53滋賀51A2 5.10 30.40 5.91 43.18 55 15.87 78 33.33"

        result = parser._parse_racer_line(line)

        # Should parse via fallback
        assert result is not None
        assert result.boat_no == 1
        assert result.racer_id == 3527

    def test_parse_racer_line_invalid(self):
        """Test parsing invalid racer line returns None"""
        parser = ProgramParser()

        assert parser._parse_racer_line("") is None
        assert parser._parse_racer_line("invalid line") is None
        assert parser._parse_racer_line("short") is None

    def test_parse_file_with_sample(self, tmp_path):
        """Test parsing a sample program file"""
        # Create sample program file
        content = """22BBGN  福岡　　　 ２日目

　１Ｒ 予選　　　　　　　　　　　　　Ｈ1800m
艇 登番 選手名   年/支部/体重/級  全国 当地  モーター ボート
1 3527中嶋誠一53長崎51A2 5.10 30.40 5.91 43.18 55 15.87 78 33.33
2 5036福田翔吾25佐賀52B1 5.09 32.63 4.33 33.33 51 28.40 85 24.72

　２Ｒ 予選　　　　　　　　　　　　　Ｈ1800m
1 4861田中宏樹35福岡53B1 4.92 26.03 4.96 37.04 54 24.66 72 29.73
"""
        file_path = tmp_path / "programs_20240115.txt"
        file_path.write_text(content, encoding="cp932")

        parser = ProgramParser()
        races = list(parser.parse_file(file_path))

        assert len(races) == 2

        # First race
        race1, racers1 = races[0]
        assert race1.stadium_code == 22
        assert race1.race_no == 1
        assert race1.distance == 1800
        assert len(racers1) == 2
        assert racers1[0].racer_id == 3527

        # Second race
        race2, racers2 = races[1]
        assert race2.race_no == 2
        assert len(racers2) == 1

    def test_parse_file_unicode_error(self, tmp_path):
        """Test fallback to utf-8 on decode error"""
        # Create file with UTF-8 encoding
        content = "22BBGN  福岡\n　１Ｒ 予選　Ｈ1800m\n1 3527選手名前53東京51A1 5.0 30.0 5.0 30.0 55 15.0 78 33.0"
        file_path = tmp_path / "programs_20240115.txt"
        file_path.write_text(content, encoding="utf-8")

        parser = ProgramParser(encoding="cp932")  # Will fail and fallback
        # Should not raise, should use fallback
        list(parser.parse_file(file_path))  # Just verify it doesn't crash


class TestResultParser:
    """Tests for ResultParser class"""

    def test_normalize_number(self):
        """Test full-width to half-width number conversion"""
        # Only numbers are converted, not letters
        assert normalize_fullwidth_numbers("１２Ｒ") == "12Ｒ"
        assert normalize_fullwidth_numbers("１２３") == "123"

    def test_parse_result_line(self):
        """Test parsing result line"""
        parser = ResultParser()
        line = "  01  4 4861 田中宏樹  54   72  6.80   4    0.05     1.49.6"

        result = parser._parse_result_line(line)

        assert result is not None
        assert result.rank == 1
        assert result.boat_no == 4
        assert result.racer_id == 4861
        assert result.course == 4
        assert result.start_timing == 0.05
        assert result.race_time == "1.49.6"

    def test_parse_result_line_different_ranks(self):
        """Test parsing different rank positions"""
        parser = ResultParser()

        line2 = "  02  3 5160 藤森陸斗  54   84  5.20   3    0.07     1.50.7"
        result2 = parser._parse_result_line(line2)
        assert result2.rank == 2
        assert result2.boat_no == 3

        line6 = "  06  2 5036 福田翔吾  52   85  4.80   2    0.13     1.57.4"
        result6 = parser._parse_result_line(line6)
        assert result6.rank == 6
        assert result6.boat_no == 2

    def test_parse_result_line_invalid(self):
        """Test parsing invalid result line returns None"""
        parser = ResultParser()

        assert parser._parse_result_line("") is None
        assert parser._parse_result_line("invalid line") is None
        assert parser._parse_result_line("単勝 4 1230円") is None

    def test_parse_file_with_sample(self, tmp_path):
        """Test parsing a sample results file"""
        content = """22KBGN  福岡　　　 ２日目

   1R       予選　             H1800m
-----------------------------------------
  01  4 4861 田中    54   72  6.80   4    0.05     1.49.6
  02  3 5160 藤森    54   84  5.20   3    0.07     1.50.7
  03  1 3527 中嶋    51   78  5.10   1    0.04     1.51.2
  04  6 4097 貫地谷  55   71  4.26   6    0.09     1.52.9
  05  5 4876 梅木    53   31  3.66   5    0.09     1.54.9
  06  2 5036 福田    52   85  5.09   2    0.13     1.57.4
単勝 4 1230円

   2R       予選　             H1800m
-----------------------------------------
  01  1 3528 田中    52   60  6.50   1    0.08     1.48.0
"""
        file_path = tmp_path / "results_20240115.txt"
        file_path.write_text(content, encoding="cp932")

        parser = ResultParser()
        races = list(parser.parse_file(file_path))

        assert len(races) == 2

        # First race
        race1, results1 = races[0]
        assert race1.stadium_code == 22
        assert race1.race_no == 1
        assert len(results1) == 6

        # Check order
        assert results1[0].rank == 1
        assert results1[0].boat_no == 4
        assert results1[5].rank == 6

    def test_parse_file_stops_at_payout_section(self, tmp_path):
        """Test that parser stops at payout section"""
        content = """22KBGN  福岡

   1R       予選　             H1800m
-----------------------------------------
  01  4 4861 田中    54   72  6.80   4    0.05     1.49.6
  02  3 5160 藤森    54   84  5.20   3    0.07     1.50.7
単勝 4 1230円
複勝 4 200円
"""
        file_path = tmp_path / "results_20240115.txt"
        file_path.write_text(content, encoding="cp932")

        parser = ResultParser()
        races = list(parser.parse_file(file_path))

        assert len(races) == 1
        _, results = races[0]
        # Should stop before payout lines
        assert len(results) == 2


class TestConvertToCsv:
    """Tests for convert_to_csv function"""

    def test_convert_programs_to_csv(self, tmp_path):
        """Test converting program files to CSV"""
        # Setup directories
        input_dir = tmp_path / "raw"
        output_dir = tmp_path / "processed"
        (input_dir / "programs").mkdir(parents=True)

        # Create sample file
        content = """22BBGN  福岡

　１Ｒ 予選　　　　　　　　　　　　　Ｈ1800m
1 3527中嶋誠一53長崎51A2 5.10 30.40 5.91 43.18 55 15.87 78 33.33
2 5036福田翔吾25佐賀52B1 5.09 32.63 4.33 33.33 51 28.40 85 24.72
"""
        (input_dir / "programs" / "programs_20240115.txt").write_text(
            content, encoding="cp932"
        )

        convert_to_csv(input_dir, output_dir, "programs")

        # Check output files exist
        assert (output_dir / "programs_races.csv").exists()
        assert (output_dir / "programs_entries.csv").exists()

        # Check content
        races_df = pd.read_csv(output_dir / "programs_races.csv")
        entries_df = pd.read_csv(output_dir / "programs_entries.csv")

        assert len(races_df) == 1
        assert len(entries_df) == 2
        assert entries_df.iloc[0]["racer_id"] == 3527

    def test_convert_results_to_csv(self, tmp_path):
        """Test converting results files to CSV"""
        # Setup directories
        input_dir = tmp_path / "raw"
        output_dir = tmp_path / "processed"
        (input_dir / "results").mkdir(parents=True)

        content = """22KBGN  福岡

   1R       予選　             H1800m
-----------------------------------------
  01  4 4861 田中    54   72  6.80   4    0.05     1.49.6
  02  3 5160 藤森    54   84  5.20   3    0.07     1.50.7
単勝 4 1230円
"""
        (input_dir / "results" / "results_20240115.txt").write_text(
            content, encoding="cp932"
        )

        convert_to_csv(input_dir, output_dir, "results")

        # Check output files
        assert (output_dir / "results_races.csv").exists()
        assert (output_dir / "results_entries.csv").exists()

        results_df = pd.read_csv(output_dir / "results_entries.csv")
        assert len(results_df) == 2
        assert results_df.iloc[0]["rank"] == 1

    def test_convert_empty_directory(self, tmp_path):
        """Test converting when no files exist"""
        input_dir = tmp_path / "raw"
        output_dir = tmp_path / "processed"
        (input_dir / "programs").mkdir(parents=True)

        convert_to_csv(input_dir, output_dir, "programs")

        # Should not create files if no data
        assert not (output_dir / "programs_races.csv").exists()


class TestDataClasses:
    """Tests for dataclasses"""

    def test_race_info(self):
        """Test RaceInfo dataclass"""
        race = RaceInfo(
            date="20240115",
            stadium_code=22,
            stadium_name="福岡",
            race_no=1,
            race_type="予選",
            distance=1800,
        )

        assert race.date == "20240115"
        assert race.stadium_code == 22
        assert race.title == ""
        assert race.day == 0

    def test_racer_entry(self):
        """Test RacerEntry dataclass"""
        entry = RacerEntry(
            boat_no=1,
            racer_id=3527,
            racer_name="中嶋誠一",
            age=53,
            branch="長崎",
            weight=51,
            racer_class="A2",
            national_win_rate=5.10,
            national_in2_rate=30.40,
            local_win_rate=5.91,
            local_in2_rate=43.18,
            motor_no=55,
            motor_in2_rate=15.87,
            boat_no_equip=78,
            boat_in2_rate=33.33,
        )

        assert entry.boat_no == 1
        assert entry.racer_class == "A2"

    def test_race_result(self):
        """Test RaceResult dataclass"""
        result = RaceResult(
            boat_no=4,
            racer_id=4861,
            rank=1,
            race_time="1.49.6",
            course=4,
            start_timing=0.05,
        )

        assert result.rank == 1
        assert result.race_time == "1.49.6"
