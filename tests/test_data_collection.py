"""
Tests for data_collection module (downloader, extractor)
"""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.downloader import BoatraceDataDownloader
from src.data_collection.extractor import LzhExtractor


class TestBoatraceDataDownloader:
    """Tests for BoatraceDataDownloader class"""

    def test_init_creates_directories(self, tmp_path):
        """Test that __init__ creates results and programs directories"""
        downloader = BoatraceDataDownloader(output_dir=tmp_path)

        assert (tmp_path / "results").exists()
        assert (tmp_path / "programs").exists()

    def test_build_url_results(self, tmp_path):
        """Test URL building for results data"""
        downloader = BoatraceDataDownloader(output_dir=tmp_path)
        url = downloader._build_url("results", date(2024, 3, 15))

        assert "202403" in url
        assert "k240315.lzh" in url
        assert url.startswith("https://")

    def test_build_url_programs(self, tmp_path):
        """Test URL building for programs data"""
        downloader = BoatraceDataDownloader(output_dir=tmp_path)
        url = downloader._build_url("programs", date(2024, 12, 1))

        assert "202412" in url
        assert "b241201.lzh" in url
        assert url.startswith("https://")

    @patch("src.data_collection.downloader.requests.get")
    def test_download_file_success(self, mock_get, tmp_path):
        """Test successful file download"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"test content"
        mock_get.return_value = mock_response

        downloader = BoatraceDataDownloader(output_dir=tmp_path)
        output_path = tmp_path / "test.lzh"

        result = downloader._download_file("http://example.com/test.lzh", output_path)

        assert result is True
        assert output_path.exists()
        assert output_path.read_bytes() == b"test content"

    @patch("src.data_collection.downloader.requests.get")
    def test_download_file_404(self, mock_get, tmp_path):
        """Test download with 404 response (no retry)"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        downloader = BoatraceDataDownloader(output_dir=tmp_path)
        output_path = tmp_path / "test.lzh"

        result = downloader._download_file("http://example.com/test.lzh", output_path)

        assert result is False
        assert not output_path.exists()
        # 404 should not retry
        assert mock_get.call_count == 1

    @patch("src.data_collection.downloader.time.sleep")
    @patch("src.data_collection.downloader.requests.get")
    def test_download_file_500_retry(self, mock_get, mock_sleep, tmp_path):
        """Test download with 500 response triggers retry"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        downloader = BoatraceDataDownloader(output_dir=tmp_path)
        output_path = tmp_path / "test.lzh"

        result = downloader._download_file(
            "http://example.com/test.lzh", output_path, max_retries=3
        )

        assert result is False
        assert mock_get.call_count == 3  # Should retry 3 times
        assert mock_sleep.call_count == 2  # Sleep between retries

    @patch("src.data_collection.downloader.time.sleep")
    @patch("src.data_collection.downloader.requests.get")
    def test_download_file_timeout_retry(self, mock_get, mock_sleep, tmp_path):
        """Test download timeout triggers retry"""
        mock_get.side_effect = requests.Timeout()

        downloader = BoatraceDataDownloader(output_dir=tmp_path)
        output_path = tmp_path / "test.lzh"

        result = downloader._download_file(
            "http://example.com/test.lzh", output_path, max_retries=2
        )

        assert result is False
        assert mock_get.call_count == 2

    @patch("src.data_collection.downloader.time.sleep")
    @patch("src.data_collection.downloader.requests.get")
    def test_download_file_retry_then_success(self, mock_get, mock_sleep, tmp_path):
        """Test download succeeds after retry"""
        # First call fails, second succeeds
        fail_response = Mock()
        fail_response.status_code = 503

        success_response = Mock()
        success_response.status_code = 200
        success_response.content = b"success"

        mock_get.side_effect = [fail_response, success_response]

        downloader = BoatraceDataDownloader(output_dir=tmp_path)
        output_path = tmp_path / "test.lzh"

        result = downloader._download_file(
            "http://example.com/test.lzh", output_path, max_retries=3
        )

        assert result is True
        assert output_path.read_bytes() == b"success"

    @patch("src.data_collection.downloader.time.sleep")
    @patch("src.data_collection.downloader.requests.get")
    def test_download_date_range_skips_existing(self, mock_get, mock_sleep, tmp_path):
        """Test that existing files are skipped"""
        # Create existing file
        (tmp_path / "results").mkdir(parents=True)
        existing_file = tmp_path / "results" / "results_20240115.lzh"
        existing_file.write_bytes(b"existing")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"new content"
        mock_get.return_value = mock_response

        downloader = BoatraceDataDownloader(output_dir=tmp_path)
        stats = downloader.download_date_range(
            date(2024, 1, 15), date(2024, 1, 15), data_types=["results"]
        )

        assert stats["results"]["skip"] == 1
        assert stats["results"]["success"] == 0
        # File should not be overwritten
        assert existing_file.read_bytes() == b"existing"

    @patch("src.data_collection.downloader.time.sleep")
    @patch("src.data_collection.downloader.requests.get")
    def test_download_date_range_multiple_days(self, mock_get, mock_sleep, tmp_path):
        """Test downloading multiple days"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"data"
        mock_get.return_value = mock_response

        downloader = BoatraceDataDownloader(output_dir=tmp_path)
        stats = downloader.download_date_range(
            date(2024, 1, 1), date(2024, 1, 3), data_types=["results"]
        )

        assert stats["results"]["success"] == 3
        assert (tmp_path / "results" / "results_20240101.lzh").exists()
        assert (tmp_path / "results" / "results_20240102.lzh").exists()
        assert (tmp_path / "results" / "results_20240103.lzh").exists()


class TestLzhExtractor:
    """Tests for LzhExtractor class"""

    def test_init_with_custom_dirs(self, tmp_path):
        """Test initialization with custom directories"""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        extractor = LzhExtractor(input_dir=input_dir, output_dir=output_dir)

        assert extractor.input_dir == input_dir
        assert extractor.output_dir == output_dir

    @patch("src.data_collection.extractor.HAS_LHAFILE", False)
    def test_extract_file_no_lhafile(self, tmp_path):
        """Test extract_file returns None when lhafile not installed"""
        extractor = LzhExtractor(input_dir=tmp_path, output_dir=tmp_path)
        lzh_path = tmp_path / "results" / "test.lzh"
        lzh_path.parent.mkdir(parents=True)
        lzh_path.touch()

        result = extractor.extract_file(lzh_path)

        assert result is None

    @patch("src.data_collection.extractor.lhafile")
    @patch("src.data_collection.extractor.HAS_LHAFILE", True)
    def test_extract_file_success(self, mock_lhafile_module, tmp_path):
        """Test successful file extraction"""
        # Setup mock archive
        mock_info = Mock()
        mock_info.filename = "test.txt"

        mock_archive = Mock()
        mock_archive.infolist.return_value = [mock_info]
        mock_archive.read.return_value = b"extracted content"

        mock_lhafile_module.Lhafile.return_value = mock_archive

        # Create input structure
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        (input_dir / "results").mkdir(parents=True)
        output_dir.mkdir()

        lzh_path = input_dir / "results" / "results_20240115.lzh"
        lzh_path.write_bytes(b"lzh data")

        extractor = LzhExtractor(input_dir=input_dir, output_dir=output_dir)
        result = extractor.extract_file(lzh_path)

        assert result is not None
        expected_output = output_dir / "results" / "results_20240115.txt"
        assert expected_output.exists()
        assert expected_output.read_bytes() == b"extracted content"

    @patch("src.data_collection.extractor.lhafile")
    @patch("src.data_collection.extractor.HAS_LHAFILE", True)
    def test_extract_file_multiple_files_in_archive(self, mock_lhafile_module, tmp_path):
        """Test extraction of archive with multiple files"""
        # Setup mock archive with multiple files
        mock_info1 = Mock()
        mock_info1.filename = "file1.txt"
        mock_info2 = Mock()
        mock_info2.filename = "file2.txt"

        mock_archive = Mock()
        mock_archive.infolist.return_value = [mock_info1, mock_info2]
        mock_archive.read.side_effect = [b"content1", b"content2"]

        mock_lhafile_module.Lhafile.return_value = mock_archive

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        (input_dir / "results").mkdir(parents=True)
        output_dir.mkdir()

        lzh_path = input_dir / "results" / "test.lzh"
        lzh_path.write_bytes(b"lzh data")

        extractor = LzhExtractor(input_dir=input_dir, output_dir=output_dir)
        result = extractor.extract_file(lzh_path)

        # Should return path to first extracted file
        assert result is not None
        # Both files should be read
        assert mock_archive.read.call_count == 2

    @patch("src.data_collection.extractor.lhafile")
    @patch("src.data_collection.extractor.HAS_LHAFILE", True)
    def test_extract_file_error_handling(self, mock_lhafile_module, tmp_path):
        """Test error handling during extraction"""
        mock_lhafile_module.Lhafile.side_effect = OSError("Read error")
        mock_lhafile_module.LhaError = Exception

        input_dir = tmp_path / "input"
        (input_dir / "results").mkdir(parents=True)

        lzh_path = input_dir / "results" / "test.lzh"
        lzh_path.write_bytes(b"corrupt data")

        extractor = LzhExtractor(input_dir=input_dir, output_dir=tmp_path)
        result = extractor.extract_file(lzh_path)

        assert result is None

    @patch("src.data_collection.extractor.lhafile")
    @patch("src.data_collection.extractor.HAS_LHAFILE", True)
    def test_extract_all_skips_existing(self, mock_lhafile_module, tmp_path):
        """Test that extract_all skips existing txt files"""
        input_dir = tmp_path / "input"
        (input_dir / "results").mkdir(parents=True)

        # Create LZH file and corresponding TXT file
        lzh_path = input_dir / "results" / "results_20240115.lzh"
        txt_path = input_dir / "results" / "results_20240115.txt"
        lzh_path.write_bytes(b"lzh data")
        txt_path.write_bytes(b"existing txt")

        extractor = LzhExtractor(input_dir=input_dir, output_dir=input_dir)
        stats = extractor.extract_all(data_types=["results"])

        assert stats["results"]["skip"] == 1
        assert stats["results"]["success"] == 0
        # Lhafile should not be called
        mock_lhafile_module.Lhafile.assert_not_called()

    @patch("src.data_collection.extractor.lhafile")
    @patch("src.data_collection.extractor.HAS_LHAFILE", True)
    def test_extract_all_multiple_files(self, mock_lhafile_module, tmp_path):
        """Test extracting multiple LZH files"""
        # Setup mock
        mock_info = Mock()
        mock_info.filename = "data.txt"
        mock_archive = Mock()
        mock_archive.infolist.return_value = [mock_info]
        mock_archive.read.return_value = b"content"
        mock_lhafile_module.Lhafile.return_value = mock_archive

        input_dir = tmp_path / "input"
        (input_dir / "results").mkdir(parents=True)

        # Create multiple LZH files
        for i in range(3):
            lzh_path = input_dir / "results" / f"results_2024010{i+1}.lzh"
            lzh_path.write_bytes(b"lzh data")

        extractor = LzhExtractor(input_dir=input_dir, output_dir=input_dir)
        stats = extractor.extract_all(data_types=["results"])

        assert stats["results"]["success"] == 3
        assert mock_lhafile_module.Lhafile.call_count == 3

    def test_extract_all_missing_directory(self, tmp_path):
        """Test extract_all handles missing directory gracefully"""
        extractor = LzhExtractor(input_dir=tmp_path, output_dir=tmp_path)
        stats = extractor.extract_all(data_types=["results"])

        assert stats["results"]["success"] == 0
        assert stats["results"]["skip"] == 0
        assert stats["results"]["fail"] == 0
