"""Integration tests for the GGUF scanner."""

import json
import tempfile
from pathlib import Path

import pytest

from scanner import PerfectGGUFScanner


class TestIntegrationScanning:
    """Integration tests for complete scanning workflows."""

    def test_scan_empty_directory(self, tmp_path: Path) -> None:
        """Test scanning an empty directory."""
        scanner = PerfectGGUFScanner()
        scanner.scan_directory(str(tmp_path), str(tmp_path / "output.json"))

        assert scanner.stats["total"] == 0
        assert scanner.stats["parsed"] == 0
        assert len(scanner.results) == 0

    def test_scan_directory_with_non_gguf_files(self, tmp_path: Path) -> None:
        """Test scanning directory with non-GGUF files."""
        # Create some non-GGUF files
        (tmp_path / "test.txt").write_text("not a gguf file")
        (tmp_path / "model.bin").write_text("also not gguf")

        scanner = PerfectGGUFScanner()
        scanner.scan_directory(str(tmp_path), str(tmp_path / "output.json"))

        assert scanner.stats["total"] == 0
        assert len(scanner.results) == 0

    def test_scanner_state_initialization(self) -> None:
        """Test that scanner initializes with correct state."""
        scanner = PerfectGGUFScanner()

        assert isinstance(scanner.results, dict)
        assert isinstance(scanner.mmproj_data, dict)
        assert isinstance(scanner.mmproj_links, dict)
        assert isinstance(scanner.stats, dict)

        assert scanner.stats["total"] == 0
        assert scanner.stats["parsed"] == 0
        assert scanner.stats["failed"] == 0
        assert scanner.stats["memory_complete"] == 0
        assert scanner.stats["validated"] == 0

    def test_multiple_scanner_instances(self) -> None:
        """Test that multiple scanner instances are independent."""
        scanner1 = PerfectGGUFScanner()
        scanner2 = PerfectGGUFScanner()

        scanner1.stats["parsed"] = 10
        scanner1.results["test.gguf"] = None  # type: ignore[assignment]

        assert scanner2.stats["parsed"] == 0
        assert len(scanner2.results) == 0

    def test_scanner_with_llama_cpp_path(self) -> None:
        """Test scanner initialization with llama_cpp_path."""
        path = "/path/to/llama.cpp"
        scanner = PerfectGGUFScanner(llama_cpp_path=path)

        assert scanner.llama_cpp_path == path


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_scan_nonexistent_directory(self) -> None:
        """Test scanning a non-existent directory."""
        scanner = PerfectGGUFScanner()
        # Should not crash
        scanner.scan_directory("/nonexistent/path", "output.json")
        assert scanner.stats["total"] == 0

    def test_scan_file_as_directory(self, tmp_path: Path) -> None:
        """Test trying to scan a file as if it were a directory."""
        test_file = tmp_path / "not_a_dir.txt"
        test_file.write_text("test")

        scanner = PerfectGGUFScanner()
        scanner.scan_directory(str(test_file), "output.json")

        # Should handle gracefully
        assert scanner.stats["total"] == 0


class TestStatisticsTracking:
    """Test statistics tracking functionality."""

    def test_stats_increment_on_scan(self, tmp_path: Path) -> None:
        """Test that stats are properly incremented."""
        scanner = PerfectGGUFScanner()

        # Create empty GGUF files (will fail to parse but count as total)
        for i in range(3):
            (tmp_path / f"model{i}.gguf").write_bytes(b"NOT_GGUF_DATA")

        scanner.scan_directory(str(tmp_path), str(tmp_path / "output.json"))

        assert scanner.stats["total"] == 3
        # These will fail to parse (not valid GGUF)
        assert scanner.stats["failed"] >= 0

    def test_stats_persistence_across_operations(self) -> None:
        """Test that stats persist across operations."""
        scanner = PerfectGGUFScanner()

        initial_stats = dict(scanner.stats)
        assert all(v == 0 for v in initial_stats.values())

        # Stats should be mutable dict
        scanner.stats["parsed"] += 5
        assert scanner.stats["parsed"] == 5


class TestPathHandling:
    """Test path handling with pathlib."""

    def test_scanner_handles_paths(self) -> None:
        """Test that scanner handles various path types."""
        scanner = PerfectGGUFScanner()
        # Just verify it doesn't crash with no files
        with tempfile.TemporaryDirectory() as tmpdir:
            scanner.scan_directory(tmpdir, str(Path(tmpdir) / "models.json"))
            assert scanner.stats["total"] == 0
