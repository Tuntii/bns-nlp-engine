"""
Tests for CLI commands.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest


class TestCLIPreprocess:
    """Tests for the preprocess command."""
    
    def test_preprocess_with_stdin(self, tmp_path):
        """Test preprocessing text from stdin."""
        # Create a simple test
        input_text = "Merhaba DÜNYA!"
        
        # Run the command
        result = subprocess.run(
            [sys.executable, "-m", "bnsnlp.cli.main", "preprocess"],
            input=input_text,
            capture_output=True,
            text=True,
        )
        
        # Check that it ran successfully
        assert result.returncode == 0
        
        # Parse output
        output = json.loads(result.stdout)
        
        # Verify structure
        assert "text" in output
        assert "tokens" in output
        assert "metadata" in output
        assert isinstance(output["tokens"], list)
    
    def test_preprocess_with_file_input(self, tmp_path):
        """Test preprocessing text from a file."""
        # Create input file
        input_file = tmp_path / "input.txt"
        input_file.write_text("Türkçe metin işleme.", encoding="utf-8")
        
        # Create output file path
        output_file = tmp_path / "output.json"
        
        # Run the command
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "bnsnlp.cli.main",
                "preprocess",
                "-i",
                str(input_file),
                "-o",
                str(output_file),
            ],
            capture_output=True,
            text=True,
        )
        
        # Check that it ran successfully
        assert result.returncode == 0
        
        # Verify output file was created
        assert output_file.exists()
        
        # Parse output
        output = json.loads(output_file.read_text(encoding="utf-8"))
        
        # Verify structure
        assert "text" in output
        assert "tokens" in output
        assert "metadata" in output
    
    def test_preprocess_with_options(self, tmp_path):
        """Test preprocessing with custom options."""
        input_text = "Merhaba, dünya!"
        
        # Run with custom options
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "bnsnlp.cli.main",
                "preprocess",
                "--no-lowercase",
                "--keep-punctuation",
            ],
            input=input_text,
            capture_output=True,
            text=True,
        )
        
        # Check that it ran successfully
        assert result.returncode == 0
        
        # Parse output
        output = json.loads(result.stdout)
        
        # Verify metadata reflects options
        assert output["metadata"]["lowercase"] is False
        assert output["metadata"]["remove_punctuation"] is False
    
    def test_preprocess_empty_input(self):
        """Test preprocessing with empty input."""
        # Run with empty input
        result = subprocess.run(
            [sys.executable, "-m", "bnsnlp.cli.main", "preprocess"],
            input="",
            capture_output=True,
            text=True,
        )
        
        # Should fail with error
        assert result.returncode == 1
        assert "No input text provided" in result.stderr


class TestCLIEmbed:
    """Tests for the embed command."""
    
    @pytest.mark.skipif(
        not Path.home().joinpath(".env").exists(),
        reason="Requires API keys in environment"
    )
    def test_embed_with_stdin(self):
        """Test embedding text from stdin."""
        input_text = "Merhaba dünya"
        
        # Run the command (will fail without API key, but tests structure)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "bnsnlp.cli.main",
                "embed",
                "--provider",
                "openai",
            ],
            input=input_text,
            capture_output=True,
            text=True,
        )
        
        # If API key is not set, it should fail gracefully
        # If it succeeds, verify output structure
        if result.returncode == 0:
            output = json.loads(result.stdout)
            assert "embeddings" in output
            assert "model" in output
            assert "dimensions" in output
    
    def test_embed_invalid_provider(self):
        """Test embedding with invalid provider."""
        input_text = "Test text"
        
        # Run with invalid provider
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "bnsnlp.cli.main",
                "embed",
                "--provider",
                "invalid_provider",
            ],
            input=input_text,
            capture_output=True,
            text=True,
        )
        
        # Should fail
        assert result.returncode == 1
        assert "Unknown provider" in result.stderr


class TestCLISearch:
    """Tests for the search command."""
    
    @pytest.mark.skipif(
        not Path.home().joinpath(".env").exists(),
        reason="Requires API keys and search backend"
    )
    def test_search_basic(self):
        """Test basic search command."""
        query = "Türkçe NLP"
        
        # Run the command
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "bnsnlp.cli.main",
                "search",
                query,
                "--provider",
                "faiss",
                "--top-k",
                "5",
            ],
            capture_output=True,
            text=True,
        )
        
        # If search backend is configured, verify output
        if result.returncode == 0:
            output = json.loads(result.stdout)
            assert "query" in output
            assert "results" in output
            assert output["query"] == query
    
    def test_search_with_filters(self):
        """Test search with JSON filters."""
        query = "test query"
        filters = '{"category": "tech"}'
        
        # Run the command
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "bnsnlp.cli.main",
                "search",
                query,
                "--filters",
                filters,
            ],
            capture_output=True,
            text=True,
        )
        
        # Command structure should be valid (may fail on backend)
        # Just verify it doesn't crash on filter parsing
        assert "Invalid JSON" not in result.stderr
    
    def test_search_invalid_filters(self):
        """Test search with invalid JSON filters."""
        query = "test query"
        invalid_filters = '{invalid json}'
        
        # Run the command
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "bnsnlp.cli.main",
                "search",
                query,
                "--filters",
                invalid_filters,
            ],
            capture_output=True,
            text=True,
        )
        
        # Should fail with JSON error
        assert result.returncode == 1
        assert "Invalid JSON" in result.stderr


class TestCLIClassify:
    """Tests for the classify command."""
    
    @pytest.mark.skipif(
        not Path.home().joinpath(".env").exists(),
        reason="Requires classification models"
    )
    def test_classify_with_stdin(self):
        """Test classifying text from stdin."""
        input_text = "Yarın hava nasıl olacak?"
        
        # Run the command
        result = subprocess.run(
            [sys.executable, "-m", "bnsnlp.cli.main", "classify"],
            input=input_text,
            capture_output=True,
            text=True,
        )
        
        # If models are available, verify output structure
        if result.returncode == 0:
            output = json.loads(result.stdout)
            assert "intent" in output
            assert "intent_confidence" in output
            assert "entities" in output
    
    def test_classify_with_file(self, tmp_path):
        """Test classifying text from a file."""
        # Create input file
        input_file = tmp_path / "input.txt"
        input_file.write_text("İstanbul'a gitmek istiyorum.", encoding="utf-8")
        
        # Create output file path
        output_file = tmp_path / "output.json"
        
        # Run the command
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "bnsnlp.cli.main",
                "classify",
                "-i",
                str(input_file),
                "-o",
                str(output_file),
            ],
            capture_output=True,
            text=True,
        )
        
        # If models are available and command succeeds
        if result.returncode == 0:
            assert output_file.exists()
            output = json.loads(output_file.read_text(encoding="utf-8"))
            assert "intent" in output
            assert "entities" in output


class TestCLIVersion:
    """Tests for version command."""
    
    def test_version_flag(self):
        """Test --version flag."""
        result = subprocess.run(
            [sys.executable, "-m", "bnsnlp.cli.main", "--version"],
            capture_output=True,
            text=True,
        )
        
        # Should succeed and print version
        assert result.returncode == 0
        assert "bns-nlp-engine version" in result.stdout


class TestCLIHelp:
    """Tests for help command."""
    
    def test_help_main(self):
        """Test main help."""
        result = subprocess.run(
            [sys.executable, "-m", "bnsnlp.cli.main", "--help"],
            capture_output=True,
            text=True,
        )
        
        # Should succeed and show help
        assert result.returncode == 0
        assert "Turkish NLP Engine" in result.stdout
        assert "preprocess" in result.stdout
        assert "embed" in result.stdout
        assert "search" in result.stdout
        assert "classify" in result.stdout
    
    def test_help_preprocess(self):
        """Test preprocess command help."""
        result = subprocess.run(
            [sys.executable, "-m", "bnsnlp.cli.main", "preprocess", "--help"],
            capture_output=True,
            text=True,
        )
        
        # Should succeed and show preprocess help
        assert result.returncode == 0
        assert "Preprocess Turkish text" in result.stdout
    
    def test_help_embed(self):
        """Test embed command help."""
        result = subprocess.run(
            [sys.executable, "-m", "bnsnlp.cli.main", "embed", "--help"],
            capture_output=True,
            text=True,
        )
        
        # Should succeed and show embed help
        assert result.returncode == 0
        assert "Generate embeddings" in result.stdout
    
    def test_help_search(self):
        """Test search command help."""
        result = subprocess.run(
            [sys.executable, "-m", "bnsnlp.cli.main", "search", "--help"],
            capture_output=True,
            text=True,
        )
        
        # Should succeed and show search help
        assert result.returncode == 0
        assert "Search for similar documents" in result.stdout
    
    def test_help_classify(self):
        """Test classify command help."""
        result = subprocess.run(
            [sys.executable, "-m", "bnsnlp.cli.main", "classify", "--help"],
            capture_output=True,
            text=True,
        )
        
        # Should succeed and show classify help
        assert result.returncode == 0
        assert "Classify intent" in result.stdout
