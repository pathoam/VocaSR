"""Tests for CorrectionFlow — correction application, vocabulary auto-add.

The correction dialog (prompt_correction) is UI-dependent, so we patch it.
But the vocabulary auto-add logic is tested with a real VocabularyManager
to verify the filtering rules (capitalized words, words with digits, length > 2).
"""

import pytest
from unittest.mock import patch, MagicMock

from voca.correction_flow import CorrectionFlow
from voca.text.vocabulary import VocabularyManager


@pytest.fixture
def vocab_manager(tmp_config_dir):
    return VocabularyManager(tmp_config_dir)


@pytest.fixture
def text_inserter():
    inserter = MagicMock()
    inserter.select_and_replace.return_value = True
    return inserter


@pytest.fixture
def collector():
    return MagicMock()


@pytest.fixture
def last_insertion():
    return {
        "text": "hello world",
        "char_count": 11,
        "audio_file": "/tmp/test.wav",
        "data_id": "sample_001",
    }


class TestNoOps:
    def test_none_insertion_does_nothing(self, text_inserter, collector, vocab_manager):
        flow = CorrectionFlow(text_inserter, collector, vocab_manager)
        flow.handle_correction(None)
        text_inserter.select_and_replace.assert_not_called()

    @patch("voca.ui.review.prompt_correction", return_value=None)
    def test_cancelled_dialog_does_nothing(self, mock_prompt, text_inserter, collector,
                                            vocab_manager, last_insertion):
        flow = CorrectionFlow(text_inserter, collector, vocab_manager, use_gui=False)
        flow.handle_correction(last_insertion)
        text_inserter.select_and_replace.assert_not_called()

    @patch("voca.ui.review.prompt_correction", return_value="hello world")
    def test_unchanged_text_does_nothing(self, mock_prompt, text_inserter, collector,
                                          vocab_manager, last_insertion):
        flow = CorrectionFlow(text_inserter, collector, vocab_manager, use_gui=False)
        flow.handle_correction(last_insertion)
        text_inserter.select_and_replace.assert_not_called()


class TestCorrectionApplied:
    @patch("voca.ui.review.prompt_correction", return_value="hello universe")
    def test_replaces_text_via_inserter(self, mock_prompt, text_inserter, collector,
                                         vocab_manager, last_insertion):
        flow = CorrectionFlow(text_inserter, collector, vocab_manager, use_gui=False)
        flow.handle_correction(last_insertion)

        text_inserter.select_and_replace.assert_called_once_with(11, "hello universe")

    @patch("voca.ui.review.prompt_correction", return_value="hello universe")
    def test_updates_training_data(self, mock_prompt, text_inserter, collector,
                                    vocab_manager, last_insertion):
        flow = CorrectionFlow(text_inserter, collector, vocab_manager, use_gui=False)
        flow.handle_correction(last_insertion)

        collector.update_sample.assert_called_once_with("sample_001", "hello universe")

    @patch("voca.ui.review.prompt_correction", return_value="hello universe")
    def test_no_training_update_without_data_id(self, mock_prompt, text_inserter, collector,
                                                  vocab_manager):
        insertion = {"text": "hello world", "char_count": 11, "audio_file": "/tmp/test.wav"}
        flow = CorrectionFlow(text_inserter, collector, vocab_manager, use_gui=False)
        flow.handle_correction(insertion)

        collector.update_sample.assert_not_called()

    @patch("voca.ui.review.prompt_correction", return_value="corrected")
    def test_no_crash_without_collector(self, mock_prompt, text_inserter, vocab_manager,
                                         last_insertion):
        flow = CorrectionFlow(text_inserter, None, vocab_manager, use_gui=False)
        flow.handle_correction(last_insertion)  # should not raise

    @patch("voca.ui.review.prompt_correction", return_value="corrected")
    def test_no_crash_when_replace_fails(self, mock_prompt, collector, vocab_manager,
                                          last_insertion):
        inserter = MagicMock()
        inserter.select_and_replace.return_value = False
        flow = CorrectionFlow(inserter, collector, vocab_manager, use_gui=False)
        flow.handle_correction(last_insertion)
        # Should not update training data or vocab if replace failed
        collector.update_sample.assert_not_called()


class TestVocabularyAutoAdd:
    """The auto-add logic: new capitalized words or words with digits get added."""

    @patch("voca.ui.review.prompt_correction", return_value="hello Kubernetes")
    def test_adds_capitalized_new_word(self, mock_prompt, text_inserter, collector,
                                        vocab_manager, last_insertion):
        flow = CorrectionFlow(text_inserter, collector, vocab_manager, use_gui=False)
        flow.handle_correction(last_insertion)

        terms = vocab_manager.get_terms()
        assert "Kubernetes" in terms

    @patch("voca.ui.review.prompt_correction", return_value="hello world2024")
    def test_adds_word_with_digits(self, mock_prompt, text_inserter, collector,
                                    vocab_manager, last_insertion):
        flow = CorrectionFlow(text_inserter, collector, vocab_manager, use_gui=False)
        flow.handle_correction(last_insertion)

        terms = vocab_manager.get_terms()
        assert "world2024" in terms

    @patch("voca.ui.review.prompt_correction", return_value="hello planet")
    def test_does_not_add_lowercase_word(self, mock_prompt, text_inserter, collector,
                                          vocab_manager, last_insertion):
        flow = CorrectionFlow(text_inserter, collector, vocab_manager, use_gui=False)
        flow.handle_correction(last_insertion)

        terms = vocab_manager.get_terms()
        assert "planet" not in terms

    @patch("voca.ui.review.prompt_correction", return_value="hello Go")
    def test_does_not_add_short_word(self, mock_prompt, text_inserter, collector,
                                      vocab_manager, last_insertion):
        """Words with len <= 2 are skipped even if capitalized."""
        flow = CorrectionFlow(text_inserter, collector, vocab_manager, use_gui=False)
        flow.handle_correction(last_insertion)

        terms = vocab_manager.get_terms()
        assert "Go" not in terms

    @patch("voca.ui.review.prompt_correction", return_value="hello World Kubernetes")
    def test_does_not_add_word_already_in_original(self, mock_prompt, text_inserter, collector,
                                                     vocab_manager):
        """'World' was in the original, so only 'Kubernetes' should be added."""
        insertion = {"text": "hello World", "char_count": 11,
                     "audio_file": "/tmp/test.wav", "data_id": "x"}
        flow = CorrectionFlow(text_inserter, collector, vocab_manager, use_gui=False)
        flow.handle_correction(insertion)

        terms = vocab_manager.get_terms()
        assert "Kubernetes" in terms
        # World was already in original — should not be added
        assert "World" not in terms

    @patch("voca.ui.review.prompt_correction", return_value="hello Kubernetes")
    def test_does_not_add_duplicate_term(self, mock_prompt, text_inserter, collector,
                                          vocab_manager, last_insertion):
        vocab_manager.add_term("Kubernetes")  # already exists
        flow = CorrectionFlow(text_inserter, collector, vocab_manager, use_gui=False)
        flow.handle_correction(last_insertion)

        terms = vocab_manager.get_terms()
        assert terms.count("Kubernetes") == 1

    @patch("voca.ui.review.prompt_correction", return_value="corrected")
    def test_no_crash_without_vocab_manager(self, mock_prompt, text_inserter, collector,
                                             last_insertion):
        flow = CorrectionFlow(text_inserter, collector, None, use_gui=False)
        flow.handle_correction(last_insertion)  # should not raise
