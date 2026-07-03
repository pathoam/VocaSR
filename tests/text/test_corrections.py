"""Tests for the correction pipeline: term replacement, number normalization, segment join."""

import json

import pytest

from voca.text.corrections import (
    CorrectionPipeline,
    CorrectionRecord,
    CorrectionResult,
    NumberNormalizationStage,
    SegmentJoinStage,
    TermReplacementStage,
)


# ---------------------------------------------------------------------------
# TermReplacementStage
# ---------------------------------------------------------------------------

class TestTermReplacementStage:
    """Tests for TermReplacementStage."""

    def test_default_corrections_apply(self, tmp_config_dir):
        stage = TermReplacementStage(tmp_config_dir)
        result = stage.apply("I use superbase for my backend")
        assert "Supabase" in result.text

    def test_voice_box_to_voca(self, tmp_config_dir):
        stage = TermReplacementStage(tmp_config_dir)
        result = stage.apply("open voice box settings")
        assert "Voca" in result.text

    def test_optagon_defaults_apply(self, tmp_config_dir):
        stage = TermReplacementStage(tmp_config_dir)
        result = stage.apply("open optigon in the frame")
        assert result.text == "open Optagon in the frame"

    def test_qwen_defaults_apply(self, tmp_config_dir):
        stage = TermReplacementStage(tmp_config_dir)
        result = stage.apply("start quinn asr")
        assert result.text == "start Qwen ASR"

    def test_default_name_correction_applies(self, tmp_config_dir):
        stage = TermReplacementStage(tmp_config_dir)
        result = stage.apply("send this to glen")
        assert result.text == "send this to Glenn"

    def test_case_insensitive_matching(self, tmp_config_dir):
        stage = TermReplacementStage(tmp_config_dir)
        result = stage.apply("I deployed on SUPERBASE yesterday")
        assert "Supabase" in result.text

    def test_longest_match_first(self, tmp_config_dir):
        """'chat GPT' should not be partially matched as just 'GPT'."""
        stage = TermReplacementStage(tmp_config_dir)
        # "g p t" is 5 chars, "visual studio code" is 18 chars
        # The longer pattern "visual studio code" should match before "v s code"
        result = stage.apply("I use visual studio code daily")
        assert result.text == "I use VS Code daily"

    def test_correction_record_tracking(self, tmp_config_dir):
        stage = TermReplacementStage(tmp_config_dir)
        result = stage.apply("deploy to superbase")
        assert len(result.corrections) >= 1
        rec = result.corrections[0]
        assert isinstance(rec, CorrectionRecord)
        assert rec.stage == "term_replacement"
        assert rec.corrected == "Supabase"

    def test_no_match_passthrough(self, tmp_config_dir):
        stage = TermReplacementStage(tmp_config_dir)
        text = "hello world this is a normal sentence"
        result = stage.apply(text)
        assert result.text == text
        assert result.corrections == []

    def test_empty_text(self, tmp_config_dir):
        stage = TermReplacementStage(tmp_config_dir)
        result = stage.apply("")
        assert result.text == ""
        assert result.corrections == []

    def test_add_correction(self, tmp_config_dir):
        stage = TermReplacementStage(tmp_config_dir)
        stage.add_correction("foobar", "FooBar")
        result = stage.apply("I like foobar")
        assert "FooBar" in result.text

    def test_add_correction_persists(self, tmp_config_dir):
        stage = TermReplacementStage(tmp_config_dir)
        stage.add_correction("bazzle", "Bazzle")
        # Reload from disk
        stage2 = TermReplacementStage(tmp_config_dir)
        result = stage2.apply("try bazzle now")
        assert "Bazzle" in result.text

    def test_remove_correction(self, tmp_config_dir):
        stage = TermReplacementStage(tmp_config_dir)
        stage.add_correction("xyzzy", "XYZZY")
        assert stage.remove_correction("xyzzy") is True
        result = stage.apply("try xyzzy now")
        assert "XYZZY" not in result.text

    def test_remove_default_correction(self, tmp_config_dir):
        stage = TermReplacementStage(tmp_config_dir)
        assert stage.remove_correction("superbase") is True
        result = stage.apply("deploy superbase")
        # Should no longer replace
        assert "Supabase" not in result.text
        # Reload — deletion should persist
        stage2 = TermReplacementStage(tmp_config_dir)
        result2 = stage2.apply("deploy superbase")
        assert "Supabase" not in result2.text

    def test_remove_nonexistent_returns_false(self, tmp_config_dir):
        stage = TermReplacementStage(tmp_config_dir)
        assert stage.remove_correction("nonexistent_pattern_xyz") is False

    def test_get_all_corrections(self, tmp_config_dir):
        stage = TermReplacementStage(tmp_config_dir)
        all_corr = stage.get_all_corrections()
        assert isinstance(all_corr, dict)
        assert "superbase" in all_corr
        # Returned value should be a copy
        all_corr["test"] = "test"
        assert "test" not in stage.corrections

    def test_reset_to_defaults(self, tmp_config_dir):
        stage = TermReplacementStage(tmp_config_dir)
        stage.add_correction("custom", "Custom")
        stage.remove_correction("superbase")
        stage.reset_to_defaults()
        assert "superbase" in stage.corrections
        assert "custom" not in stage.corrections

    def test_export_and_import(self, tmp_config_dir, tmp_path):
        stage = TermReplacementStage(tmp_config_dir)
        stage.add_correction("testterm", "TestTerm")

        export_path = str(tmp_path / "export.json")
        assert stage.export_corrections(export_path) is True

        # Create a fresh stage and import
        config2 = tmp_path / "config2"
        config2.mkdir()
        stage2 = TermReplacementStage(config2)
        assert stage2.import_corrections(export_path) is True
        result = stage2.apply("use testterm here")
        assert "TestTerm" in result.text

    def test_import_nonexistent_file(self, tmp_config_dir):
        stage = TermReplacementStage(tmp_config_dir)
        assert stage.import_corrections("/nonexistent/path.json") is False

    def test_user_corrections_file_loaded(self, tmp_config_dir):
        """User corrections.json overrides and extends defaults."""
        corrections_file = tmp_config_dir / "corrections.json"
        corrections_file.write_text(json.dumps({"myword": "MyWord"}))
        stage = TermReplacementStage(tmp_config_dir)
        result = stage.apply("try myword")
        assert "MyWord" in result.text
        # Defaults should still be present
        assert "superbase" in stage.corrections


# ---------------------------------------------------------------------------
# NumberNormalizationStage
# ---------------------------------------------------------------------------

class TestNumberNormalizationStage:
    """Tests for NumberNormalizationStage."""

    def test_simple_number_twenty_three(self):
        stage = NumberNormalizationStage()
        result = stage.apply("twenty three")
        assert result.text == "23"

    def test_simple_number_one_hundred(self):
        stage = NumberNormalizationStage()
        result = stage.apply("one hundred")
        assert result.text == "100"

    def test_compound_number(self):
        stage = NumberNormalizationStage()
        result = stage.apply("one thousand two hundred thirty four")
        assert result.text == "1234"

    def test_decimal(self):
        stage = NumberNormalizationStage()
        result = stage.apply("four point nine")
        assert result.text == "4.9"

    def test_rejection_ten_thirty(self):
        """'ten thirty' looks like a time, not a number — should not convert."""
        stage = NumberNormalizationStage()
        result = stage.apply("ten thirty")
        assert result.text == "ten thirty"

    def test_rejection_forty_ninety(self):
        """Two tens-values in a row is invalid."""
        stage = NumberNormalizationStage()
        result = stage.apply("forty ninety")
        assert result.text == "forty ninety"

    def test_rejection_thirty_fifteen(self):
        """Tens followed by teens is invalid."""
        stage = NumberNormalizationStage()
        result = stage.apply("thirty fifteen")
        assert result.text == "thirty fifteen"

    def test_mixed_text(self):
        stage = NumberNormalizationStage()
        result = stage.apply("I have twenty three apples")
        assert result.text == "I have 23 apples"

    def test_disabled_mode(self):
        stage = NumberNormalizationStage(enabled=False)
        result = stage.apply("twenty three")
        assert result.text == "twenty three"
        assert result.corrections == []

    def test_standalone_zero_not_converted(self):
        """Small standalone number words are preserved."""
        stage = NumberNormalizationStage()
        result = stage.apply("zero")
        assert result.text == "zero"

    def test_standalone_five_not_converted(self):
        stage = NumberNormalizationStage()
        result = stage.apply("I need five items")
        assert result.text == "I need five items"

    def test_standalone_six_converted(self):
        stage = NumberNormalizationStage()
        result = stage.apply("I need six items")
        assert result.text == "I need 6 items"

    def test_standalone_ten_converted_with_punctuation(self):
        stage = NumberNormalizationStage()
        result = stage.apply("There are ten.")
        assert result.text == "There are 10."

    def test_custom_small_number_threshold(self):
        stage = NumberNormalizationStage(preserve_words_up_to=2)
        result = stage.apply("I need three items")
        assert result.text == "I need 3 items"

    def test_compound_with_zero(self):
        """'five hundred' is compound and converts."""
        stage = NumberNormalizationStage()
        result = stage.apply("five hundred")
        assert result.text == "500"

    def test_correction_records(self):
        stage = NumberNormalizationStage()
        result = stage.apply("I need twenty three items")
        assert len(result.corrections) == 1
        assert result.corrections[0].stage == "number_normalization"
        assert result.corrections[0].original == "twenty three"
        assert result.corrections[0].corrected == "23"


# ---------------------------------------------------------------------------
# SegmentJoinStage
# ---------------------------------------------------------------------------

class TestSegmentJoinStage:
    """Tests for SegmentJoinStage."""

    def test_pass1_non_terminal_period_removal(self):
        """Period after a non-terminal word should be removed."""
        stage = SegmentJoinStage()
        result = stage.apply("the. Quick brown fox")
        assert "the quick" in result.text.lower()
        # The period should be gone
        assert "the." not in result.text.lower()

    def test_pass2_continuation_starters(self):
        """Period before a continuation starter should be removed."""
        stage = SegmentJoinStage()
        result = stage.apply("dependent. On the system")
        assert "dependent on" in result.text

    def test_pass3_conjunction_period_to_comma(self):
        """Period before a conjunction becomes a comma."""
        stage = SegmentJoinStage()
        result = stage.apply("It is fast. And reliable")
        assert ", and" in result.text

    def test_pass4_bare_pronoun_merge(self):
        """Bare pronoun after comma + before period should merge."""
        stage = SegmentJoinStage()
        result = stage.apply("instead, they. Collaborate on projects")
        assert "they collaborate" in result.text

    def test_pass5_relative_pronoun_modal(self):
        """Relative pronoun + modal after period should merge."""
        stage = SegmentJoinStage()
        result = stage.apply("set up. That can produce results")
        assert "set up that can produce" in result.text

    def test_pass6_short_fragment_merge(self):
        """Fragments <= 3 words should merge backward."""
        stage = SegmentJoinStage()
        # "Kind of workflow" is 3 words, not a sentence starter
        result = stage.apply("creates a new. Kind of workflow. The developer works")
        assert "new kind of workflow" in result.text

    def test_abbreviation_preservation(self):
        """Abbreviations like 'Dr.' should not be affected."""
        stage = SegmentJoinStage()
        result = stage.apply("Dr. Smith is here")
        assert "Dr. Smith" in result.text

    def test_no_op_on_clean_text(self):
        """Text without spurious breaks should pass through unchanged."""
        stage = SegmentJoinStage()
        text = "The quick brown fox jumps over the lazy dog."
        result = stage.apply(text)
        assert result.text == text
        assert result.corrections == []

    def test_correction_count_tracking(self):
        """Each join should produce a CorrectionRecord."""
        stage = SegmentJoinStage()
        result = stage.apply("It is fast. And reliable. But complex")
        assert len(result.corrections) >= 2


# ---------------------------------------------------------------------------
# CorrectionPipeline
# ---------------------------------------------------------------------------

class TestCorrectionPipeline:
    """Tests for CorrectionPipeline end-to-end."""

    def test_pipeline_runs_all_stages(self, tmp_config_dir):
        pipeline = CorrectionPipeline(tmp_config_dir)
        # Input that triggers term replacement AND number normalization
        result = pipeline.apply("I use superbase with twenty three nodes")
        assert "Supabase" in result.text
        assert "23" in result.text

    def test_pipeline_accumulates_corrections(self, tmp_config_dir):
        pipeline = CorrectionPipeline(tmp_config_dir)
        result = pipeline.apply("deploy superbase with twenty three instances")
        # Should have corrections from both term and number stages
        stages_seen = {c.stage for c in result.corrections}
        assert "term_replacement" in stages_seen
        assert "number_normalization" in stages_seen

    def test_pipeline_empty_text(self, tmp_config_dir):
        pipeline = CorrectionPipeline(tmp_config_dir)
        result = pipeline.apply("")
        assert result.text == ""
        assert result.corrections == []
