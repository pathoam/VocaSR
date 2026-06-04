"""Correction flow — handles the correction hotkey dialog and vocabulary auto-add."""

from typing import Optional

from voca.utils.logging import get_logger

logger = get_logger(__name__)


class CorrectionFlow:
    """Manages the correction hotkey: overlay dialog, text replacement, vocabulary update."""

    def __init__(self, text_inserter, training_data_collector, vocabulary_manager, use_gui: bool = False):
        self.text_inserter = text_inserter
        self.training_data_collector = training_data_collector
        self.vocabulary_manager = vocabulary_manager
        self.use_gui = use_gui

    def handle_correction(self, last_insertion: Optional[dict]) -> None:
        """Open correction overlay for the last transcription, apply replacement."""
        if not last_insertion:
            logger.debug("No recent transcription to correct")
            return

        original_text = last_insertion["text"]

        from voca.ui.review import prompt_correction
        corrected = prompt_correction(original_text, use_gui=self.use_gui)

        if corrected is None or corrected == original_text:
            return

        if self.text_inserter:
            success = self.text_inserter.select_and_replace(
                last_insertion["char_count"], corrected
            )
            if success:
                logger.info(f"Correction applied: '{original_text}' -> '{corrected}'")

                if self.training_data_collector and last_insertion.get("data_id"):
                    self.training_data_collector.update_sample(
                        last_insertion["data_id"], corrected
                    )

                if self.vocabulary_manager:
                    existing_terms = {t.lower() for t in self.vocabulary_manager.get_terms()}
                    corrected_words = set(corrected.split())
                    original_words = set(original_text.split())
                    new_words = corrected_words - original_words
                    for word in new_words:
                        if len(word) > 2 and word.lower() not in existing_terms:
                            if word[0].isupper() or any(c.isdigit() for c in word):
                                self.vocabulary_manager.add_term(word)
                                logger.info(f"Auto-added '{word}' to vocabulary")
