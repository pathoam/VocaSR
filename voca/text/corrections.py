"""
Correction pipeline for post-processing ASR transcriptions.
Replaces the old SubstitutionManager with a staged pipeline approach.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from voca.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CorrectionRecord:
    """A single correction that was applied."""
    stage: str
    original: str
    corrected: str
    position: int = 0


@dataclass
class CorrectionResult:
    """Result of running text through correction stages."""
    text: str
    corrections: List[CorrectionRecord] = field(default_factory=list)


class CorrectionStage:
    """Base class for correction pipeline stages."""

    name: str = "base"

    def apply(self, text: str) -> CorrectionResult:
        """Apply corrections to text. Must be overridden."""
        raise NotImplementedError


class TermReplacementStage(CorrectionStage):
    """
    Replaces misheard terms with correct forms.
    Direct successor of SubstitutionManager with same data format.
    """

    name = "term_replacement"

    DEFAULT_CORRECTIONS = {
        # Command triggers - normalize variations
        "voice box": "voca",
        "voice-box": "voca",
        "voicebox": "voca",
        # Common tech brand names
        "superbase": "Supabase",
        "super base": "Supabase",
        "versel": "Vercel",
        "versell": "Vercel",
        "get hub": "GitHub",
        "get lab": "GitLab",
        "docker hub": "DockerHub",
        "kubernetes": "Kubernetes",
        "coobernetes": "Kubernetes",
        "python": "Python",
        "pie thon": "Python",
        "java script": "JavaScript",
        "type script": "TypeScript",
        "react": "React",
        "view": "Vue",
        "angular": "Angular",
        "next js": "Next.js",
        "next.js": "Next.js",
        "node js": "Node.js",
        "express js": "Express.js",
        "mongo db": "MongoDB",
        "postgres": "PostgreSQL",
        "my sql": "MySQL",
        "redis": "Redis",
        "elastic search": "Elasticsearch",
        "graph ql": "GraphQL",
        "rest api": "REST API",
        "json": "JSON",
        "yaml": "YAML",
        "toml": "TOML",
        "aws": "AWS",
        "gcp": "GCP",
        "azure": "Azure",
        # Common programming terms
        "a p i": "API",
        "u r l": "URL",
        "u r i": "URI",
        "h t t p": "HTTP",
        "h t t p s": "HTTPS",
        "s q l": "SQL",
        "no sql": "NoSQL",
        "crud": "CRUD",
        "rest full": "RESTful",
        "j w t": "JWT",
        "o auth": "OAuth",
        "sass": "SaaS",
        "pass": "PaaS",
        "i a a s": "IaaS",
        # Common tech terms that get mangled
        "back end": "backend",
        "front end": "frontend",
        "full stack": "fullstack",
        "dev ops": "DevOps",
        "c i c d": "CI/CD",
        "machine learning": "machine learning",
        "a i": "AI",
        "l l m": "LLM",
        "g p t": "GPT",
        # Common commands/tools
        "npm": "npm",
        "yarn": "yarn",
        "pip": "pip",
        "docker": "Docker",
        "compose": "Compose",
        "cube control": "kubectl",
        "cube c t l": "kubectl",
        "vim": "vim",
        "emacs": "Emacs",
        "v s code": "VS Code",
        "visual studio code": "VS Code",
        # Git-specific patterns
        "get push": "git push",
        "get pull": "git pull",
        "get commit": "git commit",
        "get add": "git add",
        "get status": "git status",
        "get checkout": "git checkout",
        "get branch": "git branch",
        "get merge": "git merge",
        "get clone": "git clone",
        "get init": "git init",
        "get remote": "git remote",
        "get fetch": "git fetch",
        "get diff": "git diff",
        "get log": "git log",
        "get reset": "git reset",
    }

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.corrections_file = config_dir / "corrections.json"
        self._old_substitutions_file = config_dir / "substitutions.json"
        self.corrections: Dict[str, str] = {}
        self._deleted_defaults: List[str] = []
        self._compiled_pattern: Optional[re.Pattern] = None
        self._sorted_pairs: List[Tuple[str, str]] = []

        self._migrate_if_needed()
        self.load()

    def _migrate_if_needed(self) -> None:
        """Auto-migrate from substitutions.json if corrections.json doesn't exist."""
        if not self.corrections_file.exists() and self._old_substitutions_file.exists():
            try:
                import shutil
                shutil.copy2(self._old_substitutions_file, self.corrections_file)
                logger.info("Migrated substitutions.json -> corrections.json")
            except Exception as e:
                logger.error(f"Migration failed: {e}")

    def load(self) -> None:
        """Load corrections from config file."""
        self.corrections = self.DEFAULT_CORRECTIONS.copy()
        self._deleted_defaults = []

        if self.corrections_file.exists():
            try:
                with open(self.corrections_file, "r", encoding="utf-8") as f:
                    user_data = json.load(f)

                if "_deleted" in user_data:
                    self._deleted_defaults = user_data["_deleted"]
                    for pattern in self._deleted_defaults:
                        if pattern in self.corrections:
                            del self.corrections[pattern]

                user_corrections = {
                    k: v for k, v in user_data.items() if k != "_deleted"
                }
                self.corrections.update(user_corrections)
                logger.debug(
                    f"Loaded {len(user_corrections)} user corrections, "
                    f"{len(self._deleted_defaults)} deletions"
                )
            except Exception as e:
                logger.error(f"Failed to load corrections: {e}")

        self._compile_pattern()

    def _compile_pattern(self) -> None:
        """Compile a single combined regex for all patterns (longest-first)."""
        self._sorted_pairs = sorted(
            self.corrections.items(), key=lambda x: len(x[0]), reverse=True
        )
        if self._sorted_pairs:
            alternatives = "|".join(
                r"\b" + re.escape(pattern) + r"\b"
                for pattern, _ in self._sorted_pairs
            )
            self._compiled_pattern = re.compile(alternatives, re.IGNORECASE)
        else:
            self._compiled_pattern = None

    def apply(self, text: str) -> CorrectionResult:
        """Apply term replacements to text."""
        if not text or not self._compiled_pattern:
            return CorrectionResult(text=text)

        corrections = []
        # Build a lookup dict for the replacement function
        lookup = {p.lower(): r for p, r in self._sorted_pairs}

        def replacer(match):
            original = match.group(0)
            replacement = lookup.get(original.lower(), original)
            if replacement != original:
                corrections.append(CorrectionRecord(
                    stage=self.name,
                    original=original,
                    corrected=replacement,
                    position=match.start(),
                ))
            return replacement

        result = self._compiled_pattern.sub(replacer, text)
        return CorrectionResult(text=result, corrections=corrections)

    def save(self) -> bool:
        """Save current corrections to file."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            user_corrections = {
                k: v
                for k, v in self.corrections.items()
                if k not in self.DEFAULT_CORRECTIONS
                or self.DEFAULT_CORRECTIONS[k] != v
            }
            save_data = user_corrections.copy()
            if self._deleted_defaults:
                save_data["_deleted"] = self._deleted_defaults
            with open(self.corrections_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save corrections: {e}")
            return False

    def add_correction(self, pattern: str, replacement: str) -> None:
        """Add or update a correction."""
        pattern = pattern.lower()
        self.corrections[pattern] = replacement
        if pattern in self._deleted_defaults:
            self._deleted_defaults.remove(pattern)
        self.save()
        self._compile_pattern()

    def remove_correction(self, pattern: str) -> bool:
        """Remove a correction."""
        pattern = pattern.lower()
        if pattern in self.corrections:
            del self.corrections[pattern]
            if pattern in self.DEFAULT_CORRECTIONS:
                if pattern not in self._deleted_defaults:
                    self._deleted_defaults.append(pattern)
            else:
                if pattern in self._deleted_defaults:
                    self._deleted_defaults.remove(pattern)
            self.save()
            self._compile_pattern()
            return True
        return False

    def get_all_corrections(self) -> Dict[str, str]:
        """Get all current corrections."""
        return self.corrections.copy()

    def reset_to_defaults(self) -> None:
        """Reset corrections to defaults."""
        self.corrections = self.DEFAULT_CORRECTIONS.copy()
        self._deleted_defaults = []
        if self.corrections_file.exists():
            self.corrections_file.unlink()
        self._compile_pattern()

    def import_corrections(self, file_path: str) -> bool:
        """Import corrections from a JSON file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                imported = json.load(f)
            if isinstance(imported, dict):
                if "_deleted" in imported:
                    for pattern in imported["_deleted"]:
                        if pattern not in self._deleted_defaults:
                            self._deleted_defaults.append(pattern)
                    del imported["_deleted"]
                self.corrections.update(imported)
                for pattern in self._deleted_defaults:
                    if pattern in self.corrections:
                        del self.corrections[pattern]
                self.save()
                self._compile_pattern()
                return True
        except Exception as e:
            logger.error(f"Failed to import corrections: {e}")
        return False

    def export_corrections(self, file_path: str) -> bool:
        """Export corrections to a JSON file."""
        try:
            export_data = self.corrections.copy()
            if self._deleted_defaults:
                export_data["_deleted"] = self._deleted_defaults
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to export corrections: {e}")
            return False


class NumberNormalizationStage(CorrectionStage):
    """Converts spoken numbers to digits when appropriate."""

    name = "number_normalization"

    # Word-to-number mappings
    _ONES = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
        "eighteen": 18, "nineteen": 19,
    }
    _TENS = {
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    }
    _SCALES = {
        "hundred": 100, "thousand": 1000, "million": 1_000_000,
        "billion": 1_000_000_000,
    }

    # Pattern to match compound numbers like "twenty three" or "four point nine"
    _NUM_WORDS = set(list(_ONES.keys()) + list(_TENS.keys()) + list(_SCALES.keys()) + ["point", "and"])

    def __init__(self, enabled: bool = True, preserve_words_up_to: int = 5):
        self.enabled = enabled
        self.preserve_words_up_to = preserve_words_up_to

    def apply(self, text: str) -> CorrectionResult:
        if not self.enabled or not text:
            return CorrectionResult(text=text)

        corrections = []
        words = text.split()
        result_words = []
        i = 0

        while i < len(words):
            # Check if this word starts a number sequence
            word_lower = words[i].lower().rstrip(".,;:!?")
            if word_lower in self._NUM_WORDS and word_lower != "and":
                # Collect consecutive number words
                num_words = []
                j = i
                while j < len(words):
                    w = words[j].lower().rstrip(".,;:!?")
                    if w in self._NUM_WORDS:
                        num_words.append(w)
                        j += 1
                    else:
                        break

                # Convert compound/decimal numbers and standalone numbers above
                # the configured small-number threshold.
                if len(num_words) >= 2 or self._should_convert_single_word(num_words[0]):
                    original = " ".join(words[i:j])
                    number_str = self._words_to_number(num_words)
                    if number_str is not None:
                        # Preserve trailing punctuation from last word
                        trailing = ""
                        last_word = words[j - 1]
                        for ch in reversed(last_word):
                            if ch in ".,;:!?":
                                trailing = ch + trailing
                            else:
                                break

                        corrections.append(CorrectionRecord(
                            stage=self.name,
                            original=original,
                            corrected=number_str + trailing,
                            position=len(" ".join(result_words)),
                        ))
                        result_words.append(number_str + trailing)
                        i = j
                        continue
                    if len(num_words) >= 2:
                        result_words.extend(words[i:j])
                        i = j
                        continue

            result_words.append(words[i])
            i += 1

        return CorrectionResult(
            text=" ".join(result_words),
            corrections=corrections,
        )

    def _should_convert_single_word(self, word: str) -> bool:
        if word not in self._ONES and word not in self._TENS:
            return False
        value = self._ONES.get(word, self._TENS.get(word))
        return value is not None and value > self.preserve_words_up_to

    def _words_to_number(self, words: List[str]) -> Optional[str]:
        """Convert a list of number words to a digit string. Returns None on failure."""
        # Handle decimal: "four point nine" -> "4.9"
        if "point" in words:
            point_idx = words.index("point")
            integer_part = words[:point_idx]
            decimal_part = words[point_idx + 1:]

            if not integer_part or not decimal_part:
                return None

            int_val = self._parse_integer(integer_part)
            if int_val is None:
                return None

            # Decimal digits: each word is a single digit
            dec_digits = []
            for w in decimal_part:
                if w in self._ONES and self._ONES[w] <= 9:
                    dec_digits.append(str(self._ONES[w]))
                else:
                    return None
            return f"{int_val}.{''.join(dec_digits)}"

        # Pure integer
        # Filter out "and"
        words = [w for w in words if w != "and"]
        val = self._parse_integer(words)
        if val is not None:
            return str(val)
        return None

    def _parse_integer(self, words: List[str]) -> Optional[int]:
        """Parse a list of number words into an integer.

        Rejects sequences that aren't valid English numbers, e.g.
        "ten thirty" (model number / time) or "forty ninety" — two
        tens-values in a row without a scale word between them.
        """
        if not words:
            return None

        total = 0
        current = 0
        has_tens = False   # current group already has a tens-place word
        has_ones = False   # current group already has a ones-place word

        for word in words:
            if word in self._ONES:
                val = self._ONES[word]
                if has_ones:
                    # Two ones-values in a group: "ten twelve" — not valid
                    return None
                if val >= 10 and has_tens:
                    # e.g. "thirty fifteen" — tens already set
                    return None
                current += val
                has_ones = True
                if val >= 10:
                    # 10-19 also occupy the tens place
                    has_tens = True
            elif word in self._TENS:
                if has_ones:
                    # "twelve thirty" — ones already set, can't add tens
                    return None
                if has_tens:
                    # Two tens-values in a group: "forty ninety" — not valid
                    return None
                current += self._TENS[word]
                has_tens = True
            elif word == "hundred":
                if current == 0:
                    current = 1
                current *= 100
                # Reset — ones/tens can follow ("three hundred twenty one")
                has_tens = False
                has_ones = False
            elif word in self._SCALES:
                if current == 0:
                    current = 1
                total += current * self._SCALES[word]
                current = 0
                has_tens = False
                has_ones = False
            else:
                return None

        total += current
        return total


class SegmentJoinStage(CorrectionStage):
    """Fix spurious sentence breaks from segmented transcription and ASR over-punctuation.

    When audio is transcribed in segments, the model creates artificial sentence
    boundaries at segment edges — a period and capitalized word where the original
    speech was continuous. Qwen3-ASR also tends to over-punctuate in English
    (likely from Chinese-heavy training data). This stage detects and merges
    these breaks using two rules:

    1. Period after non-terminal word: English sentences cannot end with articles,
       prepositions, conjunctions, or determiners. If one does, the period is
       spurious and is removed.
    2. Period before coordinating conjunction: ". And", ". But", ". Or" etc. are
       merged into the previous sentence with a comma.
    """

    name = "segment_join"

    # Words that cannot end an English sentence. If the word immediately before
    # a period is in this set, the period is spurious.
    _NON_TERMINAL = {
        # Articles
        "a", "an", "the",
        # Prepositions
        "of", "on", "in", "at", "for", "with", "to", "from", "by", "about",
        "into", "through", "during", "before", "after", "between", "among",
        "upon", "within", "without", "across", "along", "behind", "below",
        "beneath", "beside", "beyond", "toward", "towards", "until", "under",
        "onto", "per", "via", "than", "like", "near", "above", "against",
        # Conjunctions
        "and", "but", "or", "nor", "so", "yet",
        # Possessives / determiners that always need a following noun.
        # Note: "this", "that", "these", "those" are excluded because they
        # can be demonstrative pronouns that end sentences ("I want that.").
        "its", "their", "his", "her", "our", "your", "my",
        "some", "any", "each", "every", "which", "what", "whose", "no",
    }

    # Coordinating conjunctions that should not start a sentence in
    # spoken transcription. When preceded by a period, merge with comma.
    _CONJUNCTIONS = {"and", "but", "or", "nor", "so", "yet"}

    # Words that, when appearing capitalized after a period, indicate
    # the period was spurious. These are words that almost always continue
    # a preceding clause rather than starting an independent sentence.
    # Excludes personal pronouns (I/He/She/We/They/It/You) and
    # demonstratives (This/That) which genuinely start sentences.
    _CONTINUATION_STARTERS = {
        # Prepositions — "dependent. On systems" → "dependent on systems"
        "on", "in", "at", "of", "for", "with", "from", "by", "about",
        "into", "through", "during", "before", "after", "between",
        "among", "upon", "within", "without", "across", "along",
        "behind", "below", "beneath", "beside", "beyond", "toward",
        "towards", "until", "under", "onto", "per", "via", "than",
        "like", "near", "above", "against",
        # Possessives — "handles. Its details" → "handles its details"
        "its", "their", "his", "her", "our", "your", "my",
        # Subordinating conjunctions
        "because", "since", "although", "though", "while", "unless",
        "whereas", "wherever", "whenever", "whether",
        # Continuation adverbs
        "instead", "also", "just", "even", "still", "then",
        "especially", "particularly", "including",
        "very", "really", "quite", "rather", "fairly", "pretty",
        "simply", "merely", "only", "already", "always", "never",
        "often", "sometimes", "usually", "constantly", "currently",
        "actually", "basically", "certainly", "definitely",
    }

    # Common abbreviations that end with a period but aren't sentence endings.
    # The period after these should never be removed.
    _ABBREVIATIONS = {
        "dr", "mr", "ms", "mrs", "jr", "sr", "st", "vs", "prof", "gen",
        "sgt", "cpl", "pvt", "lt", "capt", "maj", "col", "inc", "corp",
        "ltd", "dept", "approx", "est", "govt", "assn",
    }

    def _apply_pass(self, text: str, pattern: re.Pattern,
                    replacer, corrections: list) -> str:
        """Apply a regex pattern iteratively, collecting corrections."""
        parts = []
        last_end = 0
        for m in pattern.finditer(text):
            replacement = replacer(m)
            if replacement is not None:
                corrections.append(CorrectionRecord(
                    stage=self.name,
                    original=m.group(0),
                    corrected=replacement,
                    position=m.start(),
                ))
                parts.append(text[last_end:m.start()])
                parts.append(replacement)
                last_end = m.end()
        parts.append(text[last_end:])
        return "".join(parts)

    # Words that can start legitimate short sentences — don't merge fragments
    # beginning with these. Includes subject pronouns ("He left."), articles
    # ("The dog ran."), and demonstratives ("That works.").
    _SENTENCE_STARTERS = {
        "i", "he", "she", "we", "they", "it", "you", "one",
        "the", "a", "an", "this", "that", "these", "those",
        "there", "here", "what", "who", "how", "if",
    }

    def apply(self, text: str) -> CorrectionResult:
        if not text:
            return CorrectionResult(text=text)

        corrections = []
        result = text

        # Regex matching "word. Next" — used by passes 1 and 2.
        # [A-Z]\w* matches both "A" (single letter) and "About" (multi-letter).
        word_period_word = re.compile(r'(\b\w+)\.(\s+)([A-Z]\w*)')

        # Pass 1: Period after non-terminal word.
        # "word. Next" → "word next" when word can't end a sentence.
        result = self._apply_pass(result, word_period_word, lambda m: (
            f"{m.group(1)} {m.group(3)[0].lower()}{m.group(3)[1:]}"
            if m.group(1).lower() in self._NON_TERMINAL
            and m.group(1).lower() not in self._ABBREVIATIONS
            else None
        ), corrections)

        # Pass 2: Period before continuation starter word.
        # "dependent. On" → "dependent on", "handles. Its" → "handles its"
        result = self._apply_pass(result, word_period_word, lambda m: (
            f"{m.group(1)} {m.group(3)[0].lower()}{m.group(3)[1:]}"
            if m.group(3).lower() in self._CONTINUATION_STARTERS
            and m.group(1).lower() not in self._ABBREVIATIONS
            else None
        ), corrections)

        # Pass 3: Period or semicolon before coordinating conjunction.
        # ". And" → ", and" / ". But" → ", but"
        conj_alts = "|".join(w.capitalize() for w in self._CONJUNCTIONS)
        p3 = re.compile(r'([.;])\s+(' + conj_alts + r')\b')
        result = self._apply_pass(result, p3, lambda m: (
            f", {m.group(2).lower()}"
        ), corrections)

        # Pass 4: Bare pronoun before period after a comma.
        # "instead, they. Collaborate" → "instead, they collaborate"
        # A pronoun immediately after a comma and before a period is a
        # continuation, not a complete sentence.
        pronouns = "i|he|she|we|they|it|you"
        p4 = re.compile(
            r',\s+(' + pronouns + r')\.(\s+)([A-Z][a-z]\w*)',
            re.IGNORECASE,
        )
        result = self._apply_pass(result, p4, lambda m: (
            f", {m.group(1).lower()} {m.group(3)[0].lower()}{m.group(3)[1:]}"
        ), corrections)

        # Pass 5: Relative pronoun after period.
        # "set up. That can produce" → "set up that can produce"
        # When "That"/"Which"/"Who" after a period is followed by a modal verb,
        # it's almost certainly a relative pronoun, not a demonstrative.
        # Restricted to modals to avoid false positives like
        # "He said that. That was wrong." (be-verb = demonstrative).
        modals = "can|could|will|would|shall|should|may|might|must"
        p5 = re.compile(
            r'(\b\w+)\.\s+(That|Which|Who)\s+(' + modals + r')\b',
            re.IGNORECASE,
        )
        result = self._apply_pass(result, p5, lambda m: (
            f"{m.group(1)} {m.group(2).lower()} {m.group(3).lower()}"
            if m.group(1).lower() not in self._ABBREVIATIONS else None
        ), corrections)

        # Pass 6: Short fragment merging.
        # If a "sentence" between two periods is ≤ 3 words and doesn't
        # start with a subject pronoun, it's likely a segment fragment.
        # Remove the trailing period and merge with the next sentence.
        # Run iteratively since merging can create new short fragments.
        for _ in range(3):  # max iterations to avoid infinite loops
            merged = self._merge_short_fragments(result, corrections)
            if merged == result:
                break
            result = merged

        return CorrectionResult(text=result, corrections=corrections)

    def _merge_short_fragments(self, text: str, corrections: list) -> str:
        """Merge short sentence fragments backward into the preceding sentence.

        If a "sentence" between two periods is ≤ 3 words and doesn't look
        like a complete sentence, remove the period BEFORE it (merge backward)
        and lowercase the fragment's first letter.

        "creates a new. Kind of workflow. The developer..."
        → "creates a new kind of workflow. The developer..."
        """
        # Find all sentence boundaries: period + whitespace + capital letter
        boundaries = list(re.finditer(r'\.(\s+)([A-Z])', text))
        if len(boundaries) < 2:
            return text

        # Identify which boundaries start a short fragment
        to_remove = set()  # indices of boundaries to remove (merge backward)

        for i in range(len(boundaries) - 1):
            curr = boundaries[i]
            nxt = boundaries[i + 1]

            # Fragment text between curr and nxt boundaries
            frag_start = curr.start() + 1  # skip the period
            frag_text = text[frag_start:nxt.start()].strip()
            frag_words = [w for w in frag_text.split() if w.strip(".,;:!?")]

            if not frag_words or len(frag_words) > 3:
                continue

            # Don't merge if the period follows an abbreviation
            pre_period_text = text[:curr.start()].rstrip()
            last_word_before = pre_period_text.split()[-1].lower().rstrip(".,;:!?") if pre_period_text else ""
            if last_word_before in self._ABBREVIATIONS:
                continue

            first_word = frag_words[0].lower().rstrip(".,;:!?")

            # Don't merge if the fragment looks like a complete sentence:
            # starts with a subject pronoun or common sentence starter
            if first_word in self._SENTENCE_STARTERS:
                continue

            # This fragment should be merged backward: remove boundary[i]
            to_remove.add(i)

        if not to_remove:
            return text

        # Rebuild text, removing the periods at marked boundaries
        parts = []
        last_end = 0
        for i, m in enumerate(boundaries):
            if i in to_remove:
                # Remove this period and lowercase the next word
                parts.append(text[last_end:m.start()])  # text before the period
                cap_pos = m.start(2)  # absolute position of the capital letter
                lowered = text[cap_pos].lower()
                parts.append(" " + lowered)
                last_end = cap_pos + 1

                original = text[m.start():cap_pos + 1]
                corrections.append(CorrectionRecord(
                    stage=self.name,
                    original=original,
                    corrected=f" {lowered}",
                    position=m.start(),
                ))

        parts.append(text[last_end:])
        return "".join(parts)


class CorrectionPipeline:
    """Runs text through ordered correction stages."""

    def __init__(self, config_dir: Path, number_normalization_enabled: bool = True):
        self.stages: List[CorrectionStage] = [
            TermReplacementStage(config_dir),
            NumberNormalizationStage(enabled=number_normalization_enabled),
            SegmentJoinStage(),
        ]

    @property
    def term_stage(self) -> TermReplacementStage:
        """Direct access to the term replacement stage."""
        return self.stages[0]

    @property
    def number_stage(self) -> NumberNormalizationStage:
        """Direct access to the number normalization stage."""
        return self.stages[1]

    def apply(self, text: str) -> CorrectionResult:
        """Run text through all correction stages."""
        if not text:
            return CorrectionResult(text=text)

        all_corrections = []
        result = text

        for stage in self.stages:
            stage_result = stage.apply(result)
            result = stage_result.text
            all_corrections.extend(stage_result.corrections)

        return CorrectionResult(text=result, corrections=all_corrections)

    def reload(self) -> None:
        """Reload correction data from disk."""
        for stage in self.stages:
            if hasattr(stage, "load"):
                stage.load()
