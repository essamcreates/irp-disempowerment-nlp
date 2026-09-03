"""Deterministic synthetic noise injection for robustness experiments."""

from __future__ import annotations

import hashlib
import random
import re
from typing import Dict, List, Sequence


NOISE_VERSION = "noise_v1"

NOISE_CONDITIONS = (
    "clean",
    "casing",
    "punctuation",
    "whitespace",
    "typo",
    "word_deletion",
    "filler",
    "mixed",
)

CASING_RATE = 0.10
PUNCTUATION_RATE = 0.20
WHITESPACE_RATE = 0.10
TYPO_RATE = 0.03
WORD_DELETION_RATE = 0.02
FILLER_RATE = 0.03

MAX_WORD_DELETION_RATE = 0.10

FILLERS = (
    "um",
    "uh",
    "well",
    "you know",
)


def _normalise_input(text: object) -> str:
    """Convert an input value to text without otherwise cleaning it."""

    if text is None:
        return ""

    return str(text)


def _stable_rng(
    base_seed: int,
    example_key: str,
    condition: str,
    stage: str,
) -> random.Random:
    """
    Create a deterministic random-number generator.

    SHA-256 is used instead of Python's built-in hash() so that the same
    example, condition, stage, and base seed produce the same result across
    Python sessions.
    """

    payload = (
        f"{base_seed}|{example_key}|{condition}|{stage}"
    ).encode("utf-8")

    digest = hashlib.sha256(payload).digest()

    seed = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    return random.Random(seed)


def _target_count(
    n_items: int,
    rate: float,
) -> int:
    """
    Convert a perturbation rate into an approximate deterministic count.

    Very small candidate sets may legitimately receive zero perturbations.
    """

    if n_items <= 0:
        return 0

    return min(
        n_items,
        int((n_items * rate) + 0.5),
    )


def _sample_indices(
    rng: random.Random,
    n_items: int,
    rate: float,
) -> List[int]:
    """Sample unique candidate indexes at approximately the requested rate."""

    count = _target_count(
        n_items=n_items,
        rate=rate,
    )

    if count == 0:
        return []

    return sorted(
        rng.sample(
            range(n_items),
            k=count,
        )
    )


# ---------------------------------------------------------------------
# Individual noise transformations
# ---------------------------------------------------------------------


def inject_casing_noise(
    text: object,
    rng: random.Random,
    rate: float = CASING_RATE,
) -> str:
    """Alter the case of approximately `rate` eligible alphabetic words."""

    source = _normalise_input(text)

    matches = list(
        re.finditer(
            r"\b[A-Za-z]+\b",
            source,
        )
    )

    selected = _sample_indices(
        rng,
        len(matches),
        rate,
    )

    if not selected:
        return source

    replacements = []

    for index in selected:
        match = matches[index]
        word = match.group(0)

        use_upper = rng.choice(
            [True, False]
        )

        replacement = (
            word.upper()
            if use_upper
            else word.lower()
        )

        # Ensure that a selected token is actually changed.
        if replacement == word:
            replacement = (
                word.lower()
                if use_upper
                else word.upper()
            )

        replacements.append(
            (
                match.start(),
                match.end(),
                replacement,
            )
        )

    result = source

    for start, end, replacement in reversed(
        replacements
    ):
        result = (
            result[:start]
            + replacement
            + result[end:]
        )

    return result


def inject_punctuation_noise(
    text: object,
    rng: random.Random,
    rate: float = PUNCTUATION_RATE,
) -> str:
    """Repeat a proportion of existing sentence-ending punctuation."""

    source = _normalise_input(text)

    matches = list(
        re.finditer(
            r"[.!?]",
            source,
        )
    )

    selected = _sample_indices(
        rng,
        len(matches),
        rate,
    )

    if not selected:
        return source

    replacements = []

    for index in selected:
        match = matches[index]
        mark = match.group(0)

        if mark == ".":
            replacement = "..."
        else:
            replacement = mark * rng.choice(
                [2, 3]
            )

        replacements.append(
            (
                match.start(),
                match.end(),
                replacement,
            )
        )

    result = source

    for start, end, replacement in reversed(
        replacements
    ):
        result = (
            result[:start]
            + replacement
            + result[end:]
        )

    return result


def inject_whitespace_noise(
    text: object,
    rng: random.Random,
    rate: float = WHITESPACE_RATE,
) -> str:
    """
    Corrupt eligible whitespace boundaries without reordering words.

    Operations include:
    - duplicating a word-boundary space;
    - removing a word-boundary space;
    - inserting a space immediately before adjacent punctuation.
    """

    source = _normalise_input(text)

    candidates = []

    # Existing spaces between alphanumeric tokens.
    for match in re.finditer(
        r"(?<=[A-Za-z0-9]) (?=[A-Za-z0-9])",
        source,
    ):
        candidates.append(
            (
                match.start(),
                match.end(),
                "word_boundary",
            )
        )

    # Positions immediately before punctuation.
    for match in re.finditer(
        r"(?<=[A-Za-z0-9])(?=[,;:.!?])",
        source,
    ):
        candidates.append(
            (
                match.start(),
                match.end(),
                "punctuation_boundary",
            )
        )

    selected = _sample_indices(
        rng,
        len(candidates),
        rate,
    )

    if not selected:
        return source

    replacements = []

    for index in selected:
        start, end, boundary_type = candidates[index]

        if boundary_type == "word_boundary":
            replacement = rng.choice(
                [
                    "",
                    "  ",
                ]
            )
        else:
            replacement = " "

        replacements.append(
            (
                start,
                end,
                replacement,
            )
        )

    result = source

    for start, end, replacement in sorted(
        replacements,
        key=lambda item: item[0],
        reverse=True,
    ):
        result = (
            result[:start]
            + replacement
            + result[end:]
        )

    return result


def _perturb_word(
    word: str,
    rng: random.Random,
) -> str:
    """Apply one internal character perturbation to an alphabetic word."""

    if len(word) < 4:
        return word

    characters = list(word)

    operation = rng.choice(
        [
            "swap",
            "delete",
            "duplicate",
        ]
    )

    if operation == "swap":
        # Select two adjacent internal characters.
        index = rng.randint(
            1,
            len(characters) - 3,
        )

        characters[index], characters[index + 1] = (
            characters[index + 1],
            characters[index],
        )

        return "".join(characters)

    index = rng.randrange(
        1,
        len(characters) - 1,
    )

    if operation == "delete":
        del characters[index]

    elif operation == "duplicate":
        characters.insert(
            index,
            characters[index],
        )

    return "".join(characters)


def inject_typo_noise(
    text: object,
    rng: random.Random,
    rate: float = TYPO_RATE,
) -> str:
    """Apply character-level perturbations to eligible alphabetic words."""

    source = _normalise_input(text)

    matches = [
        match
        for match in re.finditer(
            r"\b[A-Za-z]+\b",
            source,
        )
        if len(match.group(0)) >= 4
    ]

    selected = _sample_indices(
        rng,
        len(matches),
        rate,
    )

    if not selected:
        return source

    replacements = []

    for index in selected:
        match = matches[index]

        replacement = _perturb_word(
            match.group(0),
            rng,
        )

        replacements.append(
            (
                match.start(),
                match.end(),
                replacement,
            )
        )

    result = source

    for start, end, replacement in reversed(
        replacements
    ):
        result = (
            result[:start]
            + replacement
            + result[end:]
        )

    return result


def inject_word_deletion_noise(
    text: object,
    rng: random.Random,
    rate: float = WORD_DELETION_RATE,
) -> str:
    """
    Delete a small proportion of alphabetic words while retaining at least
    90% of the original eligible word tokens.
    """

    source = _normalise_input(text)

    matches = list(
        re.finditer(
            r"\b[A-Za-z]+\b",
            source,
        )
    )

    n_words = len(matches)

    if n_words == 0:
        return source

    requested_count = _target_count(
        n_words,
        rate,
    )

    max_deletions = int(
        n_words * MAX_WORD_DELETION_RATE
    )

    deletion_count = min(
        requested_count,
        max_deletions,
    )

    if deletion_count <= 0:
        return source

    selected = sorted(
        rng.sample(
            range(n_words),
            k=deletion_count,
        )
    )

    deletion_spans = []

    for index in selected:
        match = matches[index]

        start = match.start()
        end = match.end()

        # Prefer removing following horizontal whitespace so that deletion
        # does not unnecessarily create doubled spaces.
        expanded_end = end

        while (
            expanded_end < len(source)
            and source[expanded_end] in " \t"
        ):
            expanded_end += 1

        if expanded_end > end:
            end = expanded_end

        deletion_spans.append(
            (
                start,
                end,
            )
        )

    result = source

    for start, end in reversed(
        deletion_spans
    ):
        result = (
            result[:start]
            + result[end:]
        )

    return result


def inject_filler_noise(
    text: object,
    rng: random.Random,
    rate: float = FILLER_RATE,
    fillers: Sequence[str] = FILLERS,
) -> str:
    """Insert filler/disfluency expressions at eligible word boundaries."""

    source = _normalise_input(text)

    matches = list(
        re.finditer(
            r"(?<=[A-Za-z0-9]) (?=[A-Za-z0-9])",
            source,
        )
    )

    selected = _sample_indices(
        rng,
        len(matches),
        rate,
    )

    if not selected:
        return source

    replacements = []

    for index in selected:
        match = matches[index]

        filler = rng.choice(
            list(fillers)
        )

        replacement = (
            f" {filler} "
        )

        replacements.append(
            (
                match.start(),
                match.end(),
                replacement,
            )
        )

    result = source

    for start, end, replacement in reversed(
        replacements
    ):
        result = (
            result[:start]
            + replacement
            + result[end:]
        )

    return result


# ---------------------------------------------------------------------
# Condition-level interface
# ---------------------------------------------------------------------


def apply_noise_condition(
    text: object,
    condition: str,
    *,
    example_key: str,
    base_seed: int = 42,
) -> str:
    """Apply one deterministic noise condition to a response."""

    source = _normalise_input(text)

    if condition not in NOISE_CONDITIONS:
        raise ValueError(
            f"Unknown noise condition: {condition}"
        )

    if condition == "clean":
        return source

    transforms = {
        "casing": inject_casing_noise,
        "punctuation": inject_punctuation_noise,
        "whitespace": inject_whitespace_noise,
        "typo": inject_typo_noise,
        "word_deletion": inject_word_deletion_noise,
        "filler": inject_filler_noise,
    }

    if condition != "mixed":
        rng = _stable_rng(
            base_seed=base_seed,
            example_key=example_key,
            condition=condition,
            stage=condition,
        )

        return transforms[condition](
            source,
            rng,
        )

    result = source

    for stage in (
        "casing",
        "punctuation",
        "whitespace",
        "typo",
        "word_deletion",
        "filler",
    ):
        rng = _stable_rng(
            base_seed=base_seed,
            example_key=example_key,
            condition="mixed",
            stage=stage,
        )

        result = transforms[stage](
            result,
            rng,
        )

    return result


def generate_noise_variants(
    text: object,
    *,
    source_index: object,
    pair_index: object,
    base_seed: int = 42,
) -> Dict[str, str]:
    """Generate all clean and noisy conditions for one source response."""

    example_key = (
        f"{source_index}:{pair_index}"
    )

    return {
        condition: apply_noise_condition(
            text,
            condition,
            example_key=example_key,
            base_seed=base_seed,
        )
        for condition in NOISE_CONDITIONS
    }