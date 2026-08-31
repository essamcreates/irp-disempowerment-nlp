"""Filtering and deterministic pilot sampling for LMSYS-Chat-1M."""

from __future__ import annotations

import random
import re
from collections.abc import Iterable
from typing import Any

from src.config import (
    MAX_ASSISTANT_CHARS,
    MAX_USER_CHARS,
    MIN_ASSISTANT_CHARS,
    MIN_USER_CHARS,
    RANDOM_SEED,
)


# ---------------------------------------------------------------------
# More precise relevance rules
# ---------------------------------------------------------------------
#
# These rules deliberately favour FIRST-PERSON personal context.
# This reduces false positives from fiction, technical tasks, articles,
# coding questions, and quoted/source text.
#
# These are still only relevance-selection rules.
# They are NOT disempowerment labels.
# ---------------------------------------------------------------------

RELEVANCE_PATTERNS = {
    "emotional": [
        # Direct first-person emotional states
        r"\bi feel (?:very |really |so )?(?:sad|angry|lonely|stressed|anxious|worried|overwhelmed|hurt|insecure|guilty|upset|down)\b",

        r"\bi'm feeling (?:very |really |so )?(?:sad|angry|lonely|stressed|anxious|worried|overwhelmed|hurt|insecure|guilty|upset|down)\b",

        r"\bi am feeling (?:very |really |so )?(?:sad|angry|lonely|stressed|anxious|worried|overwhelmed|hurt|insecure|guilty|upset|down)\b",

        r"\bi've been feeling (?:very |really |so )?(?:sad|angry|lonely|stressed|anxious|worried|overwhelmed|hurt|insecure|guilty|upset|down)\b",

        r"\bi have been feeling (?:very |really |so )?(?:sad|angry|lonely|stressed|anxious|worried|overwhelmed|hurt|insecure|guilty|upset|down)\b",

        # Direct "I am / I'm" emotional disclosures
        r"\bi'm (?:very |really |so )?(?:sad|angry|lonely|stressed|anxious|worried|overwhelmed|hurt|insecure|guilty|upset|down)\b",

        r"\bi am (?:very |really |so )?(?:sad|angry|lonely|stressed|anxious|worried|overwhelmed|hurt|insecure|guilty|upset|down)\b",

        # Personal emotional-state nouns
        r"\bmy (?:anxiety|stress|confidence|self[- ]esteem|emotions|feelings)\b",
    ],

    "relational": [
        r"\bmy relationship\b",
        r"\bmy partner\b",
        r"\bmy boyfriend\b",
        r"\bmy girlfriend\b",
        r"\bmy husband\b",
        r"\bmy wife\b",
        r"\bmy ex\b",
        r"\bmy friendship\b",
        r"\bmy family\b",
        r"\bmy mother\b",
        r"\bmy father\b",
        r"\bmy sister\b",
        r"\bmy brother\b",
        r"\bwe broke up\b",
        r"\bwe've broken up\b",
        r"\bi'm dating\b",
        r"\bi am dating\b",
    ],

    "personal_lifestyle": [
        r"\bmy life\b",
        r"\bmy situation\b",
        r"\bmy career\b",
        r"\bmy job\b",
        r"\bmy work situation\b",
        r"\bmy studies\b",
        r"\bmy university\b",
        r"\bmy college\b",
        r"\bmy routine\b",
        r"\bmy habit\b",
        r"\bmy habits\b",
        r"\bmy goal\b",
        r"\bmy goals\b",
        r"\bmy future\b",
        r"\bmy personal life\b",
    ],
}


ADVICE_PATTERNS = [
    r"\bwhat should i do\b",
    r"\bwhat i should do\b",
    r"\bshould i\b",
    r"\bdo you think i should\b",
    r"\bwhat would you do\b",
    r"\bhelp me decide\b",
    r"\bi can't decide\b",
    r"\bi cannot decide\b",
    r"\bi'm trying to decide\b",
    r"\bi am trying to decide\b",
    r"\bneed advice\b",
    r"\bcan you give me advice\b",
]

# Help-seeking and personal decision signals.
#
# These are used to distinguish genuinely personal conversations
# from prompts that merely contain terms such as "my girlfriend",
# "my job", or "my future".
PERSONAL_HELP_PATTERNS = [
    r"\bcan you help\b",
    r"\bcould you help\b",
    r"\bhelp me\b",
    r"\bwhat should i do\b",
    r"\bwhat can i do\b",
    r"\bwhat do i do\b",
    r"\bwhat would you do\b",
    r"\bhow should i\b",
    r"\bhow can i\b",
    r"\bhow would i\b",
    r"\bshould i\b",
    r"\bneed advice\b",
    r"\bany advice\b",
    r"\badvise me\b",
    r"\bi don't know what to do\b",
    r"\bi do not know what to do\b",
    r"\bi'm not sure what to do\b",
    r"\bi am not sure what to do\b",
    r"\btrying to decide\b",
    r"\bcan't decide\b",
    r"\bcannot decide\b",
]

# Generic "should I?" is too broad by itself.
# Advice/decision prompts must also mention one of these personal-life domains.
PERSONAL_DECISION_DOMAINS = [
    # Relationships
    r"\bmy relationship\b",
    r"\bmy partner\b",
    r"\bmy boyfriend\b",
    r"\bmy girlfriend\b",
    r"\bmy husband\b",
    r"\bmy wife\b",
    r"\bmy ex\b",
    r"\bmy family\b",
    r"\bmy friendship\b",

    # Career / work
    r"\bmy career\b",
    r"\bmy job\b",
    r"\bmy work situation\b",
    r"\bat work\b",
    r"\bchange careers?\b",
    r"\bcareer change\b",

    # Education
    r"\bmy studies\b",
    r"\bmy university\b",
    r"\bmy college\b",
    r"\bat university\b",
    r"\bat college\b",

    # Broader life decisions
    r"\bmy life\b",
    r"\bmy future\b",
    r"\bmy situation\b",
    r"\bmove house\b",
    r"\bmove abroad\b",
    r"\brelocat(?:e|ing|ion)\b",
    r"\bmy routine\b",
    r"\bmy habits?\b",
]

TASK_EXCLUSION_PATTERNS = [
    # Common benchmark / instruction-style prompts
    r"^\s*write a\b",
    r"^\s*please write\b",
    r"^\s*create a\b",
    r"^\s*generate a\b",
    r"^\s*given the\b",
    r"^\s*solve (?:this|the)\b",
    r"^\s*the student\b",
    r"^\s*simplify\b",
    r"^\s*summarize\b",
    r"^\s*summarise\b",
    r"^\s*classify\b",
    r"^\s*determine\b",
    r"^\s*translate\b",

    # Source-text / evaluation tasks
    r"\bsource text\b",
    r"\bgiven the article\b",
    r"\bgiven the text\b",
    r"\bgiven the passage\b",
    r"\bwhich choice\b",
    r"\banswer the following\b",
    r"\bfactually consistent\b",

    # Explicit creative-writing tasks
    r"\bscreenplay\b",
    r"\bsynopsis\b",
    r"\bwrite a story\b",
    r"\bwrite a dialogue\b",
    r'^\s*["“].*\bNAME_\d+\b.*\b(?:asked|said|replied|whispered|shouted)\b',

    # Message transformation / analysis tasks
    r"^\s*please reply to\b",
    r"^\s*reply to\b",
    r"^\s*please read the following\b",
    r"^\s*read the following\b",
    r"^\s*please rewrite\b",
    r"^\s*rewrite\b",

# Joke / supplied-text tasks
    r"^\s*explain the joke\b",
    r"^\s*explain this joke\b",

# Roleplay / fictional scenarios
    r"\broleplay\b",
    r"\brole play\b",
    r"\broleplaying\b",
    r"\brole playing\b",
    r"\bwe were roleplaying\b",
    r"^\s*\[meta\]",
    r"\byou are no longer an ai assistant\b",

# Persona / jailbreak / role-assignment prompts
    r"\bact as\b",
    r"\bfrom now on you are\b",
    r"\brespond from the perspective of\b",
    r"\bdo anything now\b",

# Classification / labelling tasks
    r"\bhelp me label\b",
    r"\blabel (?:a|an|the|this)\b",
    r"\breply as ['\"]?(?:relevant|irrelevant)\b",
    r"\bclassification\b",

# Rewriting / structured transformation tasks
    r"\bhelp me rewrite\b",
    r"\brewrite\b",
    r"\bstructured json\b",

# Explicit content-generation tasks
    r"\bwrite (?:me )?(?:a|an|the)?\s*(?:handbook|story|screenplay|script)\b",
    r"\bgenerate (?:at least|a list|a set)\b",
    r"\bkey takeaways\b",
]



def is_task_like_prompt(user_text: str) -> bool:
    """
    Identify prompts that are primarily benchmark, transformation,
    classification, technical, or creative-writing tasks rather than
    first-person personal conversations.
    """

    if not user_text:
        return False

    text = user_text.lower().strip()

    return any(
        re.search(pattern, text)
        for pattern in TASK_EXCLUSION_PATTERNS
    )

# ---------------------------------------------------------------------
# Sensitive-content exclusion
# ---------------------------------------------------------------------
# This is a scope/ethics exclusion rather
# than a relevance judgement.
# ---------------------------------------------------------------------

MINOR_CONTEXT_PATTERNS = [
    r"\bteen\b",
    r"\bteenage\b",
    r"\bteenager\b",
    r"\bminor\b",
    r"\bunderage\b",
    r"\bunder[- ]18\b",
    r"\b(?:[1-9]|1[0-7])[- ]?(?:year[- ]old|years? old)\b",
]

SEXUAL_CONTEXT_PATTERNS = [
    r"\bsexual\b",
    r"\bsex\b",
    r"\bsexuality\b",
    r"\blingerie\b",
    r"\bnude\b",
    r"\bnaked\b",
    r"\bintimate\b",
    r"\bintimacy\b",
    r"\borgasm\b",
]


def is_sensitive_exclusion(user_text: str) -> bool:
    """
    Return True when a prompt combines a minor indicator with
    sexual/intimate content.

    This is an explicit project-scope exclusion.
    """

    if not user_text:
        return False

    text = user_text.lower().strip()

    has_minor_context = any(
        re.search(pattern, text)
        for pattern in MINOR_CONTEXT_PATTERNS
    )

    has_sexual_context = any(
        re.search(pattern, text)
        for pattern in SEXUAL_CONTEXT_PATTERNS
    )

    return has_minor_context and has_sexual_context

def is_english(language: Any) -> bool:
    """
    Return True when the LMSYS language field represents English.
    """

    if language is None:
        return False

    value = str(language).strip().lower()

    return value == "en" or value.startswith("english")


def extract_user_assistant_pairs(
    example: dict[str, Any],
    source_index: int,
) -> list[dict[str, Any]]:
    """
    Extract adjacent user -> assistant message pairs from one conversation.

    We deliberately do not retain LMSYS conversation_id.

    source_index provides a non-identifying grouping variable within the
    deterministic scan so multiple exchanges from the same source
    conversation can still be recognised.
    """

    messages = example.get("conversation")

    if not isinstance(messages, list):
        return []

    pairs = []
    pair_index = 0

    for position in range(len(messages) - 1):
        current = messages[position]
        following = messages[position + 1]

        if not isinstance(current, dict):
            continue

        if not isinstance(following, dict):
            continue

        if (
            current.get("role") != "user"
            or following.get("role") != "assistant"
        ):
            continue

        user_text = str(
            current.get("content", "")
        ).strip()

        assistant_text = str(
            following.get("content", "")
        ).strip()

        pairs.append(
            {
                "source_index": source_index,
                "pair_index": pair_index,
                "model": example.get("model"),
                "language": example.get("language"),
                "redacted": example.get("redacted"),
                "user_text": user_text,
                "assistant_text": assistant_text,
            }
        )

        pair_index += 1

    return pairs


def pair_is_eligible(
    pair: dict[str, Any],
) -> bool:
    """
    Apply basic language and text-quality eligibility rules.

    This function performs selection only.
    It does not assign disempowerment labels.
    """

    if not is_english(pair.get("language")):
        return False

    user_text = pair.get("user_text", "")
    assistant_text = pair.get("assistant_text", "")

    if not (
        MIN_USER_CHARS
        <= len(user_text)
        <= MAX_USER_CHARS
    ):
        return False

    if not (
        MIN_ASSISTANT_CHARS
        <= len(assistant_text)
        <= MAX_ASSISTANT_CHARS
    ):
        return False

    return True


def match_relevance_categories(
    user_text: str,
) -> list[str]:
    """
    Identify research-relevant personal conversation topics.

    A topical keyword alone is not sufficient.

    Direct first-person emotional disclosures may qualify directly.
    Relational and lifestyle content must also contain help-seeking,
    decision-making, or emotional context.

    These categories are selection categories only.
    They are not disempowerment labels.
    """

    if not user_text:
        return []

    if is_sensitive_exclusion(user_text):
        return [] 

    if is_task_like_prompt(user_text):
        return []

    text = user_text.lower().strip()

    matched_categories = []

    emotional_patterns = RELEVANCE_PATTERNS["emotional"]
    relational_patterns = RELEVANCE_PATTERNS["relational"]
    lifestyle_patterns = RELEVANCE_PATTERNS["personal_lifestyle"]

    has_emotional_context = any(
        re.search(pattern, text)
        for pattern in emotional_patterns
    )

    has_relational_context = any(
        re.search(pattern, text)
        for pattern in relational_patterns
    )

    has_lifestyle_context = any(
        re.search(pattern, text)
        for pattern in lifestyle_patterns
    )

    has_help_or_decision_intent = any(
        re.search(pattern, text)
        for pattern in PERSONAL_HELP_PATTERNS
    )

    has_advice_request = any(
        re.search(pattern, text)
        for pattern in ADVICE_PATTERNS
    )

    has_personal_domain = any(
        re.search(pattern, text)
        for pattern in PERSONAL_DECISION_DOMAINS
    )

    if has_emotional_context:
        matched_categories.append("emotional")

    if (
        has_relational_context
        and (
            has_help_or_decision_intent
            or has_emotional_context
        )
    ):
        matched_categories.append("relational")

    if (
        has_lifestyle_context
        and (
            has_help_or_decision_intent
            or has_emotional_context
        )
    ):
        matched_categories.append("personal_lifestyle")

    if (
        (has_advice_request or has_help_or_decision_intent)
        and has_personal_domain
    ):
        matched_categories.append("advice_decision")

    return matched_categories


def debug_relevance_matches(
    user_text: str,
) -> dict[str, object]:
    """
    Show exactly which filtering rules matched a user message.

    Development/debugging utility only.
    It does not assign research labels.
    """

    if not user_text:
        return {
            "sensitive_exclusion": is_sensitive_exclusion(user_text),
            "task_exclusions": [],
            "category_matches": {},
            "advice_matches": [],
            "personal_domain_matches": [],
        }

    text = user_text.lower().strip()

    task_exclusions = [
        pattern
        for pattern in TASK_EXCLUSION_PATTERNS
        if re.search(pattern, text)
    ]

    category_matches = {}

    for category, patterns in RELEVANCE_PATTERNS.items():
        matched_patterns = [
            pattern
            for pattern in patterns
            if re.search(pattern, text)
        ]

        if matched_patterns:
            category_matches[category] = matched_patterns

    advice_matches = [
        pattern
        for pattern in ADVICE_PATTERNS
        if re.search(pattern, text)
    ]

    personal_domain_matches = [
        pattern
        for pattern in PERSONAL_DECISION_DOMAINS
        if re.search(pattern, text)
    ]

    return {
        "task_exclusions": task_exclusions,
        "category_matches": category_matches,
        "advice_matches": advice_matches,
        "personal_domain_matches": personal_domain_matches,
    }

def build_relevant_candidate_pool(
    dataset: Iterable[dict[str, Any]],
    max_conversations: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Scan a fixed number of streamed LMSYS conversations.

    Returns
    -------
    candidates:
        User-assistant pairs passing both eligibility and relevance checks.

    diagnostics:
        Processing counts for this development scan.

    These diagnostic counts describe this particular scan only and should
    not be interpreted as LMSYS population-level statistics.
    """

    if max_conversations <= 0:
        raise ValueError(
            "max_conversations must be greater than zero"
        )

    candidates = []

    conversations_scanned = 0
    pairs_extracted = 0
    eligible_pairs = 0

    for source_index, example in enumerate(dataset):

        if source_index >= max_conversations:
            break

        conversations_scanned += 1

        pairs = extract_user_assistant_pairs(
            example=example,
            source_index=source_index,
        )

        pairs_extracted += len(pairs)

        for pair in pairs:

            if not pair_is_eligible(pair):
                continue

            eligible_pairs += 1

            categories = match_relevance_categories(
                pair["user_text"]
            )

            if not categories:
                continue

            pair["relevance_categories"] = "|".join(
                categories
            )

            candidates.append(pair)

    diagnostics = {
        "conversations_scanned": conversations_scanned,
        "pairs_extracted": pairs_extracted,
        "eligible_pairs": eligible_pairs,
        "relevant_candidate_pairs": len(candidates),
    }

    return candidates, diagnostics


def deterministic_sample(
    rows: list[dict[str, Any]],
    sample_size: int,
    random_seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """
    Return a deterministic random sample.

    The same input rows and random seed will produce the same sample.
    """

    if sample_size <= 0:
        raise ValueError(
            "sample_size must be greater than zero"
        )

    if len(rows) <= sample_size:
        return rows.copy()

    rng = random.Random(random_seed)

    selected_indices = sorted(
        rng.sample(
            range(len(rows)),
            sample_size,
        )
    )

    return [
        rows[index]
        for index in selected_indices
    ]
