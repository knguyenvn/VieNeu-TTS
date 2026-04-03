from __future__ import annotations

from dataclasses import dataclass
import re

from .tech_text import rewrite_mixed_tech_text

DEFAULT_ACRONYM_MODE = "natural"
DEFAULT_NARRATION_STRENGTH = "balanced"

_VALID_ACRONYM_MODES = {"clear", "natural", "vi"}
_VALID_NARRATION_STRENGTHS = {"balanced", "expressive", "cinematic"}

_NARRATION_LEAD_INS = [
    "Đầu tiên",
    "Tiếp theo",
    "Cuối cùng",
    "Lưu ý",
    "Nói cách khác",
    "Ví dụ",
    "Ví dụ như",
    "Tuy nhiên",
    "Vì vậy",
    "Sau đó",
    "Trong khi đó",
    "Hiện tại",
    "Ở đây",
]
_LEAD_IN_RE = re.compile(
    rf"(?:(?<=^)|(?<=[\.\!\?\n]\s))({'|'.join(re.escape(item) for item in _NARRATION_LEAD_INS)})(?!\s*[,.:;!?])(?=\s+\S)"
)


@dataclass(frozen=True)
class TextAdaptationOptions:
    acronym_mode: str = DEFAULT_ACRONYM_MODE
    narration_mode: bool = False
    narration_strength: str = DEFAULT_NARRATION_STRENGTH


@dataclass(frozen=True)
class NarrationProfile:
    max_chars_scale: float
    continuation_pause: float
    sentence_pause: float
    strong_pause: float
    paragraph_pause: float
    crossfade: float
    turbo_pause_scale: float


_NARRATION_PROFILES = {
    "balanced": NarrationProfile(
        max_chars_scale=0.72,
        continuation_pause=0.03,
        sentence_pause=0.22,
        strong_pause=0.30,
        paragraph_pause=0.42,
        crossfade=0.015,
        turbo_pause_scale=1.35,
    ),
    "expressive": NarrationProfile(
        max_chars_scale=0.58,
        continuation_pause=0.05,
        sentence_pause=0.28,
        strong_pause=0.38,
        paragraph_pause=0.60,
        crossfade=0.02,
        turbo_pause_scale=1.65,
    ),
    "cinematic": NarrationProfile(
        max_chars_scale=0.48,
        continuation_pause=0.07,
        sentence_pause=0.34,
        strong_pause=0.46,
        paragraph_pause=0.78,
        crossfade=0.025,
        turbo_pause_scale=1.95,
    ),
}


def resolve_text_adaptation_options(
    acronym_mode: str | None = None,
    narration_mode: bool = False,
    narration_strength: str | None = None,
) -> TextAdaptationOptions:
    resolved_acronym_mode = (acronym_mode or DEFAULT_ACRONYM_MODE).lower()
    if resolved_acronym_mode not in _VALID_ACRONYM_MODES:
        resolved_acronym_mode = DEFAULT_ACRONYM_MODE

    resolved_narration_strength = (
        narration_strength or DEFAULT_NARRATION_STRENGTH
    ).lower()
    if resolved_narration_strength not in _VALID_NARRATION_STRENGTHS:
        resolved_narration_strength = DEFAULT_NARRATION_STRENGTH

    return TextAdaptationOptions(
        acronym_mode=resolved_acronym_mode,
        narration_mode=bool(narration_mode),
        narration_strength=resolved_narration_strength,
    )


def get_narration_profile(
    options: TextAdaptationOptions,
) -> NarrationProfile | None:
    if not options.narration_mode:
        return None
    return _NARRATION_PROFILES[options.narration_strength]


def get_effective_max_chars(
    max_chars: int,
    options: TextAdaptationOptions,
) -> int:
    profile = get_narration_profile(options)
    if profile is None:
        return max_chars
    return max(64, int(round(max_chars * profile.max_chars_scale)))


def adapt_text_for_tts(
    text: str,
    acronym_mode: str | None = None,
    narration_mode: bool = False,
    narration_strength: str | None = None,
) -> str:
    options = resolve_text_adaptation_options(
        acronym_mode=acronym_mode,
        narration_mode=narration_mode,
        narration_strength=narration_strength,
    )
    adapted = rewrite_mixed_tech_text(text, acronym_mode=options.acronym_mode)
    if options.narration_mode:
        adapted = _apply_narration_punctuation(adapted)
    return adapted


def _apply_narration_punctuation(text: str) -> str:
    def add_comma(match: re.Match[str]) -> str:
        return f"{match.group(1)},"

    return _LEAD_IN_RE.sub(add_comma, text)
