import numpy as np

from vieneu_utils.core_utils import PhoneChunk, get_silence_duration_v2, join_audio_chunks
from vieneu_utils.text_adaptation import (
    adapt_text_for_tts,
    get_effective_max_chars,
    get_narration_profile,
    resolve_text_adaptation_options,
)


def test_adapt_text_supports_multiple_acronym_styles():
    text = "API của GPU4AI"

    assert adapt_text_for_tts(text, acronym_mode="clear") == "API của GPU for AI"
    assert adapt_text_for_tts(text, acronym_mode="natural") == "<en>API</en> của <en>GPU</en> for <en>AI</en>"
    assert adapt_text_for_tts(text, acronym_mode="vi") == "ây pi ai của gi pi diu for ây ai"


def test_narration_mode_inserts_lead_in_comma():
    text = "Đầu tiên bạn mở API của OpenAI."

    assert adapt_text_for_tts(text, acronym_mode="natural", narration_mode=True) == (
        "Đầu tiên, bạn mở <en>API</en> của OpenAI."
    )


def test_narration_profile_reduces_chunk_size_and_scales_pauses():
    options = resolve_text_adaptation_options(
        acronym_mode="natural",
        narration_mode=True,
        narration_strength="cinematic",
    )
    profile = get_narration_profile(options)

    assert profile is not None
    assert get_effective_max_chars(256, options) < 256

    sentence_chunk = PhoneChunk(text="Xin chào.", is_sentence_end=True, is_paragraph_end=False)
    paragraph_chunk = PhoneChunk(text="Kết thúc đoạn.", is_sentence_end=True, is_paragraph_end=True)

    assert get_silence_duration_v2(
        sentence_chunk,
        pause_scale=profile.turbo_pause_scale,
        continuation_pause=profile.continuation_pause,
        paragraph_pause=profile.paragraph_pause,
    ) > 0.3
    assert get_silence_duration_v2(
        paragraph_chunk,
        pause_scale=profile.turbo_pause_scale,
        continuation_pause=profile.continuation_pause,
        paragraph_pause=profile.paragraph_pause,
    ) >= profile.paragraph_pause


def test_join_audio_chunks_uses_pause_plan_when_provided():
    a = np.ones(4, dtype=np.float32)
    b = np.ones(4, dtype=np.float32) * 2

    joined = join_audio_chunks([a, b], sr=4, silence_p=0.0, pause_plan=[0.5])

    assert joined.tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0]
