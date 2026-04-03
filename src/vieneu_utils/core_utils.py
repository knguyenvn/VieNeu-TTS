import re
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

# ─── Regex ───────────────────────────────────────────────────────────────────

RE_NEWLINE          = re.compile(r'[\r\n]+')  # dùng chung cho cả v1 và v2
RE_SENTENCE_FINDALL = re.compile(r'[^.!?]+[.!?]*|[.!?]+')

# v1 only
RE_SENTENCE_END = re.compile(r'(?<=[\.\!\?\…])\s+')
RE_MINOR_PUNCT  = re.compile(r'(?<=[\,\;\:\-\–\—])\s+')

# v2 noise cleanup
_NOISE_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'([.!?])[.,;:]+'), r'\1'),
    (re.compile(r'[.,;:]+([.!?])'), r'\1'),
    (re.compile(r'\s+[,;]\s+'),     ' '),
    (re.compile(r' {2,}'),          ' '),
]
_MULTI_PUNCT = re.compile(r'([.!?])\s*[.!?]+')

# ─── Data class ──────────────────────────────────────────────────────────────

@dataclass
class PhoneChunk:
    text: str
    is_sentence_end: bool  # True = kết thúc câu thật | False = cắt nhân tạo
    is_paragraph_end: bool = False


@dataclass
class TextChunk:
    text: str
    pause_after: float = 0.0

# ─── Audio utils ─────────────────────────────────────────────────────────────

def join_audio_chunks(
    chunks: List[np.ndarray],
    sr: int,
    silence_p: float = 0.0,
    crossfade_p: float = 0.0,
    pause_plan: Optional[List[float]] = None,
) -> np.ndarray:
    if not chunks:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]

    silence_samples   = int(sr * silence_p)
    crossfade_samples = int(sr * crossfade_p)
    final_wav = chunks[0]

    for i in range(1, len(chunks)):
        next_chunk = chunks[i]
        pause_samples = None
        if pause_plan is not None and i - 1 < len(pause_plan):
            pause_samples = int(sr * max(0.0, pause_plan[i - 1]))

        if pause_samples is not None and pause_samples > 0:
            silence = np.zeros(pause_samples, dtype=np.float32)
            final_wav = np.concatenate([final_wav, silence, next_chunk])
        elif silence_samples > 0:
            silence = np.zeros(silence_samples, dtype=np.float32)
            final_wav = np.concatenate([final_wav, silence, next_chunk])
        elif crossfade_samples > 0:
            overlap = min(len(final_wav), len(next_chunk), crossfade_samples)
            if overlap > 0:
                fade_out  = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
                fade_in   = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
                blended   = final_wav[-overlap:] * fade_out + next_chunk[:overlap] * fade_in
                final_wav = np.concatenate([final_wav[:-overlap], blended, next_chunk[overlap:]])
            else:
                final_wav = np.concatenate([final_wav, next_chunk])
        else:
            final_wav = np.concatenate([final_wav, next_chunk])

    return final_wav

# ─── v1: split raw text ──────────────────────────────────────────────────────

def split_text_into_chunks(text: str, max_chars: int = 256) -> List[str]:
    """Split raw text (chưa phonemize) thành chunks <= max_chars."""
    return [chunk.text for chunk in split_text_into_chunks_with_pauses(text, max_chars=max_chars)]


def split_text_into_chunks_with_pauses(
    text: str,
    max_chars: int = 256,
    continuation_pause: float = 0.0,
    sentence_pause: float = 0.15,
    strong_pause: float = 0.2,
    paragraph_pause: float = 0.25,
) -> List[TextChunk]:
    """Split raw text into chunks and attach a per-chunk pause plan."""
    if not text:
        return []

    paragraphs   = RE_NEWLINE.split(text.strip())
    final_chunks: List[TextChunk] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        sentences = RE_SENTENCE_END.split(para)
        paragraph_chunks: List[str] = []
        buffer = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > max_chars:
                if buffer:
                    paragraph_chunks.append(buffer)
                    buffer = ""

                sub_parts = RE_MINOR_PUNCT.split(sentence)
                for part in sub_parts:
                    part = part.strip()
                    if not part:
                        continue
                    if len(buffer) + 1 + len(part) <= max_chars:
                        buffer = (buffer + ' ' + part) if buffer else part
                    else:
                        if buffer:
                            paragraph_chunks.append(buffer)
                        buffer = part
                        if len(buffer) > max_chars:
                            words, current = buffer.split(), ""
                            for word in words:
                                if current and len(current) + 1 + len(word) > max_chars:
                                    paragraph_chunks.append(current)
                                    current = word
                                else:
                                    current = (current + ' ' + word) if current else word
                            buffer = current
            else:
                if buffer and len(buffer) + 1 + len(sentence) > max_chars:
                    paragraph_chunks.append(buffer)
                    buffer = sentence
                else:
                    buffer = (buffer + ' ' + sentence) if buffer else sentence

        if buffer:
            paragraph_chunks.append(buffer)

        if not paragraph_chunks:
            continue

        last_idx = len(paragraph_chunks) - 1
        for idx, chunk in enumerate(paragraph_chunks):
            chunk = chunk.strip()
            if not chunk:
                continue

            if idx == last_idx:
                pause_after = _text_chunk_pause(
                    chunk,
                    continuation_pause=continuation_pause,
                    sentence_pause=sentence_pause,
                    strong_pause=strong_pause,
                    paragraph_pause=paragraph_pause,
                    is_paragraph_end=True,
                )
            else:
                pause_after = _text_chunk_pause(
                    chunk,
                    continuation_pause=continuation_pause,
                    sentence_pause=sentence_pause,
                    strong_pause=strong_pause,
                    paragraph_pause=paragraph_pause,
                    is_paragraph_end=False,
                )
            final_chunks.append(TextChunk(text=chunk, pause_after=pause_after))

    return final_chunks


def _text_chunk_pause(
    chunk: str,
    continuation_pause: float,
    sentence_pause: float,
    strong_pause: float,
    paragraph_pause: float,
    is_paragraph_end: bool,
) -> float:
    stripped = chunk.strip()
    if not stripped:
        return 0.0

    tail = stripped[-1]
    if tail in "!?":
        base_pause = strong_pause
    elif tail in ".…":
        base_pause = sentence_pause
    elif tail in ",;:":
        base_pause = continuation_pause
    else:
        base_pause = continuation_pause

    if is_paragraph_end:
        return max(base_pause, paragraph_pause)
    return base_pause

# ─── v2 helpers ──────────────────────────────────────────────────────────────

def _pick_strongest(m: re.Match) -> str:
    s = m.group(0)
    return '!' if '!' in s else '?' if '?' in s else '.'


def _clean_phoneme_noise(text: str) -> str:
    for pattern, repl in _NOISE_RULES:
        text = pattern.sub(repl, text)
    return _MULTI_PUNCT.sub(_pick_strongest, text).strip()


def _find_best_split(text: str, max_size: int) -> Tuple[int, bool]:
    mid = max_size // 2
    best_comma_pos, best_comma_dist = -1, max_size
    best_space_pos, best_space_dist = -1, max_size

    for i in range(min(max_size, len(text))):
        ch = text[i]
        if ch == ',':
            d = abs(i - mid)
            if d < best_comma_dist:
                best_comma_dist, best_comma_pos = d, i
        elif ch == ' ':
            d = abs(i - mid)
            if d < best_space_dist:
                best_space_dist, best_space_pos = d, i

    if best_comma_pos != -1:
        return best_comma_pos, True
    if best_space_pos != -1:
        return best_space_pos, False
    return -1, False


def _smart_split_body(text: str, max_chunk_size: int) -> List[str]:
    result: List[str] = []
    stack = [text.strip()]

    while stack:
        seg = stack.pop()
        if not seg:
            continue
        if len(seg) <= max_chunk_size:
            result.append(seg)
            continue

        pos, _ = _find_best_split(seg, max_chunk_size)
        if pos != -1:
            left  = seg[:pos].rstrip()
            right = seg[pos + 1:].lstrip()
        else:
            cut = max_chunk_size
            while cut > 0 and seg[cut - 1] != ' ':
                cut -= 1
            if cut == 0:
                cut = max_chunk_size
            left  = seg[:cut].rstrip()
            right = seg[cut:].lstrip()

        if right:
            stack.append(right)
        if left:
            stack.append(left)

    return result


def _split_sentence(sent: str, max_chunk_size: int) -> List[PhoneChunk]:
    sent = sent.strip()
    if not sent:
        return []

    if sent[-1] in '.!?':
        body, punct = sent[:-1].rstrip(), sent[-1]
    else:
        body, punct = sent, '.'

    if not body:
        return []

    if len(sent) <= max_chunk_size:
        return [PhoneChunk(text=body + punct, is_sentence_end=True)]

    sub_chunks = _smart_split_body(body, max_chunk_size)
    if not sub_chunks:
        return [PhoneChunk(text=punct, is_sentence_end=True)]

    last_idx = len(sub_chunks) - 1
    return [
        PhoneChunk(
            text=chunk + (punct if i == last_idx else '.'),
            is_sentence_end=(i == last_idx),
            is_paragraph_end=False,
        )
        for i, chunk in enumerate(sub_chunks)
        if chunk
    ]

# ─── v2: split phoneme string ────────────────────────────────────────────────

def split_into_chunks_v2(
    full_phones: str,
    max_chunk_size: int = 256,
    min_chunk_size: int = 10,
) -> List[PhoneChunk]:
    """
    Phân đoạn chuỗi phoneme thành các PhoneChunk.
      is_sentence_end=True  → kết thúc câu thật → cần silence
      is_sentence_end=False → cắt nhân tạo → không cần silence
    """
    if not full_phones:
        return []

    full_phones = _clean_phoneme_noise(full_phones)

    raw_parts: List[PhoneChunk] = []
    for para in RE_NEWLINE.split(full_phones):
        para = para.strip()
        if not para:
            continue
        para_start = len(raw_parts)
        for sent in RE_SENTENCE_FINDALL.findall(para):
            sent = sent.strip()
            if sent:
                raw_parts.extend(_split_sentence(sent, max_chunk_size))
        if len(raw_parts) > para_start:
            raw_parts[-1].is_paragraph_end = True

    if not raw_parts:
        return []

    merged: List[PhoneChunk] = []
    i, n = 0, len(raw_parts)
    while i < n:
        cur = raw_parts[i]
        while len(cur.text) < min_chunk_size and i + 1 < n:
            nxt       = raw_parts[i + 1]
            candidate = cur.text.rstrip('.!?').rstrip() + ' ' + nxt.text
            if len(candidate) <= max_chunk_size:
                cur = PhoneChunk(
                    text=candidate,
                    is_sentence_end=nxt.is_sentence_end,
                    is_paragraph_end=nxt.is_paragraph_end,
                )
                i += 1
            else:
                break
        merged.append(cur)
        i += 1

    if len(merged) >= 2 and len(merged[-1].text) < min_chunk_size:
        last      = merged.pop()
        candidate = merged[-1].text.rstrip('.!?').rstrip() + ' ' + last.text
        if len(candidate) <= max_chunk_size:
            merged[-1] = PhoneChunk(
                text=candidate,
                is_sentence_end=last.is_sentence_end,
                is_paragraph_end=last.is_paragraph_end,
            )
        else:
            merged.append(last)

    return merged


def get_silence_duration_v2(
    chunk: PhoneChunk,
    pause_scale: float = 1.0,
    continuation_pause: float = 0.0,
    paragraph_pause: float = 0.0,
) -> float:
    """
    Silence sau chunk (giây).
      is_sentence_end=False → 0.0s
      kết thúc '!'/'?' → 0.4s
      kết thúc '.' → 0.3s
    """
    if not chunk.is_sentence_end:
        base_pause = continuation_pause
    else:
        base_pause = 0.4 if chunk.text.strip()[-1] in '!?' else 0.3

    scaled_pause = base_pause * pause_scale
    if chunk.is_paragraph_end:
        return max(scaled_pause, paragraph_pause)
    return scaled_pause

# ─── Misc ────────────────────────────────────────────────────────────────────

def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
