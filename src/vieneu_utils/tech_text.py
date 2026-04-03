from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any
from urllib.parse import parse_qsl, urlsplit

# Safe acronyms can be uppercased globally without colliding with common
# Vietnamese words.
_SAFE_ACRONYMS = {
    "acl",
    "adb",
    "api",
    "asr",
    "aws",
    "cuda",
    "cdn",
    "ci",
    "cli",
    "cms",
    "cpu",
    "csv",
    "db",
    "dns",
    "dnssec",
    "eks",
    "faq",
    "gcp",
    "gke",
    "grpc",
    "id",
    "imap",
    "ip",
    "jwt",
    "kyc",
    "mcp",
    "mfa",
    "mps",
    "orm",
    "ram",
    "rom",
    "rtx",
    "sftp",
    "sms",
    "smtp",
    "sso",
    "ssh",
    "svg",
    "tcp",
    "tpu",
    "udp",
    "uuid",
    "vps",
    "cpu",
    "crm",
    "erp",
    "etl",
    "gpt",
    "gpu",
    "hrm",
    "html",
    "http",
    "https",
    "ide",
    "iot",
    "json",
    "kpi",
    "llm",
    "ml",
    "nlp",
    "ocr",
    "okr",
    "otp",
    "pdf",
    "qa",
    "qc",
    "rag",
    "sdk",
    "sem",
    "seo",
    "sql",
    "ssl",
    "stt",
    "tls",
    "tts",
    "uri",
    "url",
    "vpn",
    "xml",
}

# Context acronyms are only rewritten when they already appear inside a
# tech-shaped token such as gpu4ai, ai4vn, camelCase, or slug-like text.
_CONTEXT_ACRONYMS = {
    "ai",
    "ios",
    "ui",
    "ux",
    "vn",
}

_EXACT_REWRITES = {
    "2fa": "two F A",
    "ai4vn": "AI for VN",
    "airdrop": "AirDrop",
    "android": "Android",
    "anthropic": "Anthropic",
    "airpods": "AirPods",
    "apache": "Apache",
    "azure": "Azure",
    "b2b": "B two B",
    "b2c": "B two C",
    "botfather": "BotFather",
    "chatbot": "chatbot",
    "chatgpt": "ChatGPT",
    "claude": "Claude",
    "chrome": "Chrome",
    "cloudflare": "Cloudflare",
    "copilot": "Copilot",
    "coder": "Coder",
    "d2c": "D two C",
    "deepseek": "DeepSeek",
    "debian": "Debian",
    "discord": "Discord",
    "docker": "Docker",
    "dockercompose": "Docker Compose",
    "dropbox": "Dropbox",
    "e2e": "E two E",
    "ec2": "E C two",
    "edge": "Edge",
    "elasticsearch": "Elasticsearch",
    "facebook": "Facebook",
    "fastapi": "FastAPI",
    "ffmpeg": "FFmpeg",
    "firebase": "Firebase",
    "firefox": "Firefox",
    "flask": "Flask",
    "frequency_penalty": "frequency penalty",
    "gemma": "Gemma",
    "gemini": "Gemini",
    "gradio": "Gradio",
    "githubactions": "GitHub Actions",
    "github": "GitHub",
    "gitignore": "Git Ignore",
    "gitlab": "GitLab",
    "gmail": "Gmail",
    "google": "Google",
    "grafana": "Grafana",
    "graphql": "GraphQL",
    "grok": "Grok",
    "gpt-4": "GPT four",
    "gpt-4o": "GPT four o",
    "gpt-4o-mini": "GPT four o mini",
    "gpt4": "GPT four",
    "gpt4.1": "GPT four point one",
    "gpt4o": "GPT four o",
    "gpt4o-mini": "GPT four o mini",
    "haiku": "Haiku",
    "iphone16": "iPhone sixteen",
    "gpu4ai": "GPU for AI",
    "java": "Java",
    "javascript": "JavaScript",
    "kafka": "Kafka",
    "kibana": "Kibana",
    "iphone": "iPhone",
    "ipad": "iPad",
    "ipados": "iPadOS",
    "k8s": "Kubernetes",
    "kubernetes": "Kubernetes",
    "lambda": "Lambda",
    "linkedin": "LinkedIn",
    "linux": "Linux",
    "llama": "Llama",
    "langchain": "LangChain",
    "langgraph": "LangGraph",
    "localhost": "localhost",
    "mariadb": "MariaDB",
    "macbook": "MacBook",
    "macbookair": "MacBook Air",
    "macbookpro": "MacBook Pro",
    "macos": "macOS",
    "messenger": "Messenger",
    "mistral": "Mistral",
    "momo": "MoMo",
    "mongodb": "MongoDB",
    "mysql": "MySQL",
    "n8n": "N eight N",
    "netlify": "Netlify",
    "next.js": "Next JS",
    "nextjs": "Next JS",
    "nestjs": "Nest JS",
    "nginx": "Nginx",
    "notion": "Notion",
    "node.js": "Node JS",
    "nodejs": "Node JS",
    "npm": "N P M",
    "oauth": "OAuth",
    "oauth2": "OAuth two",
    "o1": "o one",
    "o1-mini": "o one mini",
    "o1-preview": "o one Preview",
    "o3": "o three",
    "o3-mini": "o three mini",
    "o3-pro": "o three Pro",
    "o4-mini": "o four mini",
    "onedrive": "OneDrive",
    "openid": "OpenID",
    "openrouter": "OpenRouter",
    "openai": "OpenAI",
    "openwebui": "Open WebUI",
    "opus": "Opus",
    "outlook": "Outlook",
    "paypal": "PayPal",
    "perplexity": "Perplexity",
    "phi": "Phi",
    "pip": "pip",
    "pipx": "pip x",
    "pnpm": "P N P M",
    "postgres": "Postgres",
    "postgresql": "PostgreSQL",
    "powerpoint": "PowerPoint",
    "presence_penalty": "presence penalty",
    "prometheus": "Prometheus",
    "pro": "Pro",
    "python": "Python",
    "qwen": "Qwen",
    "rabbitmq": "RabbitMQ",
    "react.js": "React JS",
    "reactjs": "React JS",
    "reactnative": "React Native",
    "redis": "Redis",
    "repetition_penalty": "repetition penalty",
    "restapi": "REST API",
    "r2": "R two",
    "s3": "S three",
    "safari": "Safari",
    "shopee": "Shopee",
    "shopeepay": "ShopeePay",
    "skype": "Skype",
    "slack": "Slack",
    "sonnet": "Sonnet",
    "supabase": "Supabase",
    "tailwindcss": "Tailwind CSS",
    "teams": "Teams",
    "telegram": "Telegram",
    "toml": "T O M L",
    "tiktok": "TikTok",
    "top_k": "top <en>k</en>",
    "top_p": "top <en>p</en>",
    "typescript": "TypeScript",
    "ubuntu": "Ubuntu",
    "uv": "U V",
    "uvicorn": "Uvicorn",
    "vercel": "Vercel",
    "vite": "Vite",
    "visionos": "visionOS",
    "vscode": "VS Code",
    "vue.js": "Vue JS",
    "watchos": "watchOS",
    "webrtc": "WebRTC",
    "webhook": "webhook",
    "websocket": "websocket",
    "whatsapp": "WhatsApp",
    "wi-fi": "Wi-Fi",
    "wifi": "Wi-Fi",
    "windows": "Windows",
    "word": "Word",
    "www": "W W W",
    "wsl": "W S L",
    "xlsx": "X L S X",
    "youtube": "YouTube",
    "zalo": "Zalo",
    "zalopay": "ZaloPay",
    "zoom": "Zoom",
    "2d": "two D",
    "2g": "two G",
    "3d": "three D",
    "3g": "three G",
    "4g": "four G",
    "5g": "five G",
}

_ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/+:-]*")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_ALNUM_BOUNDARY_RE = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")
_VERSION_DOT_RE = re.compile(r"(?<=\d)\.(?=\d)")
_SEPARATOR_RE = re.compile(r"[-_/+:]+")
_SPACE_RE = re.compile(r"\s+")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_URL_RE = re.compile(r"\b(?:https?://|www\.)[A-Za-z0-9./?&%#=_:+-]*[A-Za-z0-9/#=_+-]")
_SPACED_PLUS_RE = re.compile(r"\s+\+\s+")
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_HOST_PORT_RE = re.compile(r"\b(?:localhost|[A-Za-z][A-Za-z0-9.-]*|\d{1,3}(?:\.\d{1,3}){3}):\d{2,5}\b")
_WORD_VERSION_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9.+-]*)\s+(\d+(?:\.\d+)?)\b")
_CONFIG_ASSIGNMENT_RE = re.compile(
    r"(?<![?&])\b(top_p|top_k|max_tokens|temperature|presence_penalty|frequency_penalty|repetition_penalty|memory_util|server_port|server_name|base_url|api_key|model_name)\b\s*[:=]\s*([A-Za-z0-9._:/?=%-]+)"
)
_CLI_FLAG_RE = re.compile(r"(?<!\w)--[a-z0-9][a-z0-9-]*(?:=[A-Za-z0-9._/-]+)?")
_WINDOWS_PATH_RE = re.compile(r"\b[A-Za-z]:[\\/](?:[^\\/\s]+[\\/])*[^\\/\s]+\b")
_PATH_RE = re.compile(r"(?<![:/])\b(?:[A-Za-z0-9._-]+/)+[A-Za-z0-9._-]+\b")
_DOTFILE_RE = re.compile(r"(?<![A-Za-z0-9])\.[A-Za-z0-9][A-Za-z0-9._-]*")
_ENV_VAR_RE = re.compile(r"\b[A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+\b")
_SNAKE_IDENTIFIER_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_DOMAIN_TLDS = {"ai", "app", "com", "dev", "io", "net", "org", "vn"}
_FILE_EXTENSIONS = {
    "cfg",
    "conf",
    "csv",
    "css",
    "docx",
    "env",
    "html",
    "ini",
    "js",
    "jsx",
    "json",
    "lock",
    "log",
    "md",
    "pdf",
    "py",
    "sh",
    "sql",
    "ts",
    "tsx",
    "txt",
    "toml",
    "xlsx",
    "xml",
    "yaml",
    "yml",
}
_EMAIL_LOCAL_SEPARATORS = {
    ".": "chấm",
    "+": "cộng",
    "-": "gạch ngang",
    "_": "gạch dưới",
}
_DECIMAL_VALUE_RE = re.compile(r"^\d+\.\d+$")
_DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}
_TEEN_WORDS = {
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}
_TENS_WORDS = {
    2: "twenty",
    3: "thirty",
    4: "forty",
    5: "fifty",
    6: "sixty",
    7: "seventy",
    8: "eighty",
    9: "ninety",
}


def _collapse_spaces(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()


def _spell_letters(token: str) -> str:
    return " ".join(char.upper() for char in token if char.isalnum())


def _number_to_english(token: str) -> str:
    if not token.isdigit():
        return token

    value = int(token)
    if value < 10:
        return _DIGIT_WORDS[token]
    if value < 20:
        return _TEEN_WORDS[value]
    if value < 100:
        tens, ones = divmod(value, 10)
        tens_word = _TENS_WORDS[tens]
        if ones == 0:
            return tens_word
        return f"{tens_word} {_DIGIT_WORDS[str(ones)]}"
    return " ".join(_DIGIT_WORDS[digit] for digit in token)


def _load_override_terms() -> dict[str, Any]:
    path = os.getenv("VIENEU_TECH_TERMS_FILE")
    if not path:
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("VIENEU_TECH_TERMS_FILE must contain a JSON object.")
        return data


@lru_cache(maxsize=1)
def get_tech_terms() -> dict[str, Any]:
    override = _load_override_terms()

    exact_rewrites = dict(_EXACT_REWRITES)
    exact_rewrites.update(
        {
            k.lower(): v
            for k, v in override.get("exact_rewrites", {}).items()
            if isinstance(k, str) and isinstance(v, str)
        }
    )

    safe_acronyms = set(_SAFE_ACRONYMS)
    safe_acronyms.update(
        {
            item.lower()
            for item in override.get("uppercase_acronyms", [])
            if isinstance(item, str)
        }
    )

    context_acronyms = set(_CONTEXT_ACRONYMS)
    context_acronyms.update(
        {
            item.lower()
            for item in override.get("context_acronyms", [])
            if isinstance(item, str)
        }
    )

    return {
        "exact_rewrites": exact_rewrites,
        "safe_acronyms": safe_acronyms,
        "context_acronyms": context_acronyms,
    }


def _canonicalize_segment(segment: str, allow_context_acronyms: bool) -> str:
    if not segment:
        return segment

    terms = get_tech_terms()
    lowered = segment.lower()

    if lowered in terms["exact_rewrites"]:
        return terms["exact_rewrites"][lowered]

    if lowered in terms["safe_acronyms"]:
        return segment.upper()

    if allow_context_acronyms and lowered in terms["context_acronyms"]:
        return segment.upper()

    return segment


def _looks_like_bridge_acronym(segment: str) -> bool:
    lowered = segment.lower()
    terms = get_tech_terms()
    return (
        (lowered in terms["safe_acronyms"] or lowered in terms["context_acronyms"])
        and segment.isalpha()
        and len(segment) >= 2
    )


def _normalize_segment_sequence(segments: list[str], convert_numbers: bool = False) -> str:
    normalized: list[str] = []
    for segment in segments:
        replacement = _canonicalize_segment(segment, allow_context_acronyms=True)
        normalized.extend(replacement.split())

    for idx in range(1, len(normalized) - 1):
        if normalized[idx] != "4":
            continue
        if _looks_like_bridge_acronym(normalized[idx - 1]) and _looks_like_bridge_acronym(normalized[idx + 1]):
            normalized[idx] = "for"

    for idx in range(len(normalized) - 1):
        if normalized[idx].lower() in {"v", "ver", "version"} and normalized[idx + 1].isdigit():
            normalized[idx] = "version"

    if convert_numbers:
        normalized = [_number_to_english(segment) if segment.isdigit() else segment for segment in normalized]

    return " ".join(normalized)


def _render_filename_stem(stem: str) -> str:
    if stem.isalpha() and stem.isupper():
        lowered = stem.lower()
        terms = get_tech_terms()
        if lowered in terms["safe_acronyms"] or lowered in terms["context_acronyms"] or len(stem) <= 3:
            return _spell_letters(stem)
        return stem.title()

    return rewrite_mixed_tech_text(stem)


def _render_filename_token(token: str) -> str | None:
    parts = token.split(".")
    if len(parts) < 2 or any(not part for part in parts):
        return None

    last_part = parts[-1].lower()
    if last_part not in _FILE_EXTENSIONS:
        return None

    rendered_parts = [_render_filename_stem(parts[0])]
    for part in parts[1:]:
        rendered_parts.append("chấm")
        if part.lower() in _FILE_EXTENSIONS:
            rendered_parts.append(_spell_letters(part))
        else:
            rendered_parts.append(_render_filename_stem(part))

    return _collapse_spaces(" ".join(rendered_parts))


def _render_dotfile_token(token: str) -> str | None:
    if not token.startswith(".") or len(token) < 2:
        return None

    parts = token[1:].split(".")
    if any(not part for part in parts):
        return None

    rendered_parts = ["chấm"]
    terms = get_tech_terms()
    for idx, part in enumerate(parts):
        lowered = part.lower()
        if idx:
            rendered_parts.append("chấm")

        if lowered in terms["exact_rewrites"]:
            rendered_parts.append(terms["exact_rewrites"][lowered])
        elif lowered in _FILE_EXTENSIONS or lowered in terms["safe_acronyms"] or lowered in terms["context_acronyms"] or len(part) <= 3:
            rendered_parts.append(_spell_letters(part))
        else:
            rendered_parts.append(rewrite_mixed_tech_text(part))

    return _collapse_spaces(" ".join(rendered_parts))


def _render_domain_token(token: str) -> str | None:
    parts = token.split(".")
    if len(parts) < 2 or any(not part for part in parts):
        return None

    tld = parts[-1].lower()
    if not tld.isalpha() or tld not in _DOMAIN_TLDS:
        return None

    rendered_parts: list[str] = []
    for idx, part in enumerate(parts):
        if idx:
            rendered_parts.append("chấm")

        if idx == len(parts) - 1:
            rendered_parts.append(_spell_letters(part))
        else:
            rendered_parts.append(rewrite_mixed_tech_text(part))

    return _collapse_spaces(" ".join(rendered_parts))


def _render_versioned_token(token: str) -> str | None:
    if not any(ch.isalpha() for ch in token) or not any(ch.isdigit() for ch in token):
        return None

    working = _VERSION_DOT_RE.sub(" point ", token)
    working = working.replace(".", " ")
    working = _SEPARATOR_RE.sub(" ", working)
    working = _CAMEL_BOUNDARY_RE.sub(" ", working)
    working = _ALNUM_BOUNDARY_RE.sub(" ", working)
    segments = working.split()

    if len(segments) <= 1:
        return None

    return _normalize_segment_sequence(segments, convert_numbers=True)


def _render_email_local_part(local_part: str) -> str:
    rendered_parts: list[str] = []
    current = []

    def flush_current() -> None:
        if current:
            rendered_parts.append(rewrite_mixed_tech_text("".join(current)))
            current.clear()

    for char in local_part:
        if char in _EMAIL_LOCAL_SEPARATORS:
            flush_current()
            rendered_parts.append(_EMAIL_LOCAL_SEPARATORS[char])
        else:
            current.append(char)

    flush_current()
    return _collapse_spaces(" ".join(rendered_parts))


def _render_email_token(email: str) -> str:
    local_part, domain = email.rsplit("@", 1)
    rendered_domain = _render_domain_token(domain) or rewrite_mixed_tech_text(domain)
    return _collapse_spaces(f"{_render_email_local_part(local_part)} a còng {rendered_domain}")


def _render_url_token(url: str) -> str:
    parse_target = url if "://" in url else f"https://{url}"
    parsed = urlsplit(parse_target)
    rendered_parts: list[str] = []

    if "://" in url and parsed.scheme:
        rendered_parts.append(_spell_letters(parsed.scheme))

    host = parsed.hostname or ""
    if host:
        rendered_parts.append(_render_domain_token(host) or rewrite_mixed_tech_text(host))

    if parsed.port:
        rendered_parts.append("port")
        rendered_parts.append(str(parsed.port))

    for segment in parsed.path.split("/"):
        if segment:
            rendered_parts.append(rewrite_mixed_tech_text(segment))

    if parsed.query:
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            rendered_parts.append(_render_query_key_token(key))
            if value:
                rendered_parts.append(_render_query_value_token(value))

    if parsed.fragment:
        rendered_parts.append(rewrite_mixed_tech_text(parsed.fragment))

    return _collapse_spaces(" ".join(rendered_parts))


def _render_path_segment(segment: str) -> str:
    filename_rendered = _render_filename_token(segment)
    if filename_rendered is not None:
        return filename_rendered

    dotfile_rendered = _render_dotfile_token(segment)
    if dotfile_rendered is not None:
        return dotfile_rendered

    return rewrite_mixed_tech_text(segment)


def _render_path_token(token: str) -> str | None:
    parts = [part for part in token.split("/") if part]
    if len(parts) < 2:
        return None

    return _collapse_spaces(" ".join(_render_path_segment(part) for part in parts))


def _render_ip_token(token: str) -> str:
    octets = token.split(".")
    rendered = [" ".join(_DIGIT_WORDS[digit] for digit in octet) for octet in octets]
    return " chấm ".join(rendered)


def _render_host_port_token(token: str) -> str:
    host, port = token.rsplit(":", 1)
    if _IP_RE.fullmatch(host):
        rendered_host = _render_ip_token(host)
    else:
        rendered_host = _render_domain_token(host) or rewrite_mixed_tech_text(host)
    return _collapse_spaces(f"{rendered_host} port {_number_to_english(port)}")


def _render_env_var_token(token: str) -> str:
    rendered_parts: list[str] = []
    terms = get_tech_terms()
    lowered_token = token.lower()
    if lowered_token in terms["exact_rewrites"]:
        return terms["exact_rewrites"][lowered_token]
    for part in token.split("_"):
        lowered = part.lower()
        if lowered in terms["exact_rewrites"]:
            rendered_parts.append(terms["exact_rewrites"][lowered])
        elif lowered in terms["safe_acronyms"] or lowered in terms["context_acronyms"]:
            rendered_parts.append(part.upper())
        elif part.isdigit():
            rendered_parts.append(_number_to_english(part))
        else:
            rendered_parts.append(part.title())
    return _collapse_spaces(" ".join(rendered_parts))


def _render_query_key_token(token: str) -> str:
    normalized = token.replace("-", "_")
    if "_" in normalized:
        return _render_snake_identifier_token(normalized)
    return rewrite_mixed_tech_text(token)


def _render_query_value_token(token: str) -> str:
    if token.isdigit():
        return _number_to_english(token)
    if _DECIMAL_VALUE_RE.fullmatch(token):
        integer_part, fractional_part = token.split(".", 1)
        return _collapse_spaces(
            f"{_number_to_english(integer_part)} point {' '.join(_DIGIT_WORDS[digit] for digit in fractional_part)}"
        )
    return rewrite_mixed_tech_text(token)


def _render_snake_identifier_token(token: str) -> str:
    rendered_parts: list[str] = []
    terms = get_tech_terms()
    lowered_token = token.lower()
    if lowered_token in terms["exact_rewrites"]:
        return terms["exact_rewrites"][lowered_token]
    for part in token.split("_"):
        lowered = part.lower()
        if lowered in terms["exact_rewrites"]:
            rendered_parts.append(terms["exact_rewrites"][lowered])
        elif lowered in terms["safe_acronyms"] or lowered in terms["context_acronyms"]:
            rendered_parts.append(part.upper())
        elif len(part) == 1 and part.isalpha():
            rendered_parts.append(part.upper())
        elif part.isdigit():
            rendered_parts.append(_number_to_english(part))
        else:
            rendered_parts.append(part)
    return _collapse_spaces(" ".join(rendered_parts))


def _render_version_number(token: str) -> str:
    if "." not in token:
        return _number_to_english(token)

    rendered_parts: list[str] = []
    for idx, part in enumerate(token.split(".")):
        if idx:
            rendered_parts.append("point")
        rendered_parts.append(_number_to_english(part))
    return " ".join(rendered_parts)


def _render_word_version_phrase(match: re.Match[str]) -> str:
    word, version = match.groups()
    lowered = word.lower()
    terms = get_tech_terms()
    if lowered not in terms["exact_rewrites"] and lowered not in terms["safe_acronyms"] and lowered not in terms["context_acronyms"]:
        return match.group(0)

    rendered_word = rewrite_mixed_tech_text(word)
    rendered_version = _render_version_number(version)
    return _collapse_spaces(f"{rendered_word} {rendered_version}")


def _render_config_assignment(match: re.Match[str]) -> str:
    key, value = match.groups()
    rendered_key = _render_query_key_token(key)
    if value.startswith(("http://", "https://", "www.")):
        rendered_value = _render_url_token(value)
    elif _EMAIL_RE.fullmatch(value):
        rendered_value = _render_email_token(value)
    elif _WINDOWS_PATH_RE.fullmatch(value):
        rendered_value = _render_windows_path_token(value)
    else:
        rendered_value = _render_query_value_token(value)
    return _collapse_spaces(f"{rendered_key} {rendered_value}")


def _render_cli_flag_token(token: str) -> str:
    body = token[2:]
    if "=" in body:
        key, value = body.split("=", 1)
    else:
        key, value = body, None

    rendered_key = rewrite_mixed_tech_text(key.replace("-", " "))
    if value is None:
        return rendered_key

    if value.isdigit():
        rendered_value = _number_to_english(value)
    elif _DECIMAL_VALUE_RE.fullmatch(value):
        integer_part, fractional_part = value.split(".", 1)
        rendered_value = _collapse_spaces(
            f"{_number_to_english(integer_part)} point {' '.join(_DIGIT_WORDS[digit] for digit in fractional_part)}"
        )
    else:
        rendered_value = rewrite_mixed_tech_text(value)
    return _collapse_spaces(f"{rendered_key} {rendered_value}")


def _render_windows_path_token(token: str) -> str:
    drive = token[0].upper()
    remainder = token[2:].replace("\\", "/")
    parts = [part for part in remainder.split("/") if part]
    rendered_parts = [f"{drive} drive"]
    rendered_parts.extend(_render_path_segment(part) for part in parts)
    return _collapse_spaces(" ".join(rendered_parts))


def rewrite_mixed_tech_text(text: str) -> str:
    """
    Rewrite common Vietnamese tech/code-switch tokens into forms that the
    bilingual phonemizer handles more naturally.
    """
    if not text:
        return text

    text = _SPACED_PLUS_RE.sub(" cộng ", text)
    text = _CONFIG_ASSIGNMENT_RE.sub(_render_config_assignment, text)
    text = _URL_RE.sub(lambda match: _render_url_token(match.group(0)), text)
    text = _EMAIL_RE.sub(lambda match: _render_email_token(match.group(0)), text)
    text = _HOST_PORT_RE.sub(lambda match: _render_host_port_token(match.group(0)), text)
    text = _IP_RE.sub(lambda match: _render_ip_token(match.group(0)), text)
    text = _WORD_VERSION_RE.sub(_render_word_version_phrase, text)
    text = _CLI_FLAG_RE.sub(lambda match: _render_cli_flag_token(match.group(0)), text)
    text = _WINDOWS_PATH_RE.sub(lambda match: _render_windows_path_token(match.group(0)), text)
    text = _PATH_RE.sub(lambda match: _render_path_token(match.group(0)) or match.group(0), text)
    text = _DOTFILE_RE.sub(lambda match: _render_dotfile_token(match.group(0)) or match.group(0), text)
    text = _ENV_VAR_RE.sub(lambda match: _render_env_var_token(match.group(0)), text)
    text = _SNAKE_IDENTIFIER_RE.sub(lambda match: _render_snake_identifier_token(match.group(0)), text)
    terms = get_tech_terms()

    def replace_token(match: re.Match[str]) -> str:
        token = match.group(0)
        lowered = token.lower()

        if lowered in terms["exact_rewrites"]:
            return terms["exact_rewrites"][lowered]

        domain_rendered = _render_domain_token(token)
        if domain_rendered is not None:
            return domain_rendered

        filename_rendered = _render_filename_token(token)
        if filename_rendered is not None:
            return filename_rendered

        dotfile_rendered = _render_dotfile_token(token)
        if dotfile_rendered is not None:
            return dotfile_rendered

        path_rendered = _render_path_token(token)
        if path_rendered is not None:
            return path_rendered

        versioned_rendered = _render_versioned_token(token)
        if versioned_rendered is not None:
            return versioned_rendered

        if "." in token:
            return token

        if _SEPARATOR_RE.search(token):
            pieces = [piece for piece in _SEPARATOR_RE.split(token) if piece]
            if len(pieces) == 1:
                return rewrite_mixed_tech_text(pieces[0])
            rewritten = [rewrite_mixed_tech_text(piece) for piece in pieces]
            return _collapse_spaces(" ".join(rewritten))

        if not any(ch.isascii() and ch.isalpha() for ch in token):
            return token

        split_token = _CAMEL_BOUNDARY_RE.sub(" ", token)
        split_token = _ALNUM_BOUNDARY_RE.sub(" ", split_token)
        segments = split_token.split()

        if len(segments) == 1:
            return _canonicalize_segment(segments[0], allow_context_acronyms=False)

        return _normalize_segment_sequence(segments)

    return _collapse_spaces(_ASCII_TOKEN_RE.sub(replace_token, text))
