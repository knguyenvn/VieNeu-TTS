import json

import pytest

from vieneu_utils import phonemize_text as phonemize_module
from vieneu_utils.tech_text import get_tech_terms, rewrite_mixed_tech_text


def setup_function():
    get_tech_terms.cache_clear()
    phonemize_module._phonemize_cached.cache_clear()


def teardown_function():
    get_tech_terms.cache_clear()
    phonemize_module._phonemize_cached.cache_clear()


def test_rewrite_handles_common_vietnamese_tech_mix():
    text = "chatgpt đọc api của openai trên macos, deploy docker lên k8s qua wifi 5g"
    assert rewrite_mixed_tech_text(text) == (
        "ChatGPT đọc API của OpenAI trên macOS, deploy Docker lên Kubernetes qua Wi-Fi five G"
    )


def test_rewrite_handles_bridged_acronyms_without_touching_vietnamese_ai():
    text = "ai đang demo gpu4ai và ai4vn"
    assert rewrite_mixed_tech_text(text) == "ai đang demo GPU for AI và AI for VN"


def test_rewrite_handles_known_domains():
    assert rewrite_mixed_tech_text("truy cập gpu4ai.vn ngay") == "truy cập GPU for AI chấm V N ngay"
    assert rewrite_mixed_tech_text("xem docs.gpu4ai.vn") == "xem docs chấm GPU for AI chấm V N"


def test_rewrite_handles_common_filenames():
    assert rewrite_mixed_tech_text("SOUL.md và AGENTS.md") == "Soul chấm M D và Agents chấm M D"
    assert rewrite_mixed_tech_text("mở api.env để cấu hình") == "mở API chấm E N V để cấu hình"


def test_rewrite_handles_multidot_filenames():
    assert rewrite_mixed_tech_text("config.prod.yaml") == "config chấm prod chấm Y A M L"


def test_rewrite_handles_dotfiles_and_paths():
    assert rewrite_mixed_tech_text("mở .env.local và .gitignore") == (
        "mở chấm E N V chấm local và chấm Git Ignore"
    )
    assert rewrite_mixed_tech_text("sửa src/app/page.tsx với docker-compose.yml") == (
        "sửa src app page chấm T S X với Docker compose chấm Y M L"
    )
    assert rewrite_mixed_tech_text(r"mở C:\Users\Kien\Desktop\config.yaml") == (
        "mở C drive Users Kien Desktop config chấm Y A M L"
    )


def test_rewrite_handles_env_vars_and_local_hosts():
    assert rewrite_mixed_tech_text("set OPENAI_API_KEY và GRADIO_SERVER_PORT") == (
        "set OpenAI API Key và Gradio Server Port"
    )
    assert rewrite_mixed_tech_text("mở localhost:3000 hoặc 127.0.0.1:8000") == (
        "mở localhost port three zero zero zero hoặc one two seven chấm zero chấm zero chấm one port eight zero zero zero"
    )
    assert rewrite_mixed_tech_text("dùng openai_api_key và gradio_server_port") == (
        "dùng OpenAI API key và Gradio server port"
    )


def test_rewrite_handles_versioned_model_names():
    assert rewrite_mixed_tech_text("claude-3.7-sonnet và gemini-2.5-pro") == (
        "Claude three point seven Sonnet và Gemini two point five Pro"
    )
    assert rewrite_mixed_tech_text("qwen2.5-coder với llama3.1") == (
        "Qwen two point five Coder với Llama three point one"
    )


def test_rewrite_handles_url_and_email_tokens():
    assert rewrite_mixed_tech_text("gửi mail tới support@gpu4ai.vn") == (
        "gửi mail tới support a còng GPU for AI chấm V N"
    )
    assert rewrite_mixed_tech_text("mở https://docs.gpu4ai.vn/api-v1") == (
        "mở H T T P S docs chấm GPU for AI chấm V N API version one"
    )
    assert rewrite_mixed_tech_text("gọi https://api.openai.com/v1/chat?model=gpt-4.1&max_tokens=4096&top_p=0.95") == (
        "gọi H T T P S API chấm OpenAI chấm C O M version one chat model GPT four point one max tokens four zero nine six top <en>p</en> zero point nine five"
    )


def test_rewrite_handles_common_cloud_and_platform_terms():
    assert rewrite_mixed_tech_text("deploy lên aws s3 rồi ssh vào vps") == (
        "deploy lên AWS S three rồi SSH vào VPS"
    )
    assert rewrite_mixed_tech_text("telegram, whatsapp, discord và paypal") == (
        "Telegram, WhatsApp, Discord và PayPal"
    )
    assert rewrite_mixed_tech_text("n8n và githubactions đọc config.toml") == (
        "N eight N và GitHub Actions đọc config chấm T O M L"
    )


def test_rewrite_handles_common_frontend_backend_terms():
    assert rewrite_mixed_tech_text("reactjs + nextjs + typescript + postgresql") == (
        "React JS cộng Next JS cộng TypeScript cộng PostgreSQL"
    )
    assert rewrite_mixed_tech_text("node.js 22 với next.js 15 trên vscode") == (
        "Node JS twenty two với Next JS fifteen trên VS Code"
    )
    assert rewrite_mixed_tech_text("python 3.12 với fastapi") == (
        "Python three point twelve với FastAPI"
    )


def test_rewrite_handles_cli_tooling_terms():
    assert rewrite_mixed_tech_text("npm install, pnpm dev, uv sync, pip install") == (
        "N P M install, P N P M dev, U V sync, pip install"
    )
    assert rewrite_mixed_tech_text("chạy --api-key=abc và --max-tokens=4096") == (
        "chạy API key abc và max tokens four zero nine six"
    )
    assert rewrite_mixed_tech_text("set --temperature=0.7 và --memory-util=0.3") == (
        "set temperature zero point seven và memory util zero point three"
    )
    assert rewrite_mixed_tech_text("set top_p và max_tokens") == (
        "set top <en>p</en> và max tokens"
    )
    assert rewrite_mixed_tech_text("set top_k, presence_penalty và frequency_penalty") == (
        "set top <en>k</en>, presence penalty và frequency penalty"
    )


def test_rewrite_handles_config_assignment_lines():
    assert rewrite_mixed_tech_text("top_p: 0.95, temperature=0.7, max_tokens: 4096") == (
        "top <en>p</en> zero point nine five, temperature zero point seven, max tokens four zero nine six"
    )
    assert rewrite_mixed_tech_text("model_name: gpt-4.1 và base_url=https://api.openai.com/v1") == (
        "model name GPT four point one và base URL H T T P S API chấm OpenAI chấm C O M version one"
    )


def test_rewrite_preserves_plain_english_text():
    assert rewrite_mixed_tech_text("hello world") == "hello world"


def test_override_file_merges_custom_terms(monkeypatch, tmp_path):
    override_path = tmp_path / "tech_terms.json"
    override_path.write_text(
        json.dumps(
            {
                "exact_rewrites": {"ragflow": "RAG Flow"},
                "uppercase_acronyms": ["rag"],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("VIENEU_TECH_TERMS_FILE", str(override_path))
    get_tech_terms.cache_clear()

    assert rewrite_mixed_tech_text("ragflow và rag") == "RAG Flow và RAG"


def test_phonemize_batch_applies_rewrite_before_g2p(monkeypatch):
    calls = []

    class FakeG2P:
        def phonemize_batch(self, texts, phoneme_dict=None):
            calls.append((texts, phoneme_dict))
            return texts

    monkeypatch.setattr(phonemize_module, "_get_g2p", lambda: FakeG2P())

    result = phonemize_module.phonemize_batch(["gpu4ai hỗ trợ tiếng Việt"], skip_normalize=True)

    assert result == ["GPU for AI hỗ trợ tiếng Việt"]
    assert calls == [(["GPU for AI hỗ trợ tiếng Việt"], None)]


def test_phonemize_batch_rewrites_before_normalizer(monkeypatch):
    calls = []

    class FakeG2P:
        def phonemize_batch(self, texts, phoneme_dict=None):
            return texts

    class FakeNormalizer:
        def normalize(self, text):
            calls.append(text)
            return f"NORM<{text}>"

    monkeypatch.setattr(phonemize_module, "_get_g2p", lambda: FakeG2P())
    monkeypatch.setattr(phonemize_module, "_get_normalizer", lambda: FakeNormalizer())

    result = phonemize_module.phonemize_batch(["chatgpt nói về openai"], skip_normalize=False)

    assert calls == ["ChatGPT nói về OpenAI"]
    assert result == ["NORM<ChatGPT nói về OpenAI>"]


def test_phonemize_with_dict_skip_normalize_still_rewrites(monkeypatch):
    calls = []

    class FakeG2P:
        def phonemize_batch(self, texts, phoneme_dict=None):
            calls.append((texts, phoneme_dict))
            return texts

    monkeypatch.setattr(phonemize_module, "_get_g2p", lambda: FakeG2P())

    result = phonemize_module.phonemize_with_dict("gpu4ai hỗ trợ tiếng Việt", skip_normalize=True)

    assert result == "GPU for AI hỗ trợ tiếng Việt"
    assert calls == [(["GPU for AI hỗ trợ tiếng Việt"], None)]


def test_normalize_text_rewrites_before_underlying_normalizer(monkeypatch):
    calls = []

    class FakeNormalizer:
        def normalize(self, text):
            calls.append(text)
            return f"NORM<{text}>"

    monkeypatch.setattr(phonemize_module, "_get_normalizer", lambda: FakeNormalizer())

    result = phonemize_module.normalize_text("gpu4ai hỗ trợ tiếng Việt")

    assert calls == ["GPU for AI hỗ trợ tiếng Việt"]
    assert result == "NORM<GPU for AI hỗ trợ tiếng Việt>"
