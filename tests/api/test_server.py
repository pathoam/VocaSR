"""Tests for the Voca API server endpoints.

Uses FastAPI's TestClient (backed by httpx) which runs the ASGI app
in-process — no real HTTP server or port needed.

The transcription service is a real object implementing the expected interface,
not a MagicMock, so we verify the server correctly calls service methods and
handles responses.
"""

import io
import json
import struct
import wave
import pytest
import numpy as np
from unittest.mock import patch

from voca.api import server as server_module


class FakeASRService:
    """Minimal ASR service that satisfies the server's interface."""

    def __init__(self):
        self.model_size = "0.6B"
        self._streaming = True
        self._sessions = {}
        self._session_results = {}
        self._sessions_lock = __import__("threading").Lock()

    def supports_streaming(self):
        return self._streaming

    def active_session_count(self):
        with self._sessions_lock:
            return len(self._sessions)

    def transcribe(self, audio_file_path, context=None):
        return f"transcribed:{audio_file_path}"

    def get_backend_info(self):
        return {
            "backend": "fake",
            "backend_preference": "auto",
            "vllm_error": None,
            "kv_cache_mb": 256,
        }

    def start_streaming_session(self, language=None, context=None):
        sid = f"session-{len(self._sessions)}"
        with self._sessions_lock:
            self._sessions[sid] = {"language": language, "context": context}
        return sid

    def feed_chunk_session(self, session_id, audio_chunk):
        pass

    def get_partial_result(self, session_id):
        return "partial..."

    def finish_streaming_session(self, session_id):
        with self._sessions_lock:
            self._sessions.pop(session_id, None)
        return "final result"

    def cleanup_session(self, session_id):
        with self._sessions_lock:
            self._sessions.pop(session_id, None)


def _make_wav_bytes(duration_sec=0.1, sample_rate=16000) -> bytes:
    """Generate a valid WAV file in memory."""
    n_samples = int(sample_rate * duration_sec)
    samples = np.zeros(n_samples, dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _reset_server_state():
    """Reset server module globals between tests."""
    old_app = server_module._app
    old_service = server_module._service
    old_max = server_module._max_streams
    old_sem = server_module._transcribe_semaphore
    server_module._app = None
    server_module._service = None
    server_module._transcribe_semaphore = None
    server_module._max_streams = 8
    with server_module._sessions_lock:
        server_module._sessions.clear()
    yield
    server_module._app = old_app
    server_module._service = old_service
    server_module._max_streams = old_max
    server_module._transcribe_semaphore = old_sem


@pytest.fixture
def fake_service():
    return FakeASRService()


@pytest.fixture
def client(fake_service):
    from fastapi.testclient import TestClient
    server_module.configure(fake_service, max_streams=4)
    app = server_module._get_app()
    return TestClient(app)


# ---------------------------------------------------------------
# Context helpers (pure functions, no server needed)
# ---------------------------------------------------------------

class TestContextHelpers:
    def test_context_from_terms_list(self):
        result = server_module._context_from_terms(["Kubernetes", "PyTorch"])
        assert "Kubernetes" in result
        assert "PyTorch" in result

    def test_context_from_terms_empty_list(self):
        assert server_module._context_from_terms([]) is None

    def test_context_from_terms_none(self):
        assert server_module._context_from_terms(None) is None

    def test_context_from_terms_not_list(self):
        assert server_module._context_from_terms("not a list") is None

    def test_parse_context_terms_json_string(self):
        result = server_module._parse_context_terms('["foo", "bar"]')
        assert result == ["foo", "bar"]

    def test_parse_context_terms_csv_string(self):
        result = server_module._parse_context_terms("foo, bar, baz")
        assert result == ["foo", "bar", "baz"]

    def test_parse_context_terms_list_passthrough(self):
        result = server_module._parse_context_terms(["a", "b"])
        assert result == ["a", "b"]

    def test_parse_context_terms_none(self):
        assert server_module._parse_context_terms(None) is None

    def test_parse_context_terms_empty_string(self):
        assert server_module._parse_context_terms("") is None

    def test_resolve_context_merges_both(self):
        result = server_module._resolve_context("Use formal tone.", ["Kubernetes"])
        assert "formal tone" in result
        assert "Kubernetes" in result

    def test_resolve_context_only_terms(self):
        result = server_module._resolve_context(None, ["PyTorch"])
        assert "PyTorch" in result

    def test_resolve_context_only_text(self):
        result = server_module._resolve_context("Be concise.", None)
        assert "concise" in result

    def test_resolve_context_both_none(self):
        assert server_module._resolve_context(None, None) is None


# ---------------------------------------------------------------
# /v1/status
# ---------------------------------------------------------------

class TestStatusEndpoint:
    def test_status_returns_200(self, client):
        resp = client.get("/v1/status")
        assert resp.status_code == 200

    def test_status_fields(self, client):
        data = client.get("/v1/status").json()
        assert data["status"] == "ready"
        assert data["model"] == "qwen-0.6B"
        assert data["streaming_supported"] is True
        assert data["backend"] == "fake"
        assert data["max_streams"] == 4
        assert data["active_streams"] == 0
        assert "kv_cache_mb" in data

    def test_status_not_configured(self):
        from fastapi.testclient import TestClient
        server_module._service = None
        app = server_module._get_app()
        c = TestClient(app)
        data = c.get("/v1/status").json()
        assert data["status"] == "not_configured"


# ---------------------------------------------------------------
# /v1/transcribe
# ---------------------------------------------------------------

class TestTranscribeEndpoint:
    @patch("voca.api.auth.AUTH_ENABLED", False)
    def test_transcribe_returns_text(self, client):
        wav = _make_wav_bytes()
        resp = client.post(
            "/v1/transcribe",
            files={"file": ("test.wav", wav, "audio/wav")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "transcribed:" in data["text"]
        assert "processing_time_seconds" in data

    @patch("voca.api.auth.AUTH_ENABLED", False)
    def test_transcribe_503_without_service(self):
        from fastapi.testclient import TestClient
        server_module._service = None
        app = server_module._get_app()
        c = TestClient(app)
        wav = _make_wav_bytes()
        resp = c.post("/v1/transcribe", files={"file": ("test.wav", wav, "audio/wav")})
        assert resp.status_code == 503

    @patch("voca.api.auth.AUTH_ENABLED", False)
    def test_transcribe_with_context(self, client, fake_service):
        """Verify context is passed through to the service."""
        calls = []
        original = fake_service.transcribe

        def tracking_transcribe(path, context=None):
            calls.append(context)
            return original(path, context)

        fake_service.transcribe = tracking_transcribe
        wav = _make_wav_bytes()
        resp = client.post(
            "/v1/transcribe",
            files={"file": ("test.wav", wav, "audio/wav")},
            data={"context": "Use formal tone."},
        )
        assert resp.status_code == 200
        assert len(calls) == 1
        assert "formal tone" in calls[0]


# ---------------------------------------------------------------
# /v1/streams/reset
# ---------------------------------------------------------------

class TestStreamsResetEndpoint:
    @patch("voca.api.auth.AUTH_ENABLED", False)
    def test_reset_returns_200(self, client):
        resp = client.post("/v1/streams/reset")
        assert resp.status_code == 200
        data = resp.json()
        assert "cleaned" in data
        assert "active_streams" in data

    @patch("voca.api.auth.AUTH_ENABLED", False)
    def test_reset_cleans_tracked_sessions(self, client, fake_service):
        # Manually add some sessions
        with server_module._sessions_lock:
            server_module._sessions["s1"] = {"last_active": 0, "user_id": "u1", "frame_id": None}
            server_module._sessions["s2"] = {"last_active": 0, "user_id": "u1", "frame_id": None}

        resp = client.post("/v1/streams/reset")
        data = resp.json()
        assert data["cleaned"] >= 2

        with server_module._sessions_lock:
            assert len(server_module._sessions) == 0


# ---------------------------------------------------------------
# /v1/stream WebSocket
# ---------------------------------------------------------------

class TestStreamWebSocket:
    @patch("voca.api.auth.AUTH_ENABLED", False)
    def test_websocket_streaming_flow(self, client, fake_service):
        with client.websocket_connect("/v1/stream") as ws:
            # Send config message
            ws.send_json({"language": "en"})

            # Should get status response
            msg = ws.receive_json()
            assert msg["status"] == "streaming"
            assert "session_id" in msg

            # Send audio chunk (raw int16 PCM)
            chunk = np.zeros(1600, dtype=np.int16)
            ws.send_bytes(chunk.tobytes())

            # Should get partial result
            msg = ws.receive_json()
            assert "partial" in msg

            # Send finish command
            ws.send_json({"action": "finish"})

            # Should get final result
            msg = ws.receive_json()
            assert msg.get("final") is True
            assert "text" in msg

    @patch("voca.api.auth.AUTH_ENABLED", False)
    def test_websocket_no_streaming_support(self, fake_service):
        from fastapi.testclient import TestClient
        fake_service._streaming = False
        server_module.configure(fake_service, max_streams=4)
        app = server_module._get_app()
        c = TestClient(app)

        # WebSocket should be rejected
        with pytest.raises(Exception):
            with c.websocket_connect("/v1/stream") as ws:
                pass
