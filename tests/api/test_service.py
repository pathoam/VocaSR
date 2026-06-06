import importlib
import sys


def test_service_import_does_not_load_desktop_modules():
    for name in ("pynput", "voca.system.hotkeys", "voca.system.text_insertion", "voca.ui.gui"):
        sys.modules.pop(name, None)

    importlib.import_module("voca.service")

    assert "pynput" not in sys.modules
    assert "voca.system.hotkeys" not in sys.modules
    assert "voca.system.text_insertion" not in sys.modules
    assert "voca.ui.gui" not in sys.modules


def test_setting_prefers_voca_env(monkeypatch):
    service = importlib.import_module("voca.service")

    class Config:
        def get_setting(self, key, default):
            return 1111

    monkeypatch.setenv("VOCA_API_PORT", "2222")
    monkeypatch.setenv("VOICEBOX_API_PORT", "3333")

    assert service._setting(Config(), ("VOCA_API_PORT", "VOICEBOX_API_PORT"), "api_port", 9876, int) == 2222


def test_setting_keeps_voicebox_env_alias(monkeypatch):
    service = importlib.import_module("voca.service")

    class Config:
        def get_setting(self, key, default):
            return 1111

    monkeypatch.delenv("VOCA_API_PORT", raising=False)
    monkeypatch.setenv("VOICEBOX_API_PORT", "3333")

    assert service._setting(Config(), ("VOCA_API_PORT", "VOICEBOX_API_PORT"), "api_port", 9876, int) == 3333
