#!/usr/bin/env python3

import sys

if "--serve-only" in sys.argv:
    from voca.service import main as service_main

    service_main()
    raise SystemExit

import time
import threading
from typing import Optional

from voca.audio.capture import AudioRecorder
from voca.transcription.base import TranscriptionService, StreamingTranscriptionService, TranscriptionError
from voca.system.hotkeys import HotkeyManager
from voca.system.text_insertion import TextInserter
from voca.config.manager import ConfigManager
from voca.text.corrections import CorrectionPipeline
from voca.text.vocabulary import VocabularyManager
from voca.data.collector import TrainingDataCollector
from voca.commands.detector import CommandDetector
from voca.commands.processor import CommandProcessor
from voca.commands.responder import CommandResponder
from voca.recording import RecordingOrchestrator
from voca.pipeline import TranscriptionPipeline, AppState
from voca.correction_flow import CorrectionFlow
from voca.utils.logging import get_logger


class VocaApp:
    """Main Voca application coordinator."""

    def __init__(self):
        import random

        self.app_id = random.randint(1000, 9999)
        self.logger = get_logger(__name__)
        self.logger.debug(f"VocaApp instance created with ID: {self.app_id}")
        self.config_manager = ConfigManager()

        # Initialize components
        self.audio_recorder: Optional[AudioRecorder] = None
        self.transcription_service: Optional[TranscriptionService] = None
        self.hotkey_manager: Optional[HotkeyManager] = None
        self.text_inserter: Optional[TextInserter] = None
        self.correction_pipeline: Optional[CorrectionPipeline] = None
        self.vocabulary_manager: Optional[VocabularyManager] = None
        self.training_data_collector: Optional[TrainingDataCollector] = None

        # Command mode components
        self.command_detector: Optional[CommandDetector] = None
        self.command_processor: Optional[CommandProcessor] = None
        self.command_responder: Optional[CommandResponder] = None

        # Extracted sub-components
        self._recording: Optional[RecordingOrchestrator] = None
        self._pipeline: Optional[TranscriptionPipeline] = None
        self._correction_flow: Optional[CorrectionFlow] = None

        # Runtime state
        self.current_audio_file: Optional[str] = None
        self._running = False
        self._state_lock = threading.Lock()
        self._use_gui = False

        # Callbacks for GUI integration
        self.on_transcription_complete = None
        self.on_status_change = None
        self.on_error = None

    # -----------------------------------------------------------------
    # Public API for GUI
    # -----------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def state(self) -> AppState:
        if self._pipeline:
            return self._pipeline.state
        return AppState.IDLE

    @state.setter
    def state(self, value: AppState) -> None:
        if self._pipeline:
            self._pipeline.state = value

    @property
    def _last_insertion(self):
        if self._pipeline:
            return self._pipeline.last_insertion
        return None

    def reinitialize_transcription(self) -> None:
        """Re-create the transcription service from current config."""
        self._initialize_transcription()
        self._rebuild_orchestrator()

    def change_hotkey(self, hotkey: str) -> None:
        """Change the hotkey combination."""
        if self.hotkey_manager:
            self.hotkey_manager.set_hotkey(hotkey)
            self.hotkey_manager.start_listening()
        self.config_manager.set_setting("hotkey", hotkey)

    def reload_corrections(self) -> None:
        """Reload correction rules from disk."""
        if self.correction_pipeline:
            self.correction_pipeline.reload()

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def _report_error(self, message: str, error_type: str = "General", suggestion: str = "") -> None:
        self.logger.error(f"{error_type}: {message}")
        if self.on_error:
            self.on_error(message, error_type, suggestion)

    def start(self) -> bool:
        try:
            self._initialize_audio()
            self._initialize_vocabulary()
            self._initialize_transcription()
            self._initialize_text_inserter()
            self._initialize_corrections()
            self._initialize_training_data()
            self._initialize_commands()
            self._initialize_hotkeys()
            self._build_sub_components()

            self.hotkey_manager.start_listening()
            self._running = True

            print(f"✅ Voca ready! Press {self.config_manager.get_hotkey()} to start recording")
            return True
        except Exception as e:
            print(f"Failed to start Voca: {e}")
            return False

    def stop(self) -> None:
        self._running = False
        with self._state_lock:
            if self.audio_recorder and self.audio_recorder.is_recording():
                try:
                    self.audio_recorder.stop_recording()
                except Exception as e:
                    print(f"Error stopping recording: {e}")
            if self.hotkey_manager:
                self.hotkey_manager.stop_listening()
            if self.current_audio_file and self._recording:
                self._recording.cleanup_audio_file(self.current_audio_file)

    # -----------------------------------------------------------------
    # Component initialization
    # -----------------------------------------------------------------

    def _initialize_audio(self) -> None:
        sample_rate = self.config_manager.get_audio_sample_rate()
        channels = self.config_manager.get_audio_channels()
        self.audio_recorder = AudioRecorder(sample_rate=sample_rate, channels=channels)

    def _initialize_transcription(self) -> None:
        from voca.transcription.registry import create_asr_service, available_modes

        mode = self.config_manager.get_transcription_mode()
        vocab_context = self.vocabulary_manager.get_context_string() if self.vocabulary_manager else None

        if mode not in available_modes():
            raise RuntimeError(f"Unknown transcription mode: {mode}")

        self.transcription_service = create_asr_service(mode, self.config_manager, vocab_context)

        if mode == "qwen" and hasattr(self.transcription_service, '_load_model'):
            print("Loading Qwen ASR model (may download on first run)...")
            self.transcription_service._load_model()
            print("Qwen ASR model loaded.")

        if not self.transcription_service.is_available():
            if mode == "qwen" and "local" in available_modes():
                self.logger.warning(
                    "Qwen ASR not available. Falling back to local Whisper. "
                    "Install qwen-asr (pip install qwen-asr) then restart."
                )
                print(
                    "⚠️  Qwen ASR not available, falling back to local Whisper. "
                    "Install qwen-asr (pip install qwen-asr) then restart."
                )
                self.transcription_service = create_asr_service("local", self.config_manager, vocab_context)
                if not self.transcription_service.is_available():
                    raise RuntimeError("Neither Qwen ASR nor local Whisper is available")
            else:
                raise RuntimeError(f"Transcription service ({mode}) is not available")

    def _initialize_text_inserter(self) -> None:
        platform_name = self.config_manager.get_platform()
        self.text_inserter = TextInserter(platform_name=platform_name)

    def _initialize_vocabulary(self) -> None:
        config_dir = self.config_manager.config_dir
        triggers = self.config_manager.get_command_triggers() if self.config_manager.is_command_mode_enabled() else []
        self.vocabulary_manager = VocabularyManager(config_dir, default_terms=triggers)

        def _on_vocabulary_change(context_string):
            if self.transcription_service and hasattr(self.transcription_service, 'set_context'):
                self.transcription_service.set_context(context_string)
                self.logger.info(f"ASR context updated: {context_string}")

        self.vocabulary_manager.on_change(_on_vocabulary_change)

    def _initialize_corrections(self) -> None:
        config_dir = self.config_manager.config_dir
        number_norm = self.config_manager.get_setting("number_normalization_enabled", True)
        self.correction_pipeline = CorrectionPipeline(config_dir, number_normalization_enabled=number_norm)

    def _initialize_training_data(self) -> None:
        config_dir = self.config_manager.config_dir
        enabled = self.config_manager.get_setting("training_data_enabled", True)
        max_mb = self.config_manager.get_setting("training_data_max_mb", 2048)
        self.training_data_collector = TrainingDataCollector(config_dir, max_mb=max_mb, enabled=enabled)

    def _initialize_commands(self) -> None:
        if self.config_manager.is_command_mode_enabled():
            triggers = self.config_manager.get_command_triggers()
            self.command_detector = CommandDetector(triggers)
            cmd_config = self.config_manager.get_command_mode_config()
            self.command_processor = CommandProcessor(
                openrouter_api_key=cmd_config.get("openrouter_api_key"),
                local_llm_endpoint=cmd_config.get("local_llm_endpoint"),
                model=cmd_config.get("openrouter_model"),
            )
            self.command_processor.vocabulary_manager = self.vocabulary_manager
            self.command_processor.text_inserter = self.text_inserter
            self.command_processor.voca_app = self
            response_method = cmd_config.get("response_method", "notification")
            self.command_responder = CommandResponder(method=response_method, gui_callback=None)

    def _initialize_hotkeys(self) -> None:
        hotkey = self.config_manager.get_hotkey()
        self.hotkey_manager = HotkeyManager(callback=self._on_hotkey_pressed)
        correction_hotkey = self.config_manager.get_setting("correction_hotkey", "ctrl+alt+space")
        if correction_hotkey:
            self.hotkey_manager.register_hotkey(correction_hotkey, self._on_correction_hotkey)
        self.hotkey_manager.set_hotkey(hotkey)

    def _build_sub_components(self) -> None:
        """Create the recording orchestrator, pipeline, and correction flow."""
        self._pipeline = TranscriptionPipeline(
            transcription_service=self.transcription_service,
            correction_pipeline=self.correction_pipeline,
            command_detector=self.command_detector,
            command_processor=self.command_processor,
            command_responder=self.command_responder,
            text_inserter=self.text_inserter,
            training_data_collector=self.training_data_collector,
            config_manager=self.config_manager,
            on_transcription_complete=self.on_transcription_complete,
            on_error=self.on_error,
        )
        self._recording = RecordingOrchestrator(
            audio_recorder=self.audio_recorder,
            transcription_service=self.transcription_service,
            config_manager=self.config_manager,
            on_audio_ready=self._on_audio_ready,
            on_error=lambda msg: self._report_error(msg, "Recording"),
        )
        self._correction_flow = CorrectionFlow(
            text_inserter=self.text_inserter,
            training_data_collector=self.training_data_collector,
            vocabulary_manager=self.vocabulary_manager,
            use_gui=self._use_gui,
        )

    def _rebuild_orchestrator(self) -> None:
        """Re-create recording orchestrator and pipeline after transcription service change."""
        if self._recording:
            self._recording.transcription_service = self.transcription_service
        if self._pipeline:
            self._pipeline.transcription_service = self.transcription_service

    # -----------------------------------------------------------------
    # Hotkey handlers
    # -----------------------------------------------------------------

    def _on_hotkey_pressed(self) -> None:
        self.logger.debug(f"Hotkey pressed! Current state: {self.state.value}")
        with self._state_lock:
            if not self._running:
                self.logger.debug("App not running, ignoring hotkey")
                return
            if self.state == AppState.IDLE:
                self.logger.debug("Starting recording...")
                self._start_recording()
            elif self.state == AppState.RECORDING:
                self.logger.debug("Stopping recording and transcribing...")
                self._stop_recording_and_transcribe()
            else:
                self.logger.debug(f"Hotkey pressed but app is in {self.state.value} state")

    def _on_correction_hotkey(self) -> None:
        if self._correction_flow and self._pipeline:
            self._correction_flow.handle_correction(self._pipeline.last_insertion)
            if self._pipeline.last_insertion:
                self._pipeline._last_insertion = None

    def _start_recording(self) -> None:
        try:
            self._pipeline.state = AppState.RECORDING
            print("🎤 Recording...", end="", flush=True)
            self._recording.start_recording()
        except Exception as e:
            print(f"Failed to start recording: {e}")
            self._pipeline.state = AppState.ERROR
            time.sleep(2)
            self._pipeline.state = AppState.IDLE

    def _stop_recording_and_transcribe(self) -> None:
        try:
            print(" transcribing...", end="", flush=True)
            self._pipeline.state = AppState.TRANSCRIBING
            self._recording.stop_recording()
        except Exception as e:
            print(f"Failed to stop recording: {e}")
            self._pipeline.state = AppState.ERROR
            time.sleep(2)
            self._pipeline.state = AppState.IDLE

    def _on_audio_ready(self, audio_file: str, was_streaming: bool) -> None:
        """Callback from RecordingOrchestrator when recording stops."""
        self.current_audio_file = audio_file
        if was_streaming:
            print(f"🧵 Creating streaming finish thread for: {audio_file}")
            t = threading.Thread(target=self._run_streaming_pipeline, args=(audio_file,))
        else:
            print(f"🧵 Creating transcription thread for: {audio_file}")
            t = threading.Thread(target=self._run_transcription_pipeline, args=(audio_file,))
        t.daemon = True
        t.start()
        print(f"🧵 Transcription thread started")

    def _run_transcription_pipeline(self, audio_file: str) -> None:
        try:
            self._pipeline.transcribe_and_insert(audio_file)
        finally:
            self._recording.cleanup_audio_file(audio_file)
            self.current_audio_file = None

    def _run_streaming_pipeline(self, audio_file: str) -> None:
        try:
            self._pipeline.finish_streaming_and_insert(audio_file)
        finally:
            self._recording.cleanup_audio_file(audio_file)
            self.current_audio_file = None

    # -----------------------------------------------------------------
    # Config reload
    # -----------------------------------------------------------------

    def reload_config(self) -> None:
        self.config_manager._load_config()

        if self.correction_pipeline:
            self.correction_pipeline.reload()
        if self.vocabulary_manager:
            self.vocabulary_manager.load()

        # Hotkey
        new_hotkey = self.config_manager.get_hotkey()
        if self.hotkey_manager and self.hotkey_manager.get_current_hotkey() != new_hotkey:
            try:
                self.hotkey_manager.set_hotkey(new_hotkey)
                self.hotkey_manager.start_listening()
                self.logger.info(f"Hotkey updated to: {new_hotkey}")
            except Exception as e:
                self.logger.error(f"Failed to update hotkey: {e}")

        # Transcription settings
        if self._transcription_settings_changed():
            try:
                self.logger.debug("Reinitializing transcription service...")
                self._initialize_transcription()
                self._rebuild_orchestrator()
                self.logger.info("Transcription service updated!")
            except Exception as e:
                self.logger.error(f"Failed to update transcription service: {e}")

        # Audio settings
        current_sample_rate = self.config_manager.get_audio_sample_rate()
        current_channels = self.config_manager.get_audio_channels()
        if self.audio_recorder and (
            current_sample_rate != getattr(self.audio_recorder, "sample_rate", 16000)
            or current_channels != getattr(self.audio_recorder, "channels", 1)
        ):
            self.logger.warning("Audio settings changed - restart Voca for changes to take effect")

        # Command mode
        if self.config_manager.is_command_mode_enabled():
            cmd_config = self.config_manager.get_command_mode_config()
            triggers = cmd_config.get("triggers", ["voca"])
            if self.vocabulary_manager:
                self.vocabulary_manager.set_default_terms(triggers)
            if not self.command_detector:
                self._initialize_commands()
                print("✅ Command mode enabled!")
            else:
                self.command_detector.triggers = triggers
                self.command_detector._build_trigger_pattern()
                if self.command_processor:
                    self.command_processor.set_openrouter_key(cmd_config.get("openrouter_api_key", ""))
                    self.command_processor.set_local_endpoint(cmd_config.get("local_llm_endpoint", ""))
                    self.command_processor.set_model(cmd_config.get("openrouter_model", ""))
                if self.command_responder:
                    self.command_responder.set_method(cmd_config.get("response_method", "notification"))
                print("✅ Command mode settings updated!")
        else:
            if self.command_detector:
                self.command_detector = None
                self.command_processor = None
                self.command_responder = None
            if self.vocabulary_manager:
                self.vocabulary_manager.set_default_terms([])
                print("✅ Command mode disabled")

        print("🔄 Configuration reloaded!")

    def _transcription_settings_changed(self) -> bool:
        """Check if transcription-related config differs from the live service."""
        current_mode = self.config_manager.get_transcription_mode()
        current_language = self.config_manager.get_transcription_language()

        if hasattr(self.transcription_service, "model_size"):
            current_model = self.config_manager.get_local_model_size()
            if current_model != getattr(self.transcription_service, "model_size", None):
                return True
            if current_language != getattr(self.transcription_service, "language", "auto"):
                return True
        elif hasattr(self.transcription_service, "api_key"):
            current_api_key = self.config_manager.get_api_key()
            if current_api_key != getattr(self.transcription_service, "api_key", None):
                return True
            if current_language != getattr(self.transcription_service, "language", "auto"):
                return True
        return False

    # -----------------------------------------------------------------
    # Status / run_forever
    # -----------------------------------------------------------------

    def get_status(self) -> dict:
        return {
            "state": self.state.value,
            "running": self._running,
            "transcription_mode": self.config_manager.get_transcription_mode(),
            "hotkey": self.config_manager.get_hotkey(),
            "recording": self.audio_recorder.is_recording() if self.audio_recorder else False,
        }

    def run_forever(self) -> None:
        if not self.start():
            return
        try:
            print("Press Ctrl+C to exit\n")
            old_settings = None
            try:
                import termios
                if sys.stdin.isatty():
                    old_settings = termios.tcgetattr(sys.stdin)
                    new_settings = termios.tcgetattr(sys.stdin)
                    new_settings[3] = new_settings[3] & ~termios.ECHO
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, new_settings)
            except (ImportError, Exception):
                pass
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nExiting...")
        finally:
            if old_settings is not None:
                try:
                    import termios
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                except Exception:
                    pass
            self.stop()

    def test_hotkey_capture(self, timeout: int = 10) -> bool:
        hotkey = self.config_manager.get_hotkey()
        platform = self.config_manager.get_platform()
        print(f"\nHotkey Test")
        print("=" * 40)
        print(f"Platform: {platform}")
        print(f"Hotkey: {hotkey}")
        print(f"Timeout: {timeout} seconds")
        print("=" * 40)
        if platform == "macos":
            print("\nNote: On macOS, this requires Accessibility permissions.")
            print("If the test fails, check System Settings → Privacy & Security → Accessibility")
        print(f"\nPress '{hotkey}' within {timeout} seconds...")
        captured = threading.Event()

        def on_hotkey():
            print("\n✅ Hotkey captured successfully!")
            captured.set()

        try:
            test_manager = HotkeyManager(callback=on_hotkey)
            test_manager.set_hotkey(hotkey)
            test_manager.start_listening()
            success = captured.wait(timeout=timeout)
            test_manager.stop_listening()
            if not success:
                print("\n❌ Hotkey was NOT captured.")
                print("\nPossible causes:")
                if platform == "macos":
                    print("  • Accessibility permissions not granted")
                    print("  • Terminal app not added to Accessibility list")
                print("  • Hotkey conflict with another application")
                print("  • Invalid hotkey combination")
                print(f"\nTry a different hotkey with: --test-hotkey f11")
            return success
        except Exception as e:
            print(f"\n❌ Error setting up hotkey listener: {e}")
            return False


def _setup_signal_handlers():
    """Ensure Ctrl+C always exits, even during model downloads in threads."""
    import signal
    _first_interrupt = [True]

    def _handler(signum, frame):
        if _first_interrupt[0]:
            _first_interrupt[0] = False
            print("\nInterrupted. Press Ctrl+C again to force quit.")
            raise KeyboardInterrupt
        else:
            print("\nForce quitting...")
            import os
            os._exit(1)

    signal.signal(signal.SIGINT, _handler)


def _init_qwen_service(config_manager):
    from voca.service import init_qwen_service

    return init_qwen_service(config_manager)


def _start_api_server_only(config_manager):
    from voca.service import start_api_server_only

    start_api_server_only(config_manager)


def _start_api_daemon(app, config_manager):
    from voca.api.server import configure, start_server
    port = config_manager.get_setting("api_port", 9876)
    max_streams = config_manager.get_setting("api_max_streams", 8)
    number_norm = config_manager.get_setting("number_normalization_enabled", True)
    original_start = app.start

    def _patched_start():
        result = original_start()
        if result and app.transcription_service:
            configure(
                app.transcription_service,
                max_streams=max_streams,
                number_normalization_enabled=number_norm,
            )
            start_server(host="127.0.0.1", port=port, daemon=True)
            print(f"API server running on http://127.0.0.1:{port}")
        return result

    app.start = _patched_start


def _start_api_headless_for_gui(config_manager):
    from voca.api.server import configure, start_server
    service = _init_qwen_service(config_manager)
    port = config_manager.get_setting("api_port", 9876)
    max_streams = config_manager.get_setting("api_max_streams", 8)
    number_norm = config_manager.get_setting("number_normalization_enabled", True)
    configure(
        service,
        max_streams=max_streams,
        number_normalization_enabled=number_norm,
    )
    start_server(host="127.0.0.1", port=port, daemon=True)
    print(f"API server running on http://127.0.0.1:{port}")


def main():
    """Main entry point."""
    _setup_signal_handlers()

    debug_mode = "--debug" in sys.argv
    if debug_mode:
        from voca.utils.logging import set_debug_mode
        set_debug_mode(True)

    logger = get_logger(__name__)
    if debug_mode:
        logger.info("Debug mode enabled - verbose logging active")

    cli_mode = "--cli" in sys.argv
    force_mode = "--force" in sys.argv
    serve_mode = "--serve" in sys.argv
    serve_only_mode = "--serve-only" in sys.argv

    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            logger.info("Running in test mode...")
            app = VocaApp()
            if not app.start():
                sys.exit(1)
            logger.info("Test completed, exiting...")
            app.stop()
            return
        elif sys.argv[1] == "--config":
            app = VocaApp()
            print(f"Configuration file: {app.config_manager.get_config_path()}")
            return
        elif sys.argv[1] == "--test-hotkey":
            app = VocaApp()
            if len(sys.argv) > 2 and not sys.argv[2].startswith("--"):
                app.config_manager.config["hotkey"] = sys.argv[2]
            success = app.test_hotkey_capture()
            sys.exit(0 if success else 1)
        elif sys.argv[1] == "--help":
            print("Usage: voca [--cli|--test|--test-hotkey|--config|--help|--debug|--force|--serve|--serve-only]")
            print("")
            print("Voca runs in GUI mode by default (with system tray icon).")
            print("")
            print("Options:")
            print("  --cli         : Run in command-line mode (no GUI)")
            print("  --serve       : GUI mode + local API server on configured port")
            print("  --serve-only  : Headless API server only (no GUI/hotkeys)")
            print("  --test        : Test initialization and exit")
            print("  --test-hotkey : Test hotkey capture (diagnose permission issues)")
            print("                  Optionally specify hotkey: --test-hotkey f11")
            print("  --config      : Show configuration file path")
            print("  --force       : Kill any existing Voca instance before starting")
            print("  --help        : Show this help")
            print("  --debug       : Enable verbose debug logging (can combine with other flags)")
            return

    from voca.utils.singleton import ensure_single_instance
    if not ensure_single_instance(kill_existing=force_mode):
        if not force_mode:
            print("Hint: Use --force to stop the existing instance and start a new one.")
        sys.exit(1)

    config_manager = ConfigManager()
    if config_manager.is_first_run():
        from voca.ui.setup_wizard import run_setup_wizard
        run_setup_wizard(config_manager)

    if serve_only_mode:
        logger.info("Voca - Headless API server mode")
        _start_api_server_only(config_manager)
        return

    if cli_mode:
        logger.info("Voca - Voice-to-Text Transcription Tool (CLI mode)")
        app = VocaApp()
        if serve_mode:
            _start_api_daemon(app, config_manager)
        app.run_forever()
    else:
        if serve_mode:
            _start_api_headless_for_gui(config_manager)
        from voca.ui.gui import run_gui
        sys.exit(run_gui())


if __name__ == "__main__":
    main()
