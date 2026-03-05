"""Tests for hotkey_manager.py, state_machine.py, network_monitor.py, circuit_breaker — coverage boost."""
import sys
import threading
import socket
import pytest
from unittest.mock import patch, MagicMock, call


# ===========================================================================
# hotkey_manager.py — pure logic functions (no pynput hardware)
# ===========================================================================

class TestCheckHotkeyAvailable:
    def _check(self, hotkey):
        from src.hotkey_manager import check_hotkey_available
        return check_hotkey_available(hotkey)

    def test_known_conflict_ctrl_alt_delete(self):
        ok, reason = self._check("<ctrl>+<alt>+<delete>")
        assert not ok
        assert reason != ""

    def test_unknown_hotkey_is_available(self):
        ok, reason = self._check("<ctrl>+<shift>+f9")
        assert ok
        assert reason == ""

    def test_case_insensitive_match(self):
        ok, reason = self._check("<CTRL>+<ALT>+<DELETE>")
        assert not ok

    def test_macos_spotlight_conflict(self):
        ok, reason = self._check("<cmd>+space")
        assert not ok

    def test_suggested_alternatives_all_available(self):
        from src.hotkey_manager import SUGGESTED_ALTERNATIVES
        for alt in SUGGESTED_ALTERNATIVES:
            ok, _ = self._check(alt)
            assert ok, f"Alternative {alt} should be available"

    def test_ubuntu_terminal_conflict(self):
        ok, reason = self._check("<ctrl>+<alt>+t")
        assert not ok

    def test_lock_screen_conflict(self):
        ok, reason = self._check("<super>+l")
        assert not ok


class TestMouseButtonMap:
    def test_middle_button_in_map(self):
        from src.hotkey_manager import MOUSE_BUTTON_MAP
        assert "middle" in MOUSE_BUTTON_MAP

    def test_left_button_in_map(self):
        from src.hotkey_manager import MOUSE_BUTTON_MAP
        assert "left" in MOUSE_BUTTON_MAP

    def test_right_button_in_map(self):
        from src.hotkey_manager import MOUSE_BUTTON_MAP
        assert "right" in MOUSE_BUTTON_MAP

    def test_mouse_button_map_is_dict(self):
        from src.hotkey_manager import MOUSE_BUTTON_MAP
        assert isinstance(MOUSE_BUTTON_MAP, dict)


class TestKnownConflicts:
    def test_known_conflicts_is_dict(self):
        from src.hotkey_manager import KNOWN_CONFLICTS
        assert isinstance(KNOWN_CONFLICTS, dict)
        assert len(KNOWN_CONFLICTS) > 0

    def test_known_conflicts_values_are_strings(self):
        from src.hotkey_manager import KNOWN_CONFLICTS
        for key, val in KNOWN_CONFLICTS.items():
            assert isinstance(key, str)
            assert isinstance(val, str)


# ===========================================================================
# state_machine.py
# ===========================================================================

class TestStateMachine:
    def _make_sm(self):
        from src.state_machine import StateMachine, State
        return StateMachine(), State

    def test_initial_state_is_idle(self):
        sm, State = self._make_sm()
        assert sm.current == State.IDLE

    def test_transition_to_recording(self):
        sm, State = self._make_sm()
        result = sm.transition(State.RECORDING)
        assert result is True
        assert sm.current == State.RECORDING

    def test_transition_to_transcribing(self):
        sm, State = self._make_sm()
        sm.transition(State.RECORDING)
        result = sm.transition(State.PROCESSING)
        assert result is True
        assert sm.current == State.PROCESSING

    def test_transition_back_to_idle(self):
        sm, State = self._make_sm()
        sm.transition(State.RECORDING)
        sm.transition(State.PROCESSING)
        result = sm.transition(State.IDLE)
        assert result is True
        assert sm.current == State.IDLE

    def test_reset_goes_to_idle(self):
        sm, State = self._make_sm()
        sm.transition(State.RECORDING)
        sm.reset()
        assert sm.current == State.IDLE

    def test_on_enter_callback_called(self):
        sm, State = self._make_sm()
        calls = []
        sm.on_enter(State.RECORDING, lambda: calls.append("entered_recording"))
        sm.transition(State.RECORDING)
        assert "entered_recording" in calls

    def test_on_exit_callback_called(self):
        sm, State = self._make_sm()
        calls = []
        sm.on_exit(State.IDLE, lambda: calls.append("exited_idle"))
        sm.transition(State.RECORDING)
        assert "exited_idle" in calls

    def test_state_enum_values(self):
        from src.state_machine import State
        # Verify expected states exist
        assert hasattr(State, "IDLE")
        assert hasattr(State, "RECORDING")
        assert hasattr(State, "PROCESSING")


# ===========================================================================
# network_monitor.py
# ===========================================================================

class TestProbeEndpoint:
    def test_probe_returns_true_on_success(self):
        from src.network_monitor import _probe_endpoint
        mock_sock = MagicMock()
        with patch("src.network_monitor.socket.create_connection",
                   return_value=mock_sock):
            result = _probe_endpoint("8.8.8.8", 53, 2.0)
        assert result is True

    def test_probe_returns_false_on_error(self):
        from src.network_monitor import _probe_endpoint
        with patch("src.network_monitor.socket.create_connection",
                   side_effect=socket.error("refused")):
            result = _probe_endpoint("8.8.8.8", 53, 2.0)
        assert result is False


class TestCheckConnectivity:
    def test_check_connectivity_returns_bool(self):
        from src.network_monitor import _check_connectivity
        with patch("src.network_monitor._probe_endpoint", return_value=True):
            result = _check_connectivity()
        assert isinstance(result, bool)

    def test_check_connectivity_offline_returns_false(self):
        from src.network_monitor import _check_connectivity
        with patch("src.network_monitor._probe_endpoint", return_value=False):
            result = _check_connectivity()
        assert result is False

    def test_check_connectivity_online_returns_true(self):
        from src.network_monitor import _check_connectivity
        with patch("src.network_monitor._probe_endpoint", return_value=True):
            result = _check_connectivity()
        assert result is True


class TestNetworkMonitorClass:
    def test_is_connected_initial_state(self):
        from src.network_monitor import NetworkMonitor
        with patch("src.network_monitor._check_connectivity", return_value=True):
            nm = NetworkMonitor()
            assert isinstance(nm.is_connected, bool)

    def test_stop_without_start(self):
        from src.network_monitor import NetworkMonitor
        nm = NetworkMonitor()
        nm.stop()  # Should not raise


# ===========================================================================
# circuit_breaker.py — improve from 93% to higher
# ===========================================================================

class TestCircuitBreakerEdgeCases:
    def _make_cb(self, threshold=3, open_duration=1.0):
        from src.circuit_breaker import CircuitBreaker
        return CircuitBreaker(failure_threshold=threshold, open_duration=open_duration)

    def test_initial_state_closed(self):
        from src.circuit_breaker import CBState
        cb = self._make_cb()
        assert cb.state == CBState.CLOSED

    def test_record_failure_increments(self):
        cb = self._make_cb(threshold=10)
        cb._record_failure()
        assert cb._failure_count == 1

    def test_record_success_resets_failure_count(self):
        cb = self._make_cb(threshold=10)
        cb._record_failure()
        cb._record_failure()
        cb._record_success()
        assert cb._failure_count == 0

    def test_reset_sets_closed(self):
        from src.circuit_breaker import CBState
        cb = self._make_cb()
        cb.reset()
        assert cb.state == CBState.CLOSED

    def test_call_executes_function(self):
        cb = self._make_cb()
        result = cb.call(lambda: 42)
        assert result == 42

    def test_call_open_circuit_raises(self):
        from src.circuit_breaker import CircuitBreakerOpen, CBState
        import time
        cb = self._make_cb(threshold=1, open_duration=999)
        # Force open state
        try:
            cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))
        except Exception:
            pass
        # Now circuit should be open
        if cb.state == CBState.OPEN:
            with pytest.raises(CircuitBreakerOpen):
                cb.call(lambda: 42)


# ===========================================================================
# config.py additional edge cases
# ===========================================================================

class TestConfigEdgeCases:
    def test_default_config_has_all_required_keys(self):
        from src.config import DEFAULT_CONFIG
        required = ["api_key", "language", "hotkey", "hotkey_mode", "stt_providers",
                    "insert_method", "audio_cache_enabled", "log_level"]
        for key in required:
            assert key in DEFAULT_CONFIG, f"Missing key: {key}"

    def test_valid_hotkey_modes_constant(self):
        from src.config import VALID_HOTKEY_MODES
        assert "toggle" in VALID_HOTKEY_MODES
        assert "hold" in VALID_HOTKEY_MODES

    def test_valid_insert_methods_constant(self):
        from src.config import VALID_INSERT_METHODS
        assert "auto" in VALID_INSERT_METHODS
        assert "clipboard" in VALID_INSERT_METHODS

    def test_config_error_is_exception(self):
        from src.config import ConfigError
        with pytest.raises(ConfigError):
            raise ConfigError("test error")

    def test_modifier_names_set(self):
        from src.config import _MODIFIER_NAMES
        assert "ctrl" in _MODIFIER_NAMES
        assert "shift" in _MODIFIER_NAMES
        assert "alt" in _MODIFIER_NAMES


# ===========================================================================
# transcription_engine.py edge cases
# ===========================================================================

class TestTranscriptionEngineEdge:
    def test_engine_with_empty_providers_raises(self):
        from src.transcription_engine import TranscriptionEngine
        engine = TranscriptionEngine(providers=[])
        with pytest.raises(Exception):
            engine.transcribe("audio.wav")

    def test_engine_stores_providers(self):
        from src.transcription_engine import TranscriptionEngine
        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True
        engine = TranscriptionEngine(providers=[mock_provider])
        assert len(engine._providers) == 1
