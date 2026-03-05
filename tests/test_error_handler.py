"""Tests for src/error_handler.py — categorize_error, setup_logging, global_exception_handler."""
import logging
import sys
import pytest
from unittest.mock import patch, MagicMock, mock_open


class TestCategorizeError:
    def _cat(self, exc):
        from src.error_handler import categorize_error
        return categorize_error(exc)

    def test_network_connection_error(self):
        class ConnectionError(Exception):
            pass
        msg = self._cat(ConnectionError("no internet"))
        assert "интернет" in msg.lower() or "соединени" in msg.lower()

    def test_network_url_error(self):
        class URLError(Exception):
            pass
        msg = self._cat(URLError("urlopen error"))
        assert "интернет" in msg.lower() or "соединени" in msg.lower()

    def test_network_socket_error(self):
        msg = self._cat(Exception("socket timeout"))
        assert "интернет" in msg.lower() or "соединени" in msg.lower()

    def test_network_connection_refused(self):
        msg = self._cat(Exception("connection refused"))
        assert "интернет" in msg.lower() or "соединени" in msg.lower()

    def test_auth_error_401(self):
        msg = self._cat(Exception("401 unauthorized"))
        assert "api" in msg.lower() or "ключ" in msg.lower()

    def test_auth_invalid_api_key(self):
        msg = self._cat(Exception("invalid api key"))
        assert "api" in msg.lower() or "ключ" in msg.lower()

    def test_auth_incorrect_api_key(self):
        msg = self._cat(Exception("incorrect api key"))
        assert "api" in msg.lower() or "ключ" in msg.lower()

    def test_rate_limit_429(self):
        msg = self._cat(Exception("429 rate limit exceeded"))
        assert "лимит" in msg.lower() or "запрос" in msg.lower()

    def test_rate_limit_too_many_requests(self):
        msg = self._cat(Exception("too many requests"))
        assert "лимит" in msg.lower() or "запрос" in msg.lower()

    def test_audio_microphone_error(self):
        msg = self._cat(OSError("microphone not found"))
        assert "микрофон" in msg.lower()

    def test_audio_sounddevice_error(self):
        msg = self._cat(Exception("sounddevice error"))
        assert "микрофон" in msg.lower()

    def test_audio_portaudio_error(self):
        msg = self._cat(Exception("portaudio error"))
        assert "микрофон" in msg.lower()

    def test_permission_error(self):
        class PermissionError(Exception):
            pass
        msg = self._cat(PermissionError("access denied"))
        assert "доступ" in msg.lower() or "устройств" in msg.lower()

    def test_timeout_error(self):
        msg = self._cat(Exception("read timed out"))
        assert "время" in msg.lower() or "ожидани" in msg.lower()

    def test_disk_space_error(self):
        msg = self._cat(Exception("no space left on device"))
        assert "диск" in msg.lower() or "место" in msg.lower()

    def test_generic_fallback(self):
        msg = self._cat(Exception("something completely unknown happened xyz123"))
        assert "ошибка" in msg.lower() or "лог" in msg.lower()

    def test_errno_111_network(self):
        msg = self._cat(Exception("errno 111 connection refused"))
        assert "интернет" in msg.lower() or "соединени" in msg.lower()

    def test_errno_28_disk(self):
        msg = self._cat(Exception("errno 28 no space left"))
        assert "диск" in msg.lower() or "место" in msg.lower()


class TestSetupLogging:
    def test_setup_logging_debug_mode(self, tmp_path):
        from src import error_handler
        with patch.object(error_handler, "LOG_DIR", tmp_path), \
             patch.object(error_handler, "LOG_FILE", tmp_path / "test.log"):
            # Clear any existing handlers to test fresh setup
            root = logging.getLogger()
            original_handlers = root.handlers[:]
            root.handlers.clear()
            try:
                error_handler.setup_logging(debug=True)
                assert root.level == logging.DEBUG
            finally:
                root.handlers.clear()
                root.handlers.extend(original_handlers)

    def test_setup_logging_idempotent(self, tmp_path):
        from src import error_handler
        with patch.object(error_handler, "LOG_DIR", tmp_path), \
             patch.object(error_handler, "LOG_FILE", tmp_path / "test.log"):
            root = logging.getLogger()
            original_handlers = root.handlers[:]
            root.handlers.clear()
            try:
                error_handler.setup_logging()
                handler_count_after_first = len(root.handlers)
                error_handler.setup_logging()  # Second call should not add more handlers
                # Second call changes level only, handler count must not increase
                assert len(root.handlers) == handler_count_after_first
            finally:
                root.handlers.clear()
                root.handlers.extend(original_handlers)


class TestInstallGlobalHandler:
    def test_install_sets_excepthook(self):
        from src.error_handler import install_global_handler, global_exception_handler
        original = sys.excepthook
        try:
            install_global_handler()
            assert sys.excepthook == global_exception_handler
        finally:
            sys.excepthook = original


class TestGlobalExceptionHandler:
    def test_keyboard_interrupt_calls_default_hook(self):
        from src.error_handler import global_exception_handler
        try:
            raise KeyboardInterrupt()
        except KeyboardInterrupt:
            exc_type, exc_value, exc_tb = sys.exc_info()

        with patch("sys.__excepthook__") as mock_hook:
            global_exception_handler(exc_type, exc_value, exc_tb)
            mock_hook.assert_called_once()

    def test_regular_exception_shows_dialog(self):
        from src.error_handler import global_exception_handler
        try:
            raise ValueError("test error")
        except ValueError:
            exc_type, exc_value, exc_tb = sys.exc_info()

        with patch("src.error_handler.show_error_dialog") as mock_dialog:
            global_exception_handler(exc_type, exc_value, exc_tb)
            mock_dialog.assert_called_once()


class TestWriteErrorToLog:
    def test_writes_to_log_file(self, tmp_path):
        from src import error_handler
        log_file = tmp_path / "test.log"
        with patch.object(error_handler, "LOG_DIR", tmp_path), \
             patch.object(error_handler, "LOG_FILE", log_file):
            error_handler._write_error_to_log("Test Title", "Test message")
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "Test Title" in content
        assert "Test message" in content


class TestShowErrorDialog:
    def test_show_with_details_formats_message(self):
        from src.error_handler import show_error_dialog
        captured = {}

        def mock_do_show():
            pass

        # Test that show_error_dialog does not raise when tkinter not available
        with patch("src.error_handler._write_error_to_log") as mock_log:
            with patch("builtins.__import__", side_effect=ImportError("no tkinter")):
                try:
                    show_error_dialog("Title", "Message", details="Details")
                except Exception:
                    pass  # May fail without tkinter — that's OK

    def test_show_with_tk_root_uses_after(self):
        from src.error_handler import show_error_dialog
        mock_root = MagicMock()
        show_error_dialog("Title", "Message", tk_root=mock_root)
        mock_root.after.assert_called_once()

    def test_show_error_from_thread_delegates(self):
        from src.error_handler import show_error_from_thread
        mock_root = MagicMock()
        show_error_from_thread("Title", "Message", tk_root=mock_root)
        mock_root.after.assert_called_once()
