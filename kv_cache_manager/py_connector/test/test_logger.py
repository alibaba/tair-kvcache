"""Unit tests for KVCM logger configuration."""

import importlib
import logging
import os
import unittest
from unittest.mock import patch


class TestSetLogLevel(unittest.TestCase):
    """Tests for set_log_level() function."""

    def setUp(self):
        # Re-import logger module for each test to get fresh state
        import kv_cache_manager.py_connector.common.logger as logger_mod
        self.logger_mod = logger_mod

    def test_set_debug_level(self):
        self.logger_mod.set_log_level("DEBUG")
        self.assertEqual(self.logger_mod.logger.level, logging.DEBUG)

    def test_set_info_level(self):
        self.logger_mod.set_log_level("INFO")
        self.assertEqual(self.logger_mod.logger.level, logging.INFO)

    def test_set_warning_level(self):
        self.logger_mod.set_log_level("WARNING")
        self.assertEqual(self.logger_mod.logger.level, logging.WARNING)

    def test_set_error_level(self):
        self.logger_mod.set_log_level("ERROR")
        self.assertEqual(self.logger_mod.logger.level, logging.ERROR)

    def test_set_case_insensitive(self):
        self.logger_mod.set_log_level("debug")
        self.assertEqual(self.logger_mod.logger.level, logging.DEBUG)

    def test_invalid_level_falls_back_to_warning(self):
        self.logger_mod.set_log_level("INVALID")
        self.assertEqual(self.logger_mod.logger.level, logging.WARNING)

    def test_empty_string_falls_back_to_warning(self):
        self.logger_mod.set_log_level("")
        self.assertEqual(self.logger_mod.logger.level, logging.WARNING)

    def tearDown(self):
        self.logger_mod.logger.setLevel(logging.WARNING)


class TestEnvironmentVariable(unittest.TestCase):
    """Tests for KVCM_LOG_LEVEL environment variable."""

    def test_env_var_debug(self):
        with patch.dict(os.environ, {"KVCM_LOG_LEVEL": "DEBUG"}):
            import kv_cache_manager.py_connector.common.logger as logger_mod
            importlib.reload(logger_mod)
            self.assertEqual(logger_mod.logger.level, logging.DEBUG)

    def test_env_var_info(self):
        with patch.dict(os.environ, {"KVCM_LOG_LEVEL": "INFO"}):
            import kv_cache_manager.py_connector.common.logger as logger_mod
            importlib.reload(logger_mod)
            self.assertEqual(logger_mod.logger.level, logging.INFO)

    def test_env_var_not_set_defaults_to_warning(self):
        env = os.environ.copy()
        env.pop("KVCM_LOG_LEVEL", None)
        with patch.dict(os.environ, env, clear=True):
            import kv_cache_manager.py_connector.common.logger as logger_mod
            importlib.reload(logger_mod)
            self.assertEqual(logger_mod.logger.level, logging.WARNING)

    def test_env_var_invalid_falls_back_to_warning(self):
        with patch.dict(os.environ, {"KVCM_LOG_LEVEL": "GARBAGE"}):
            import kv_cache_manager.py_connector.common.logger as logger_mod
            importlib.reload(logger_mod)
            self.assertEqual(logger_mod.logger.level, logging.WARNING)

    def tearDown(self):
        # Restore default level after env var tests
        import kv_cache_manager.py_connector.common.logger as logger_mod
        importlib.reload(logger_mod)


class TestConfigureLogLevel(unittest.TestCase):
    """Tests for configure_log_level() priority logic."""

    def setUp(self):
        import kv_cache_manager.py_connector.common.logger as logger_mod
        self.logger_mod = logger_mod

    def test_env_var_takes_priority_over_param(self):
        with patch.dict(os.environ, {"KVCM_LOG_LEVEL": "DEBUG"}):
            self.logger_mod.configure_log_level("INFO")
            self.assertEqual(self.logger_mod.logger.level, logging.DEBUG)

    def test_param_used_when_env_var_not_set(self):
        env = os.environ.copy()
        env.pop("KVCM_LOG_LEVEL", None)
        with patch.dict(os.environ, env, clear=True):
            self.logger_mod.configure_log_level("INFO")
            self.assertEqual(self.logger_mod.logger.level, logging.INFO)

    def test_default_when_neither_set(self):
        env = os.environ.copy()
        env.pop("KVCM_LOG_LEVEL", None)
        with patch.dict(os.environ, env, clear=True):
            self.logger_mod.configure_log_level("")
            self.assertEqual(self.logger_mod.logger.level, logging.WARNING)

    def test_none_param_treated_as_empty(self):
        env = os.environ.copy()
        env.pop("KVCM_LOG_LEVEL", None)
        with patch.dict(os.environ, env, clear=True):
            self.logger_mod.configure_log_level(None)
            self.assertEqual(self.logger_mod.logger.level, logging.WARNING)

    def tearDown(self):
        import kv_cache_manager.py_connector.common.logger as logger_mod
        importlib.reload(logger_mod)


if __name__ == "__main__":
    unittest.main()
