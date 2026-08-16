from unittest.mock import patch

from config import Config
from src.log.Logger import Logger


def test_info_level_hides_debug_from_stdout_and_file(tmp_path, capsys):
    logger = Logger(log_dir=tmp_path, log_file="app.log", worker_id=1, level="info")

    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")

    output = capsys.readouterr().out
    content = (tmp_path / "app.log").read_text(encoding="utf-8")
    assert "debug message" not in output
    assert "debug message" not in content
    for message in ("info message", "warning message", "error message"):
        assert message in output
        assert message in content


def test_debug_level_writes_debug_messages(tmp_path, capsys):
    logger = Logger(log_dir=tmp_path, log_file="app.log", level="DEBUG")

    logger.debug("debug message")

    assert "debug message" in capsys.readouterr().out
    assert "debug message" in (tmp_path / "app.log").read_text(encoding="utf-8")


def test_off_hides_routine_messages_but_keeps_warnings_and_errors(tmp_path, capsys):
    logger = Logger(log_dir=tmp_path, log_file="app.log", level="OFF")

    logger.debug("debug message")
    logger.info("info message")
    assert capsys.readouterr().out == ""
    assert not (tmp_path / "app.log").exists()

    logger.warning("warning message")
    logger.error("error message")

    output = capsys.readouterr().out
    content = (tmp_path / "app.log").read_text(encoding="utf-8")
    assert "warning message" in output
    assert "error message" in output
    assert "warning message" in content
    assert "error message" in content


def test_invalid_level_defaults_to_info(tmp_path, capsys):
    logger = Logger(log_dir=tmp_path, log_file="app.log", level="verbose")

    logger.debug("debug message")
    logger.info("info message")

    assert "debug message" not in capsys.readouterr().out
    assert "info message" in (tmp_path / "app.log").read_text(encoding="utf-8")


def test_logger_uses_configured_level_when_not_explicitly_provided(tmp_path):
    with patch.object(Config, "get_nested", return_value="debug"):
        logger = Logger(log_dir=tmp_path, log_file="app.log")

    assert logger.level == "DEBUG"
