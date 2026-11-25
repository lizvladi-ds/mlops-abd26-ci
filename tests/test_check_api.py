"""Сброс соединения при старте API не должен немедленно ломать проверку."""
import io
import sys
from http.client import RemoteDisconnected
import pytest
import check_api


@pytest.mark.parametrize("startup_error", [
    ConnectionResetError("Server is starting"),
    RemoteDisconnected("Server closed connection before replying"),
])
def test_check_retries_startup_connection_error(monkeypatch, capsys, startup_error):
    attempts = []
    responses = iter([
        io.BytesIO(b'{"status": "ok"}'),
        io.BytesIO(b'{"predictions": [0]}'),
    ])

    def fake_urlopen(request, timeout):
        attempts.append(request)
        if len(attempts) == 1:
            raise startup_error
        return next(responses)

    monkeypatch.setattr(check_api, "urlopen", fake_urlopen)
    monkeypatch.setattr(check_api.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(sys, "argv", ["check_api.py"])
    check_api.main()
    assert len(attempts) == 3
    assert "HTTP smoke check passed" in capsys.readouterr().out
