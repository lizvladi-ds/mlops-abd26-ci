"""Отдельный интеграционный тест; обычный pytest его не собирает."""
import os
import pytest

@pytest.fixture
def connection():
    import psycopg
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            # Временная таблица изолирована в пределах подключения теста.
            cur.execute("CREATE TEMP TABLE predictions (id INTEGER, label INTEGER)")
        yield conn

def test_prediction_can_be_saved_and_read(connection):
    with connection.cursor() as cur:
        cur.execute("INSERT INTO predictions (id, label) VALUES (%s, %s)", (1, 2))
        cur.execute("SELECT label FROM predictions WHERE id = %s", (1,))
        assert cur.fetchone() == (2,)
