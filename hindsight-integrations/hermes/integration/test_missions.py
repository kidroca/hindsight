"""Demand-driven mission contracts, using a recording client only."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import hindsight_hermes as hindsight
import pytest


@pytest.mark.parametrize("model", [False, True])
def test_mission_drift_cached_and_identity_change(model):
    p = hindsight.HindsightMemoryProvider()
    p._bank_mission = "reflect purpose"
    p._bank_retain_mission = "   "
    config = {"reflect_mission": "old", "retain_mission": "remote-owned"}
    response = SimpleNamespace(config=SimpleNamespace(**config)) if model else {"config": config}
    banks = SimpleNamespace(get_bank_config=AsyncMock(return_value=response), update_bank_config=AsyncMock())
    p._client = SimpleNamespace(banks=banks)
    operation = AsyncMock(return_value="ok")
    assert banks.get_bank_config.call_count == 0
    assert p._run_hindsight_operation(operation) == "ok"
    assert banks.update_bank_config.call_args.args[1].updates == {"reflect_mission": "reflect purpose"}
    p._run_hindsight_operation(operation)
    assert banks.get_bank_config.call_count == 1
    p._bank_id = "other"
    p._run_hindsight_operation(operation)
    assert banks.get_bank_config.call_count == 2
    p._bank_mission = "new mission"
    p._run_hindsight_operation(operation)
    assert banks.get_bank_config.call_count == 3


def test_missions_follow_captured_bank_after_session_rotation():
    p = hindsight.HindsightMemoryProvider()
    p._bank_mission = "managed"
    p._bank_id = "new-session-bank"
    banks = SimpleNamespace(get_bank_config=AsyncMock(return_value={"config": {}}), update_bank_config=AsyncMock())
    p._client = SimpleNamespace(
        banks=banks,
        aretain_batch=AsyncMock(return_value=SimpleNamespace()),
        arecall=AsyncMock(return_value=SimpleNamespace(results=[])),
        areflect=AsyncMock(return_value=SimpleNamespace(text="ok")),
    )
    p._retain_batch({"content": "queued old turn"}, bank_id="old-session-bank", retain_async=True)
    assert banks.update_bank_config.call_args.args[0] == "old-session-bank"
    p._recall("old prefetch", bank_id="old-prefetch-bank")
    assert banks.update_bank_config.call_args.args[0] == "old-prefetch-bank"
    p._reflect("old prefetch", bank_id="old-reflect-bank")
    assert banks.update_bank_config.call_args.args[0] == "old-reflect-bank"
    assert p._bank_id == "new-session-bank"
    p._api_url = "https://other.example.invalid"
    p._run_hindsight_operation(AsyncMock(), bank_id="old-reflect-bank")
    assert banks.get_bank_config.call_count == 4


def test_mission_failure_retries_only_on_operation(caplog):
    p = hindsight.HindsightMemoryProvider()
    p._bank_mission = "wanted"
    banks = SimpleNamespace(
        get_bank_config=AsyncMock(
            side_effect=[RuntimeError("offline"), RuntimeError("offline"), {"config": {"reflect_mission": "wanted"}}]
        ),
        update_bank_config=AsyncMock(),
    )
    p._client = SimpleNamespace(banks=banks)
    operation = AsyncMock(return_value="still works")
    for _ in range(3):
        assert p._run_hindsight_operation(operation) == "still works"
    assert banks.get_bank_config.call_count == 3
    assert banks.update_bank_config.call_count == 0
    assert sum("mission synchronization failed" in r.message for r in caplog.records) == 1
    p._run_hindsight_operation(operation)
    assert banks.get_bank_config.call_count == 3


def test_blank_missions_no_requests_and_embedded_reconnect(monkeypatch):
    p = hindsight.HindsightMemoryProvider()
    p._mode = "local_embedded"
    p._bank_mission = "  "
    first, second = SimpleNamespace(), SimpleNamespace()
    clients = iter([first, second])
    monkeypatch.setattr(p, "_new_embedded_client", lambda: next(clients))
    operation = AsyncMock(side_effect=[ConnectionError("connection refused"), "reconnected"])
    assert p._run_hindsight_operation(operation) == "reconnected"
    assert operation.call_args_list[0].args == (first,)
    assert operation.call_args_list[1].args == (second,)


def test_both_missions_update_failure_retries_on_next_operation(caplog):
    p = hindsight.HindsightMemoryProvider()
    p._bank_mission = "reflect purpose"
    p._bank_retain_mission = "retain purpose"
    banks = SimpleNamespace(
        get_bank_config=AsyncMock(return_value={"config": {"reflect_mission": "old", "retain_mission": "old"}}),
        update_bank_config=AsyncMock(side_effect=[RuntimeError("unavailable"), RuntimeError("unavailable"), None]),
    )
    p._client = SimpleNamespace(banks=banks)
    operation = AsyncMock(return_value="operation succeeds")
    for attempt in range(3):
        assert p._run_hindsight_operation(operation) == "operation succeeds"
        assert banks.update_bank_config.call_count == attempt + 1
    for call in banks.update_bank_config.call_args_list:
        assert call.args[0] == p._bank_id
        assert call.args[1].updates == {"reflect_mission": "reflect purpose", "retain_mission": "retain purpose"}
    assert sum("mission synchronization failed" in r.message for r in caplog.records) == 1
    p._run_hindsight_operation(operation)
    assert banks.get_bank_config.call_count == 3
    assert banks.update_bank_config.call_count == 3
    assert operation.call_count == 4


def test_missions_retry_after_embedded_reconnect(monkeypatch):
    p = hindsight.HindsightMemoryProvider()
    p._mode = "local_embedded"
    p._bank_mission = "reflect purpose"
    p._bank_retain_mission = "retain purpose"
    first = SimpleNamespace(
        banks=SimpleNamespace(
            get_bank_config=AsyncMock(side_effect=ConnectionError("connection refused")), update_bank_config=AsyncMock()
        )
    )
    second = SimpleNamespace(
        banks=SimpleNamespace(get_bank_config=AsyncMock(return_value={"config": {}}), update_bank_config=AsyncMock())
    )
    clients = iter([first, second])
    monkeypatch.setattr(p, "_new_embedded_client", lambda: next(clients))
    operation = AsyncMock(side_effect=[ConnectionError("connection refused"), "reconnected"])
    assert p._run_hindsight_operation(operation) == "reconnected"
    assert [call.args[0] for call in operation.call_args_list] == [first, second]
    # Failed mission synchronization retries only on a later real operation,
    # not during the existing one-shot reconnect recovery.
    second.banks.get_bank_config.assert_not_called()
    operation.side_effect = None
    operation.return_value = "next operation"
    assert p._run_hindsight_operation(operation) == "next operation"
    assert second.banks.update_bank_config.call_args.args[1].updates == {
        "reflect_mission": "reflect purpose",
        "retain_mission": "retain purpose",
    }
    p._run_hindsight_operation(operation)
    second.banks.get_bank_config.assert_awaited_once()
