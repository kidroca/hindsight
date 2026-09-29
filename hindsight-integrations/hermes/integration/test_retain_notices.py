"""Retain notices exercise the real writer with recall disabled and no live API."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from agent.agent_init import _memory_provider_init_kwargs
from hindsight_hermes import HindsightMemoryProvider


def test_real_candidate_notice_callbacks_and_memory_manager(monkeypatch):
    import hindsight_hermes as module
    from agent.agent_init import _GATEWAY_IDENTITY_PARAMS, _init_memory
    from agent.status_output import StatusOutputMixin

    class Agent(StatusOutputMixin):
        pass

    a = Agent()
    a.session_id, a.session_cwd, a._session_db = "actual-callbacks", None, None
    a.enabled_toolsets, a.disabled_toolsets, a.tools = [], [], []
    for key in _GATEWAY_IDENTITY_PARAMS:
        setattr(a, "_" + key, "")
    notices, clears = [], []
    a.notice_callback, a.notice_clear_callback = notices.append, clears.append
    monkeypatch.setattr(
        module,
        "_load_config",
        lambda: {
            "mode": "local_external",
            "auto_recall": False,
            "auto_retain": True,
        },
    )
    p = HindsightMemoryProvider()
    monkeypatch.setattr("plugins.memory.load_memory_provider", lambda name: p)
    _init_memory(
        a, {"memory": {"provider": "hindsight", "memory_enabled": False, "user_profile_enabled": False}}, False, "tui"
    )
    assert a._memory_manager.providers == [p]
    p._client = Client()
    monkeypatch.setattr(p, "_resolve_retain_target", lambda fallback: ("s", "append"))
    try:
        fail(p)
        assert notices[0].kind == "sticky"
        assert p._notice_callback == a._emit_notice
        p._retain_async = False
        turn(p)
        assert clears == ["hindsight.retain"]
        assert notices[-1].kind == "ttl"
        assert not p._auto_recall and p._auto_retain
        assert "Automatic recall is disabled" in p.system_prompt_block()
    finally:
        p.shutdown()


class Client:
    def __init__(self):
        self.reply = SimpleNamespace(operation_ids=["a", "b"])
        self.error = None
        self.statuses = {}
        self.checks = []
        self.operations = self

    async def aretain_batch(self, **kwargs):
        if self.error:
            raise self.error
        return self.reply

    async def get_operation_status(self, bank_id, operation_id):
        self.checks.append(operation_id)
        status = self.statuses.get(operation_id, "pending")
        if isinstance(status, Exception):
            raise status
        return {"status": status}

    async def aclose(self):
        pass


@pytest.fixture
def provider(monkeypatch, tmp_path):
    import hindsight_hermes as module

    monkeypatch.setattr(module, "_warn_if_client_outdated", lambda: None)
    monkeypatch.setattr(
        module,
        "_load_config",
        lambda: {
            "mode": "local_external",
            "auto_recall": False,
            "auto_retain": True,
            "retain_async": True,
        },
    )
    p = HindsightMemoryProvider()
    notices, clears = [], []
    p.initialize("s", notice_callback=notices.append, notice_clear_callback=clears.append)
    p._client = Client()
    monkeypatch.setattr(p, "_resolve_retain_target", lambda fallback: ("s", "append"))
    yield p, notices, clears
    p.shutdown()


def turn(p):
    p.sync_turn("user", "assistant")
    p._retain_queue.join()
    assert p._writer_thread.is_alive()


def fail(p):
    p._client.error = RuntimeError("retain unavailable")
    turn(p)
    p._client.error = None


@pytest.mark.parametrize("platform", ["cli", "telegram", "tui", "acp"])
def test_agent_init_forwards_notice_callbacks(platform):
    a = Mock(session_id="s", session_cwd="/remote/work", _session_db=None)
    kwargs = _memory_provider_init_kwargs(a, platform)
    assert kwargs["notice_callback"] == a._emit_notice
    assert kwargs["notice_clear_callback"] == a._emit_notice_clear


def test_same_operation_id_in_different_banks_is_not_conflated(provider, monkeypatch):
    p, notices, clears = provider
    fail(p)
    p._track_retain_ops({"operation_id": "same"}, "old-bank")
    p._track_retain_ops({"operation_ids": ["same"]}, "new-bank")
    assert p._pending_retain_ops == {("old-bank", "same"), ("new-bank", "same")}
    checked = []

    def status(bank, op, **kwargs):
        checked.append((bank, op))
        return "completed" if bank == "old-bank" else "pending"

    monkeypatch.setattr(p, "_retain_op_status", status)
    p._observe_retain_ops()
    assert set(checked) == {("old-bank", "same"), ("new-bank", "same")}
    assert p._pending_retain_ops == {("new-bank", "same")}
    assert not clears
    monkeypatch.setattr(p, "_retain_op_status", lambda *a, **kw: "completed")
    p._observe_retain_ops()
    assert clears == ["hindsight.retain"]


def test_stock_host_without_notice_type_keeps_retention_usable(provider, monkeypatch):
    import sys

    p, notices, clears = provider
    p._notice_callback = p._notice_clear_callback = None
    monkeypatch.setitem(sys.modules, "agent.credits_tracker", None)
    fail(p)
    p._retain_async = False
    turn(p)
    assert not p._retain_failed
    assert not notices and not clears


def test_host_missing_notice_type_with_callbacks_is_nonfatal(provider, monkeypatch):
    import sys

    p, notices, clears = provider
    monkeypatch.setitem(sys.modules, "agent.credits_tracker", None)
    fail(p)
    p._retain_async = False
    turn(p)
    assert not p._retain_failed
    assert not notices and not clears


def test_writer_deduplicates_failure_and_sync_recovery(provider):
    p, notices, clears = provider
    fail(p)
    fail(p)
    assert len(notices) == 1 and notices[0].kind == "sticky"
    p._retain_async = False
    turn(p)
    assert clears == [notices[0].key]
    assert len(notices) == 2 and "writing works again" in notices[1].text.lower()
    assert "restored" not in notices[1].text.lower()


@pytest.mark.parametrize("status", ["completed", "failed", "unknown", "", "pending", "404", "error", "timeout"])
def test_demand_observation_requires_entire_group_completed(provider, status):
    from hindsight_client_api.exceptions import NotFoundException

    p, notices, clears = provider
    fail(p)
    turn(p)
    assert not clears
    assert not p._client.checks
    p._client.statuses = {
        "a": "completed",
        "b": {
            "404": NotFoundException(),
            "error": RuntimeError("status unavailable"),
            "timeout": TimeoutError(),
        }.get(status, status),
    }
    p._client.reply = SimpleNamespace()
    turn(p)
    assert set(p._client.checks) == {"a", "b"}
    assert bool(clears) is (status == "completed")


def test_failed_switch_flush_and_throwing_callbacks_do_not_kill_writer(provider):
    p, notices, clears = provider
    p._retain_every_n_turns = 3
    p.sync_turn("buffered", "turn")
    p._client.error = RuntimeError("switch failed")
    p.on_session_switch("new")
    p._retain_queue.join()
    assert len(notices) == 1

    def broken(*args):
        raise RuntimeError("driver gone")

    p._notice_callback = p._notice_clear_callback = broken
    p._client.error = None
    p._retain_async = False
    p._retain_every_n_turns = 1
    turn(p)
    fail(p)
    turn(p)


def test_agent_memory_initialization_reaches_real_writer(provider, monkeypatch):
    from agent.agent_init import _init_memory

    p, notices, clears = provider
    a = Mock(
        session_id="s",
        session_cwd="/remote/work",
        _session_db=None,
        enabled_toolsets=[],
        disabled_toolsets=[],
        tools=[],
    )
    a._emit_notice.side_effect = notices.append
    monkeypatch.setattr("plugins.memory.load_memory_provider", lambda name: p)
    _init_memory(
        a,
        {"memory": {"provider": "hindsight", "memory_enabled": False, "user_profile_enabled": False}},
        False,
        "telegram",
    )
    assert a._memory_manager.providers == [p]
    fail(p)
    assert len(notices) == 1
    a._emit_notice.assert_called_once()


def test_status_timeout_is_bounded_and_writer_continues(provider):
    import asyncio

    p, notices, clears = provider
    fail(p)
    turn(p)

    async def stuck(**kwargs):
        await asyncio.Event().wait()

    p._client.get_operation_status = stuck
    p._prefetch_retain_drain_timeout = 0.02
    finished = threading.Event()
    p.sync_turn("next", "turn")
    p._retain_queue.put(finished.set)
    assert finished.wait(2), "status observation must not stall the writer"
    assert not clears


def test_failed_member_prevents_later_partial_group_recovery(provider):
    p, notices, clears = provider
    turn(p)
    p._client.statuses = {"a": "failed", "b": "pending"}
    p._client.reply = SimpleNamespace()
    turn(p)
    assert len(notices) == 1
    p._client.statuses["b"] = "completed"
    turn(p)
    assert not clears


def test_timeout_drops_visibility_wait_without_recovery(provider):
    p, notices, clears = provider
    fail(p)
    turn(p)
    assert not p._wait_for_server_retain_ops(lambda: True, 0.001)
    assert not clears
    assert not p._pending_retain_ops


@pytest.mark.parametrize("status", ["completed", "failed", "404"])
def test_terminality_releases_wait_but_only_completed_recovers(provider, status):
    from hindsight_client_api.exceptions import NotFoundException

    p, notices, clears = provider
    fail(p)
    turn(p)
    p._client.statuses = {"a": "completed", "b": NotFoundException() if status == "404" else status}
    assert p._wait_for_server_retain_ops(lambda: False, 1)
    assert not p._pending_retain_ops
    assert bool(clears) is (status == "completed")


def test_older_async_completion_cannot_clear_newer_failure(provider):
    p, notices, clears = provider
    turn(p)
    fail(p)
    p._client.statuses = {"a": "completed", "b": "completed"}
    p._client.reply = SimpleNamespace()
    p._tool_retain({"content": "next real activity"})
    assert not clears
    assert len(notices) == 1
    p._retain_async = False
    turn(p)
    assert len(clears) == 1


def test_inflight_sync_writer_cannot_clear_later_tool_failure(provider):
    p, notices, clears = provider
    started, release = threading.Event(), threading.Event()
    p._retain_async = False

    async def retain(**kwargs):
        if kwargs["items"][0]["content"] == "failure":
            raise RuntimeError("later tool failure")
        started.set()
        import asyncio

        assert await asyncio.to_thread(release.wait, 3)
        return SimpleNamespace()

    p._client.aretain_batch = retain
    p.sync_turn("old", "success")
    assert started.wait(3)
    try:
        with pytest.raises(RuntimeError):
            p._tool_retain({"content": "failure"})
    finally:
        release.set()
    p._retain_queue.join()
    assert not clears
    assert len(notices) == 1
    turn(p)
    assert len(clears) == 1


@pytest.mark.parametrize("newer_failure", [False, True])
@pytest.mark.parametrize("callback_raises", [False, True])
def test_warning_callback_sync_reentry_orders_recovery(provider, newer_failure, callback_raises):
    p, notices, clears = provider
    sticky = set()
    reentered = []
    p._retain_async = False

    def notice(value):
        if value.kind == "sticky" and not reentered:
            reentered.append(True)
            p._client.error = None
            p._retain_batch({"content": "nested synchronous recovery"}, bank_id=p._bank_id, retain_async=False)
            if newer_failure:
                p._client.error = RuntimeError("newer nested failure")
                try:
                    p._tool_retain({"content": "nested failure"})
                except RuntimeError:
                    pass
                finally:
                    p._client.error = None
            sticky.add(value.key)
        notices.append(value)
        if callback_raises:
            raise RuntimeError("driver failed after storing notice")

    def clear(key):
        sticky.discard(key)
        clears.append(key)
        if callback_raises:
            raise RuntimeError("driver failed after clearing notice")

    p._notice_callback = notice
    p._notice_clear_callback = clear
    fail(p)
    assert reentered == [True]
    assert bool(sticky) is newer_failure
    assert p._retain_failed is newer_failure
    assert [n.kind for n in notices] == (["sticky"] if newer_failure else ["sticky", "ttl"])
    assert len(clears) == (0 if newer_failure else 1)
    turn(p)
    assert not sticky
    assert not p._retain_failed
    assert len(clears) == 1


def test_writer_warning_delivery_serializes_tool_recovery(provider, monkeypatch):
    p, notices, clears = provider
    paused, release, attempted, finished = (threading.Event() for _ in range(4))
    original_notice = p._retain_notice
    original_observe = p._observe_retain_ops
    sticky = set()

    def notice(callback, value):
        if getattr(value, "kind", None) == "sticky":
            p._client.error = None
            p._retain_batch({"content": "post-failure acceptance"}, bank_id=p._bank_id, retain_async=True)
            p._client.statuses = {"a": "completed", "b": "completed"}
            paused.set()
            assert release.wait(3)
            sticky.add(value.key)
        original_notice(callback, value)

    def clear(key):
        sticky.discard(key)
        clears.append(key)

    def observe(*args, **kwargs):
        if threading.current_thread().name == "explicit-retain":
            attempted.set()
        return original_observe(*args, **kwargs)

    monkeypatch.setattr(p, "_retain_notice", notice)
    monkeypatch.setattr(p, "_observe_retain_ops", observe)
    p._notice_clear_callback = clear
    p._client.error = RuntimeError("writer failure")
    p.sync_turn("failed", "turn")
    assert paused.wait(3)

    def tool():
        try:
            p._tool_retain({"content": "recovery"})
        finally:
            finished.set()

    worker = threading.Thread(target=tool, name="explicit-retain")
    worker.start()
    try:
        assert attempted.wait(3)
        assert not finished.wait(0.1)
    finally:
        release.set()
        worker.join(3)
    p._retain_queue.join()
    assert not worker.is_alive()
    assert not sticky
    assert len(clears) == 1
