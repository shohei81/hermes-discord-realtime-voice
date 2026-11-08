"""Wiring tests against the installed Hermes: compatibility checks, the wrapped Discord adapter
and gateway runner methods, delegation, voice threads and voice follow."""

import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import discord_realtime_voice
from discord_realtime_voice import integration as integ
from discord_realtime_voice import live_voice as lv
from discord_realtime_voice import realtime_voice as rv


def make_plugin(**overrides):
    values = dict(integ.Settings.DEFAULTS, **overrides)
    return integ.RealtimeVoice(integ.Settings(lambda key, default: values.get(key, default)))


@pytest.fixture(scope="module")
def installed():
    plugin = make_plugin()
    assert plugin.install() is True
    return plugin


def bare_adapter(**attrs):
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = object.__new__(DiscordAdapter)
    defaults = dict(
        _client=None, _voice_clients={}, _voice_receivers={}, _voice_listen_tasks={},
        _voice_text_channels={}, _voice_sources={}, _voice_input_callback=None,
        _on_voice_disconnect=None, _voice_locks={}, _voice_timeout_tasks={}, _voice_mixers={},
    )
    defaults.update(attrs)
    for k, v in defaults.items():
        setattr(adapter, k, v)
    return adapter


# ── compatibility ────────────────────────────────────────────────────────


class TestCompatibility:
    def test_this_hermes_has_everything(self):
        assert integ.missing_requirements() == []

    def test_unknown_hermes_leaves_plugin_inactive(self, monkeypatch, caplog):
        from plugins.platforms.discord.adapter import DiscordAdapter

        before = DiscordAdapter.__dict__["play_in_voice_channel"]
        monkeypatch.setattr(integ, "REQUIRED", integ.REQUIRED + (("gateway.run_voice", "Nope.gone"),))
        plugin = make_plugin()
        assert plugin.install() is False
        assert "gateway.run_voice:Nope.gone" in caplog.text
        assert DiscordAdapter.__dict__["play_in_voice_channel"] is before

    def test_install_wraps_once(self, installed):
        from plugins.platforms.discord.adapter import DiscordAdapter

        wrapped = DiscordAdapter.join_voice_channel
        assert make_plugin().install() is True
        assert DiscordAdapter.join_voice_channel is wrapped
        assert wrapped.__discord_realtime_voice__ is True

    def test_register_wires_discord_handler(self, installed):
        ctx = MagicMock()
        ctx.get_config = lambda key, default=None: default
        discord_realtime_voice.register(ctx)
        platform, factory = ctx.register_platform_handler.call_args.args
        assert platform == "discord" and callable(factory)

    def test_settings_coerce_string_booleans(self):
        assert make_plugin(realtime_enabled="true").settings["realtime_enabled"] is True
        assert make_plugin(follow_enabled="off").settings["follow_enabled"] is False
        assert make_plugin(model=None).settings["model"] == "gpt-realtime-2.1-mini"


# ── audio drain ──────────────────────────────────────────────────────────


def make_receiver():
    from plugins.platforms.discord.adapter import VoiceReceiver

    vc = MagicMock()
    vc.channel = None
    return VoiceReceiver(vc)


class TestDrainUserAudio:
    def test_returns_target_user_audio_and_drops_others(self):
        receiver = make_receiver()
        receiver.map_ssrc(1, 42)
        receiver.map_ssrc(2, 99)
        receiver._buffers[1] = bytearray(b"\x01" * 10)
        receiver._buffers[2] = bytearray(b"\x02" * 10)
        assert integ.drain_user_audio(receiver, 42) == b"\x01" * 10
        assert len(receiver._buffers) == 0

    def test_zero_accepts_any_mapped_user(self):
        receiver = make_receiver()
        receiver.map_ssrc(1, 42)
        receiver._buffers[1] = bytearray(b"\x01" * 10)
        receiver._buffers[3] = bytearray(b"\x03" * 10)  # unmapped, not inferable
        assert integ.drain_user_audio(receiver, 0) == b"\x01" * 10

    def test_receiver_has_what_drain_needs(self):
        assert integ.missing_instance_attrs(make_receiver(), integ.REQUIRED_RECEIVER_ATTRS) == []


# ── wrapped adapter methods ──────────────────────────────────────────────


class TestAdapterWrappers:
    @pytest.mark.asyncio
    async def test_play_yields_to_realtime(self, installed):
        vc = MagicMock()
        vc.is_connected.return_value = True
        adapter = bare_adapter(_voice_clients={111: vc})
        integ.bridges_of(adapter)[111] = object()
        assert await adapter.play_in_voice_channel(111, "/tmp/x.ogg") is False
        vc.play.assert_not_called()

    @pytest.mark.asyncio
    async def test_leave_closes_bridge_and_discards_pending_audio(self, installed):
        receiver = MagicMock()
        receiver.flush_pending.return_value = []
        adapter = bare_adapter(_voice_receivers={111: receiver}, _voice_text_channels={111: 123})
        bridge = SimpleNamespace(close=AsyncMock())
        integ.bridges_of(adapter)[111] = bridge
        await adapter.leave_voice_channel(111)
        bridge.close.assert_awaited_once()
        receiver.flush_pending.assert_called()
        assert 111 not in integ.bridges_of(adapter)
        assert 111 not in adapter._voice_text_channels  # original teardown still ran

    @pytest.mark.asyncio
    async def test_join_starts_realtime_after_original_succeeds(self):
        plugin = make_plugin()
        plugin.after_join = AsyncMock(return_value=True)
        original = AsyncMock(return_value=True)
        join = plugin._wrap_join(original)
        channel = SimpleNamespace(guild=SimpleNamespace(id=111))
        assert await join("adapter", channel, text_channel_id=5) is True
        original.assert_awaited_once_with("adapter", channel, text_channel_id=5)
        plugin.after_join.assert_awaited_once_with("adapter", 111)

    @pytest.mark.asyncio
    async def test_join_keeps_stt_when_realtime_raises(self):
        plugin = make_plugin()
        plugin.after_join = AsyncMock(side_effect=RuntimeError("ws down"))
        join = plugin._wrap_join(AsyncMock(return_value=True))
        assert await join("adapter", SimpleNamespace(guild=SimpleNamespace(id=1))) is True

    def test_voice_reply_suppressed_while_realtime_active(self, installed, tmp_path):
        from gateway.config import Platform
        from gateway.platforms.base import MessageEvent
        from gateway.run import GatewayRunner
        from gateway.session import SessionSource

        runner = object.__new__(GatewayRunner)
        adapter = bare_adapter(_voice_text_channels={111: 123})
        integ.bridges_of(adapter)[111] = object()
        runner._delivery_adapter_for = lambda source: adapter
        event = MessageEvent(text="hi", source=SessionSource(platform=Platform.DISCORD, chat_id="123", user_id="42"))
        assert runner._should_send_voice_reply(event, "hello", []) is False
        assert integ.active_for_chat(adapter, "123") and not integ.active_for_chat(adapter, "456")


# ── realtime session start ───────────────────────────────────────────────


def join_ready_adapter(monkeypatch, **attrs):
    monkeypatch.setenv("HERMES_REALTIME_OPENAI_API_KEY", "sk-test")
    vc = MagicMock()
    vc.is_playing.return_value = False
    receiver = make_receiver()
    receiver._running = True  # as after VoiceReceiver.start()
    runner = SimpleNamespace(_handle_voice_channel_input=lambda *a, **k: None)
    callback = SimpleNamespace(func=SimpleNamespace(__self__=runner))
    adapter = bare_adapter(_voice_clients={111: vc}, _voice_receivers={111: receiver},
                           _voice_input_callback=callback, **attrs)
    adapter._voice_listen_loop = AsyncMock()
    return adapter, vc


class TestAfterJoin:
    @pytest.mark.asyncio
    async def test_disabled_keeps_stt(self, monkeypatch):
        adapter, vc = join_ready_adapter(monkeypatch)
        assert await make_plugin().after_join(adapter, 111) is False
        vc.play.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_api_key_keeps_stt(self, monkeypatch):
        adapter, _vc = join_ready_adapter(monkeypatch)
        monkeypatch.delenv("HERMES_REALTIME_OPENAI_API_KEY")
        stt = asyncio.ensure_future(asyncio.sleep(3600))
        adapter._voice_listen_tasks[111] = stt
        assert await make_plugin(realtime_enabled=True).after_join(adapter, 111) is False
        assert adapter._voice_listen_tasks[111] is stt and not stt.cancelled()
        stt.cancel()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model,expected", [("gpt-live-1", "LiveVoiceBridge"),
                                                ("gpt-realtime-2.1-mini", "RealtimeVoiceBridge")])
    async def test_model_picks_bridge_and_replaces_stt(self, monkeypatch, model, expected):
        adapter, vc = join_ready_adapter(monkeypatch)
        stt = asyncio.ensure_future(asyncio.sleep(3600))
        adapter._voice_listen_tasks[111] = stt
        started = []

        async def fake_start(self):
            started.append((type(self).__name__, self._instructions))
            self._recv_task = asyncio.ensure_future(asyncio.sleep(3600))  # session stays open

        monkeypatch.setattr(lv.LiveVoiceBridge, "start", fake_start)
        monkeypatch.setattr(rv.RealtimeVoiceBridge, "start", fake_start)
        plugin = make_plugin(realtime_enabled=True, model=model, prompt_preset="ja", instructions="ゆっくり話す",
                             include_soul=False, include_memory=False)
        assert await plugin.after_join(adapter, 111) is True
        await asyncio.sleep(0)
        assert stt.cancelled()
        name, instructions = started[0]
        assert name == expected
        assert "ゆっくり話す" in instructions
        bridge = integ.bridges_of(adapter)[111]
        vc.play.assert_called_once_with(bridge.source)
        adapter._voice_listen_tasks[111].cancel()
        bridge._recv_task.cancel()

    @pytest.mark.asyncio
    async def test_start_failure_restores_stt_loop(self, monkeypatch):
        adapter, vc = join_ready_adapter(monkeypatch)

        async def broken_start(self):
            raise RuntimeError("401")

        monkeypatch.setattr(rv.RealtimeVoiceBridge, "start", broken_start)
        plugin = make_plugin(realtime_enabled=True, include_soul=False, include_memory=False)
        with pytest.raises(RuntimeError):
            await plugin.after_join(adapter, 111)
        assert 111 in adapter._voice_listen_tasks
        assert 111 not in integ.bridges_of(adapter)
        await asyncio.sleep(0)
        adapter._voice_listen_loop.assert_awaited_once_with(111)

    @pytest.mark.asyncio
    async def test_move_keeps_existing_session(self, monkeypatch):
        adapter, vc = join_ready_adapter(monkeypatch)
        integ.bridges_of(adapter)[111] = object()
        assert await make_plugin(realtime_enabled=True).after_join(adapter, 111) is False
        vc.play.assert_not_called()


# ── voice thread ─────────────────────────────────────────────────────────


class TestVoiceThread:
    def adapter(self, channel):
        client = MagicMock()
        client.get_channel = MagicMock(return_value=channel)
        return bare_adapter(
            _voice_text_channels={111: 123}, _client=client,
            _voice_sources={111: {"platform": "discord", "chat_id": "123", "chat_type": "group", "user_id": "9"}})

    @pytest.mark.asyncio
    async def test_call_gets_its_own_thread_session(self):
        channel = MagicMock()
        channel.create_thread = AsyncMock(return_value=SimpleNamespace(id=456, name="Voice — 10/13 15:00"))
        adapter = self.adapter(channel)
        await integ.open_voice_thread(adapter, 111)
        assert channel.create_thread.await_args.kwargs["name"].startswith("Voice — ")
        assert adapter._voice_text_channels[111] == 456
        assert adapter._voice_sources[111] == {
            "platform": "discord", "chat_id": "456", "chat_name": "Voice — 10/13 15:00",
            "chat_type": "thread", "user_id": "9", "thread_id": "456", "parent_chat_id": "123",
        }

    @pytest.mark.asyncio
    async def test_thread_failure_keeps_text_channel(self):
        channel = MagicMock()
        channel.create_thread = AsyncMock(side_effect=RuntimeError("missing permissions"))
        adapter = self.adapter(channel)
        await integ.open_voice_thread(adapter, 111)
        assert adapter._voice_text_channels[111] == 123
        assert adapter._voice_sources[111]["chat_id"] == "123"

    @pytest.mark.asyncio
    async def test_channels_without_threads_are_left_alone(self):
        adapter = self.adapter(object())  # e.g. a DM channel
        await integ.open_voice_thread(adapter, 111)
        assert adapter._voice_text_channels[111] == 123


# ── delegation ───────────────────────────────────────────────────────────


def make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._session_db = None
    runner.session_store = MagicMock()
    runner._is_user_authorized = lambda source: True
    return runner


def delegate_adapter():
    from gateway.config import Platform
    from gateway.session import SessionSource

    adapter = AsyncMock()
    adapter._voice_text_channels = {111: 123}
    adapter._voice_sources = {}
    adapter._allowed_role_ids = set()
    adapter._client = MagicMock()
    adapter._client.get_channel = MagicMock(return_value=AsyncMock())
    adapter._thread_id_and_chat_for_channel = MagicMock(return_value=(None, "123"))
    adapter._resolve_channel_prompt = MagicMock(return_value=None)
    adapter._guild_channel_labels = MagicMock(return_value=None)
    adapter.build_source = MagicMock(side_effect=lambda **kw: SessionSource(platform=Platform.DISCORD, **kw))
    return adapter


class TestDelegation:
    @pytest.mark.asyncio
    async def test_runs_agent_turn_and_returns_reply(self):
        from gateway.platforms.base import MessageType

        runner, adapter = make_runner(), delegate_adapter()
        runner._canonicalize = MagicMock(return_value=object())
        runner._is_user_authorized_for_source = lambda source: True
        runner._handle_message = AsyncMock(return_value="予定は3件です")

        reply = await integ.delegate_to_agent(runner, adapter, 111, 42, "今日の予定は？")

        assert reply == "予定は3件です"
        event = runner._handle_message.call_args[0][0]
        assert event.text.startswith("[Voice call]") and event.text.endswith("今日の予定は？")
        assert event.message_type == MessageType.TEXT
        assert event.source.chat_id == "123"
        assert event.raw_message.guild_id == 111
        adapter.send.assert_awaited_once_with("123", "予定は3件です")
        logged = adapter._client.get_channel.return_value.send.call_args[0][0]
        assert logged == "**[Voice → Hermes]** 今日の予定は？"

    @pytest.mark.asyncio
    async def test_unauthorized_user_is_refused(self):
        runner, adapter = make_runner(), delegate_adapter()
        runner._canonicalize = MagicMock(return_value=object())
        runner._is_user_authorized_for_source = lambda source: False
        runner._handle_message = AsyncMock()
        assert "not authorized" in await integ.delegate_to_agent(runner, adapter, 111, 42, "hi")
        runner._handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_unserved_profile_is_refused(self):
        runner, adapter = make_runner(), delegate_adapter()
        runner._canonicalize = MagicMock(return_value=None)
        runner._handle_message = AsyncMock()
        assert "does not serve" in await integ.delegate_to_agent(runner, adapter, 111, 42, "hi")
        runner._handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_mentions_are_defused_in_the_log_line(self):
        runner, adapter = make_runner(), delegate_adapter()
        runner._canonicalize = MagicMock(return_value=object())
        runner._is_user_authorized_for_source = lambda source: True
        runner._handle_message = AsyncMock(return_value="")
        await integ.delegate_to_agent(runner, adapter, 111, 42, "@everyone hi")
        logged = adapter._client.get_channel.return_value.send.call_args[0][0]
        assert "@everyone" not in logged


# ── voice follow ─────────────────────────────────────────────────────────


class TestVoiceFollow:
    def adapter(self, monkeypatch):
        client = MagicMock()
        client.get_channel = MagicMock(return_value=SimpleNamespace(name="home"))
        adapter = bare_adapter(_client=client, _on_voice_disconnect=MagicMock())
        adapter._allowed_user_ids = {"42"}
        adapter.build_source = MagicMock(return_value=SimpleNamespace(to_dict=lambda: {"user_id": "42"}))
        adapter.join_voice_channel = AsyncMock(return_value=True)
        adapter.leave_voice_channel = AsyncMock()
        monkeypatch.setenv("DISCORD_HOME_CHANNEL", "123")
        return adapter

    @staticmethod
    def member(user_id=42, guild_id=111):
        return SimpleNamespace(id=user_id, display_name="me", guild=SimpleNamespace(id=guild_id))

    @staticmethod
    def state(channel):
        return SimpleNamespace(channel=channel)

    def test_target_defaults_to_single_allowed_user_and_home_channel(self, monkeypatch):
        assert make_plugin(follow_enabled=True).follow_target(self.adapter(monkeypatch)) == (42, 123)

    def test_explicit_target_wins(self, monkeypatch):
        plugin = make_plugin(follow_enabled=True, follow_user_id="7", follow_text_channel_id="8")
        assert plugin.follow_target(self.adapter(monkeypatch)) == (7, 8)

    def test_target_none_when_disabled(self, monkeypatch):
        assert make_plugin().follow_target(self.adapter(monkeypatch)) is None

    def test_target_none_when_user_ambiguous(self, monkeypatch):
        adapter = self.adapter(monkeypatch)
        adapter._allowed_user_ids = {"42", "43"}
        assert make_plugin(follow_enabled=True).follow_target(adapter) is None

    @pytest.mark.asyncio
    async def test_joins_when_followed_user_enters(self, monkeypatch):
        adapter = self.adapter(monkeypatch)
        vc_channel = SimpleNamespace(name="General")
        await make_plugin(follow_enabled=True).follow_voice_state(
            adapter, self.member(), self.state(None), self.state(vc_channel))
        adapter.join_voice_channel.assert_awaited_once_with(vc_channel, text_channel_id=123, source={"user_id": "42"})

    @pytest.mark.asyncio
    async def test_moves_when_followed_user_switches(self, monkeypatch):
        adapter = self.adapter(monkeypatch)
        a, b = SimpleNamespace(name="A"), SimpleNamespace(name="B")
        await make_plugin(follow_enabled=True).follow_voice_state(adapter, self.member(), self.state(a), self.state(b))
        assert adapter.join_voice_channel.call_args[0][0] is b

    @pytest.mark.asyncio
    async def test_leaves_when_followed_user_leaves(self, monkeypatch):
        adapter = self.adapter(monkeypatch)
        adapter._voice_clients[111] = MagicMock()
        await make_plugin(follow_enabled=True).follow_voice_state(
            adapter, self.member(), self.state(SimpleNamespace(name="A")), self.state(None))
        adapter.leave_voice_channel.assert_awaited_once_with(111)
        adapter._on_voice_disconnect.assert_called_once_with("123")

    @pytest.mark.asyncio
    async def test_ignores_other_users_and_mute_toggles(self, monkeypatch):
        adapter = self.adapter(monkeypatch)
        ch = SimpleNamespace(name="A")
        plugin = make_plugin(follow_enabled=True)
        await plugin.follow_voice_state(adapter, self.member(user_id=99), self.state(None), self.state(ch))
        await plugin.follow_voice_state(adapter, self.member(), self.state(ch), self.state(ch))
        adapter.join_voice_channel.assert_not_called()
        adapter.leave_voice_channel.assert_not_called()

    def test_connect_adds_listener(self):
        native = MagicMock()
        make_plugin().on_discord_connect(native, bare_adapter())
        callback, name = native.add_listener.call_args.args
        assert name == "on_voice_state_update" and inspect.iscoroutinefunction(callback)

    def test_connect_skips_unknown_adapter_shape(self, caplog):
        native = MagicMock()
        make_plugin().on_discord_connect(native, SimpleNamespace())
        native.add_listener.assert_not_called()
        assert "voice follow is off" in caplog.text
