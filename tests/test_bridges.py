"""Bridge protocol tests: Realtime (function calling) and GPT-Live (client delegation).

Both bridges run against fake WebSockets; nothing here touches Hermes.
"""

import asyncio
import base64
import json
from unittest.mock import AsyncMock

import numpy as np
import pytest

from discord_realtime_voice import live_voice as lv
from discord_realtime_voice import realtime_voice as rv


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send(self, data):
        self.sent.append(json.loads(data))

    async def close(self):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(3600)
        raise StopAsyncIteration


def _pcm(*samples):
    return np.array(samples, dtype="<i2").tobytes()


async def _started_bridge(delegate=None):
    ws = FakeWS()

    async def connect(url, api_key):
        assert api_key == "sk-test"
        return ws

    bridge = rv.RealtimeVoiceBridge(
        api_key="sk-test",
        delegate=delegate or AsyncMock(return_value="done"),
        connect=connect,
    )
    await bridge.start()
    return bridge, ws


# =====================================================================
# Audio conversion + playback source
# =====================================================================

class TestAudioConversion:
    def test_discord_frame_to_realtime_is_quarter_size(self):
        assert len(rv.discord_to_realtime(b"\x00" * rv.FRAME_BYTES)) == rv.FRAME_BYTES // 4

    def test_realtime_to_discord_is_four_times_size(self):
        assert len(rv.realtime_to_discord(b"\x00" * 960)) == rv.FRAME_BYTES

    def test_round_trip_preserves_constant_signal(self):
        realtime = rv.realtime_to_discord(_pcm(1000, -500))
        assert np.frombuffer(realtime, dtype="<i2").tolist() == [1000] * 4 + [-500] * 4
        assert np.frombuffer(rv.discord_to_realtime(realtime), dtype="<i2").tolist() == [1000, -500]

    def test_stereo_channels_are_averaged(self):
        # two stereo frames (L, R): (100, 300), (100, 300) -> one mono sample 200
        assert np.frombuffer(rv.discord_to_realtime(_pcm(100, 300, 100, 300)), dtype="<i2").tolist() == [200]


class TestRealtimeAudioSource:
    def test_read_returns_silence_when_empty(self):
        src = rv.RealtimeAudioSource()
        assert src.read() == b"\x00" * rv.FRAME_BYTES
        assert src.played_bytes == 0

    def test_read_pads_partial_frame_and_counts_played(self):
        src = rv.RealtimeAudioSource()
        src.feed(b"\x01" * 100)
        frame = src.read()
        assert len(frame) == rv.FRAME_BYTES
        assert frame[:100] == b"\x01" * 100
        assert src.played_bytes == 100

    def test_clear_returns_unplayed_bytes(self):
        src = rv.RealtimeAudioSource()
        src.feed(b"\x01" * (rv.FRAME_BYTES * 2))
        src.read()
        assert src.clear() == rv.FRAME_BYTES
        assert src.read() == b"\x00" * rv.FRAME_BYTES


# =====================================================================
# Bridge protocol
# =====================================================================

class TestRealtimeVoiceBridge:
    @pytest.mark.asyncio
    async def test_start_configures_session_with_ask_hermes(self):
        bridge, ws = await _started_bridge()
        try:
            session = ws.sent[0]["session"]
            assert ws.sent[0]["type"] == "session.update"
            assert session["model"] == "gpt-realtime-2.1-mini"
            assert session["audio"]["input"]["turn_detection"] == {"type": "semantic_vad"}
            assert session["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
            assert [t["name"] for t in session["tools"]] == ["ask_hermes"]
            assert bridge.running
        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_send_audio_appends_downsampled_pcm(self):
        bridge, ws = await _started_bridge()
        try:
            await bridge.send_audio(b"\x00" * rv.FRAME_BYTES)
            msg = ws.sent[-1]
            assert msg["type"] == "input_audio_buffer.append"
            assert len(base64.b64decode(msg["audio"])) == rv.FRAME_BYTES // 4
        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_output_audio_delta_is_queued_for_playback(self):
        bridge, _ws = await _started_bridge()
        try:
            await bridge._handle_event({
                "type": "response.output_audio.delta",
                "item_id": "item_1",
                "delta": base64.b64encode(b"\x00" * 960).decode(),
            })
            assert bridge.source.clear() == rv.FRAME_BYTES
        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_function_call_delegates_and_returns_output(self):
        delegate = AsyncMock(return_value="明日は晴れです")
        bridge, ws = await _started_bridge(delegate)
        try:
            await bridge._handle_event({
                "type": "response.done",
                "response": {"output": [{
                    "type": "function_call",
                    "name": "ask_hermes",
                    "call_id": "call_1",
                    "arguments": json.dumps({"task": "明日の天気"}),
                }]},
            })
            await asyncio.gather(*list(bridge._tool_tasks))
            delegate.assert_awaited_once_with("明日の天気")
            output_msg, create_msg = ws.sent[-2], ws.sent[-1]
            assert output_msg["item"]["type"] == "function_call_output"
            assert output_msg["item"]["call_id"] == "call_1"
            assert json.loads(output_msg["item"]["output"]) == {"result": "明日は晴れです"}
            assert create_msg == {"type": "response.create"}
        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_delegate_failure_is_reported_to_model(self):
        bridge, ws = await _started_bridge(AsyncMock(side_effect=RuntimeError("boom")))
        try:
            await bridge._run_tool({"name": "ask_hermes", "call_id": "c", "arguments": "{}"})
            assert "boom" in json.loads(ws.sent[-2]["item"]["output"])["result"]
        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_barge_in_truncates_unplayed_audio(self):
        bridge, ws = await _started_bridge()
        try:
            await bridge._handle_event({
                "type": "response.output_audio.delta",
                "item_id": "item_1",
                "delta": base64.b64encode(b"\x00" * 960 * 3).decode(),
            })
            bridge.source.read()  # one 20 ms frame played
            await bridge._handle_event({"type": "input_audio_buffer.speech_started"})
            assert ws.sent[-1] == {
                "type": "conversation.item.truncate",
                "item_id": "item_1",
                "content_index": 0,
                "audio_end_ms": 20,
            }
            assert bridge.source.read() == b"\x00" * rv.FRAME_BYTES
        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_speech_started_without_pending_audio_sends_nothing(self):
        bridge, ws = await _started_bridge()
        try:
            sent_before = len(ws.sent)
            await bridge._handle_event({"type": "input_audio_buffer.speech_started"})
            assert len(ws.sent) == sent_before
        finally:
            await bridge.close()


class FakeLiveWS:
    def __init__(self, first_reply=None):
        self.sent = []
        self._first_reply = first_reply or {"type": "session.started", "session": {}}

    async def send(self, data):
        self.sent.append(json.loads(data))

    async def recv(self):
        return json.dumps(self._first_reply)

    async def close(self):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(3600)
        raise StopAsyncIteration


async def _started_live_bridge(delegate=None, **kwargs):
    ws = FakeLiveWS()

    async def connect(url, api_key):
        assert url == lv.LIVE_URL
        assert api_key == "sk-test"
        return ws

    bridge = lv.LiveVoiceBridge(
        api_key="sk-test",
        delegate=delegate or AsyncMock(return_value="done"),
        connect=connect,
        **kwargs,
    )
    await bridge.start()
    return bridge, ws


class TestLiveVoiceBridge:
    @pytest.mark.asyncio
    async def test_start_uses_client_delegation(self):
        bridge, ws = await _started_live_bridge(instructions="Speak calmly.")
        try:
            start = ws.sent[0]
            assert start["type"] == "session.start"
            session = start["session"]
            assert session["model"] == "gpt-live-1"
            assert session["delegation"] == {"type": "client"}
            assert session["audio"] == {
                "format": {"type": "audio/pcm", "rate": 24000},
                "output": {"voice": "marin"},
            }
            assert session["instructions"] == "Speak calmly."
            assert bridge.running
        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_start_raises_on_error(self):
        ws = FakeLiveWS(first_reply={"type": "error", "error": {"message": "bad model"}})

        async def connect(url, api_key):
            return ws

        bridge = lv.LiveVoiceBridge(api_key="k", delegate=AsyncMock(), connect=connect)
        with pytest.raises(RuntimeError, match="bad model"):
            await bridge.start()

    @pytest.mark.asyncio
    async def test_send_audio_appends_downsampled_pcm(self):
        bridge, ws = await _started_live_bridge()
        try:
            await bridge.send_audio(b"\x00" * rv.FRAME_BYTES)
            event = ws.sent[-1]
            assert event["type"] == "session.input_audio.append"
            assert len(base64.b64decode(event["audio"])) == rv.FRAME_BYTES // 4
        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_output_audio_is_queued_for_playback(self):
        bridge, _ws = await _started_live_bridge()
        try:
            await bridge._handle_event({
                "type": "session.output_audio.delta",
                "delta": base64.b64encode(b"\x00" * 960).decode(),
            })
            assert len(bridge.source.read()) == rv.FRAME_BYTES
            assert bridge.source.played_bytes == rv.FRAME_BYTES
        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_delegation_sends_transcript_and_returns_commentary(self, monkeypatch):
        monkeypatch.setattr(lv, "DELEGATION_SETTLE_S", 0)
        delegate = AsyncMock(return_value="digest には新着が 3 件あるよ。")
        bridge, ws = await _started_live_bridge(delegate=delegate)
        try:
            for etype, delta in [
                ("session.input_transcript.delta", "ねえ、"),
                ("session.input_transcript.delta", "digest 見といて"),
                ("session.output_transcript.delta", "ちょっと見てみるね"),
            ]:
                await bridge._handle_event({"type": etype, "delta": delta})
            await bridge._handle_event({
                "type": "session.delegation.created",
                "delegation": {"id": "item_1", "type": "delegation", "target": "client"},
            })
            await asyncio.gather(*bridge._tool_tasks)

            task = delegate.await_args.args[0]
            assert "User: ねえ、digest 見といて" in task
            assert "You (voice): ちょっと見てみるね" in task
            assert ws.sent[-1] == {
                "type": "session.commentary.append",
                "delegation_id": "item_1",
                "content": "digest には新着が 3 件あるよ。",
            }
        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_delegation_failure_is_reported(self, monkeypatch):
        monkeypatch.setattr(lv, "DELEGATION_SETTLE_S", 0)
        bridge, ws = await _started_live_bridge(delegate=AsyncMock(side_effect=RuntimeError("boom")))
        try:
            await bridge._handle_event({"type": "session.delegation.created", "delegation": {"id": "d"}})
            await asyncio.gather(*bridge._tool_tasks)
            assert ws.sent[-1]["delegation_id"] == "d"
            assert "boom" in ws.sent[-1]["content"]
        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_commentary_is_capped(self, monkeypatch):
        monkeypatch.setattr(lv, "DELEGATION_SETTLE_S", 0)
        bridge, ws = await _started_live_bridge(delegate=AsyncMock(return_value="あ" * 5000))
        try:
            await bridge._handle_event({"type": "session.delegation.created", "delegation": {"id": "d"}})
            await asyncio.gather(*bridge._tool_tasks)
            assert len(ws.sent[-1]["content"]) == lv.MAX_COMMENTARY_CHARS
        finally:
            await bridge.close()

    def test_transcript_keeps_only_recent_turns(self):
        bridge = lv.LiveVoiceBridge(api_key="k", delegate=AsyncMock())
        for i in range(lv.TRANSCRIPT_TURNS * 10):
            bridge._add_transcript("user" if i % 2 == 0 else "assistant", f"t{i}")
        assert len(bridge._transcript) <= lv.TRANSCRIPT_TURNS * 4
        task = bridge._task_from_transcript()
        assert f"t{lv.TRANSCRIPT_TURNS * 10 - 1}" in task
        assert "t0\n" not in task
