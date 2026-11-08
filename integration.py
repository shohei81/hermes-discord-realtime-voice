"""Wiring between Hermes' Discord voice support and the realtime bridges.

Hermes has no public extension point for voice-channel audio, so this module wraps a few
methods of the built-in Discord adapter and gateway runner. Every name it relies on is listed
in ``REQUIRED`` and checked before anything is patched: when a Hermes release moves one of
them, the plugin logs what is missing and stays inactive instead of breaking the gateway.

Wrapped methods (all call the original):

* ``DiscordAdapter.join_voice_channel``  -> after a successful join, swap the STT listen loop
  for a realtime bridge
* ``DiscordAdapter.leave_voice_channel`` -> close the bridge first, discard unsent audio
* ``DiscordAdapter.play_in_voice_channel`` -> refuse while a bridge owns the audio stream
* ``GatewayVoiceMixin._should_send_voice_reply`` -> no TTS on top of a realtime session
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import weakref
from contextlib import suppress
from types import SimpleNamespace
from typing import Any, Callable, Optional

from . import prompts

logger = logging.getLogger(__name__)

BRIDGES_ATTR = "_discord_realtime_voice_bridges"  # per adapter: guild_id -> bridge

# (module, attribute path) pairs that must exist before patching.
REQUIRED: tuple[tuple[str, str], ...] = (
    ("plugins.platforms.discord.adapter", "DiscordAdapter.join_voice_channel"),
    ("plugins.platforms.discord.adapter", "DiscordAdapter.leave_voice_channel"),
    ("plugins.platforms.discord.adapter", "DiscordAdapter.play_in_voice_channel"),
    ("plugins.platforms.discord.adapter", "DiscordAdapter._voice_listen_loop"),
    ("plugins.platforms.discord.adapter", "DiscordAdapter._reset_voice_timeout"),
    ("plugins.platforms.discord.adapter", "DiscordAdapter.build_source"),
    ("plugins.platforms.discord.adapter", "VoiceReceiver.flush_pending"),
    ("plugins.platforms.discord.adapter", "VoiceReceiver._infer_user_for_ssrc"),
    ("plugins.platforms.discord.adapter", "VoiceReceiver.SAMPLE_RATE"),
    ("plugins.platforms.discord.adapter", "VoiceReceiver.CHANNELS"),
    ("gateway.run_voice", "GatewayVoiceMixin._should_send_voice_reply"),
    ("gateway.run_voice", "GatewayVoiceMixin._voice_input_source"),
    ("gateway.run_voice", "GatewayVoiceMixin._handle_voice_channel_input"),
    ("gateway.platforms.base", "MessageEvent"),
    ("gateway.platforms.base", "MessageType"),
    ("gateway.session_identity", "clear_identity"),
)

# Instance attributes of a live DiscordAdapter (checked when the adapter connects).
REQUIRED_ADAPTER_ATTRS = (
    "_client", "_voice_clients", "_voice_receivers", "_voice_listen_tasks",
    "_voice_text_channels", "_voice_sources", "_voice_input_callback", "_on_voice_disconnect",
)
# Instance attributes of a VoiceReceiver used to drain audio.
REQUIRED_RECEIVER_ATTRS = ("_lock", "_buffers", "_ssrc_to_user", "_last_packet_time", "_running")


def _resolve(module: str, path: str) -> Any:
    import importlib

    obj: Any = importlib.import_module(module)
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def missing_requirements() -> list[str]:
    """Names from ``REQUIRED`` that this Hermes build does not have."""
    missing = []
    for module, path in REQUIRED:
        try:
            _resolve(module, path)
        except Exception:
            missing.append(f"{module}:{path}")
    return missing


def missing_instance_attrs(obj: Any, names: tuple[str, ...]) -> list[str]:
    return [n for n in names if not hasattr(obj, n)]


def drain_user_audio(receiver: Any, user_id: int = 0) -> bytes:
    """Pop all buffered PCM for ``user_id`` (0 = any mapped user) from a ``VoiceReceiver``.

    Streaming counterpart of ``VoiceReceiver.check_silence`` for realtime mode, where turn
    detection happens server-side. Audio from other speakers is discarded so buffers never
    grow unbounded.
    """
    out = bytearray()
    with receiver._lock:
        for ssrc in list(receiver._buffers):
            uid = receiver._ssrc_to_user.get(ssrc) or receiver._infer_user_for_ssrc(ssrc)
            buf = receiver._buffers.pop(ssrc)
            receiver._last_packet_time.pop(ssrc, None)
            if uid and (not user_id or uid == user_id):
                out.extend(buf)
    return bytes(out)


def bridges_of(adapter: Any) -> dict:
    bridges = getattr(adapter, BRIDGES_ATTR, None)
    if bridges is None:
        bridges = {}
        setattr(adapter, BRIDGES_ATTR, bridges)
    return bridges


def runner_of(adapter: Any) -> Any:
    """The gateway runner, recovered from the voice-input callback it binds on the adapter."""
    callback = getattr(adapter, "_voice_input_callback", None)
    func = getattr(callback, "func", None)  # functools.partial(runner._handle_voice_channel_input)
    return getattr(func, "__self__", None)


class Settings:
    """Reads this plugin's settings on every call, so config edits apply on the next join."""

    DEFAULTS: dict[str, Any] = {
        "realtime_enabled": False,
        "model": "gpt-realtime-2.1-mini",
        "voice": "marin",
        "prompt_preset": "en",
        "instructions": "",
        "include_soul": True,
        "include_memory": True,
        "open_thread": True,
        "api_key_env": "HERMES_REALTIME_OPENAI_API_KEY",
        "follow_enabled": False,
        "follow_user_id": "",
        "follow_text_channel_id": "",
    }

    def __init__(self, get_config: Callable[[str, Any], Any]):
        self._get = get_config

    def __getitem__(self, key: str) -> Any:
        default = self.DEFAULTS[key]
        try:
            value = self._get(key, default)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("discord-realtime-voice: could not read setting %s: %s", key, e)
            return default
        if value is None:
            return default
        if isinstance(default, bool) and isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return value


class RealtimeVoice:
    """Plugin state and behavior; one instance per loaded plugin."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._patched = False
        self._adapters: "weakref.WeakSet[Any]" = weakref.WeakSet()

    # ── installation ────────────────────────────────────────────────────

    def install(self) -> bool:
        """Patch Hermes when every required name exists; False (and a warning) otherwise."""
        if self._patched:
            return True
        missing = missing_requirements()
        if missing:
            logger.warning(
                "discord-realtime-voice: this Hermes build lacks %s; the plugin stays inactive "
                "(Discord voice keeps working without it). Check for a plugin update.",
                ", ".join(missing),
            )
            return False
        from gateway.run_voice import GatewayVoiceMixin
        from plugins.platforms.discord.adapter import DiscordAdapter

        _wrap(DiscordAdapter, "join_voice_channel", self._wrap_join)
        _wrap(DiscordAdapter, "leave_voice_channel", self._wrap_leave)
        _wrap(DiscordAdapter, "play_in_voice_channel", self._wrap_play)
        _wrap(GatewayVoiceMixin, "_should_send_voice_reply", self._wrap_should_send_voice_reply)
        self._patched = True
        return True

    def on_discord_connect(self, native: Any, adapter: Any) -> None:
        """``register_platform_handler`` factory: remember the adapter, add the follow listener."""
        missing = missing_instance_attrs(adapter, REQUIRED_ADAPTER_ATTRS)
        if missing:
            logger.warning("discord-realtime-voice: Discord adapter lacks %s; voice follow is off",
                           ", ".join(missing))
            return
        self._adapters.add(adapter)
        add_listener = getattr(native, "add_listener", None)
        if not callable(add_listener):
            logger.warning("discord-realtime-voice: Discord client has no add_listener; voice follow is off")
            return

        async def on_voice_state_update(member, before, after):
            try:
                await self.follow_voice_state(adapter, member, before, after)
            except Exception as e:
                logger.warning("discord-realtime-voice: voice follow failed: %s", e, exc_info=True)

        add_listener(on_voice_state_update, "on_voice_state_update")

    # ── wrappers ────────────────────────────────────────────────────────

    def _wrap_join(self, original):
        plugin = self

        @functools.wraps(original)
        async def join_voice_channel(adapter, channel, *args, **kwargs):
            joined = await original(adapter, channel, *args, **kwargs)
            if joined:
                try:
                    await plugin.after_join(adapter, channel.guild.id)
                except Exception as e:
                    logger.warning("discord-realtime-voice: realtime voice failed to start, "
                                   "using STT/TTS: %s", e, exc_info=True)
            return joined

        return join_voice_channel

    def _wrap_leave(self, original):
        @functools.wraps(original)
        async def leave_voice_channel(adapter, guild_id, *args, **kwargs):
            bridge = bridges_of(adapter).pop(guild_id, None)
            if bridge is not None:
                receiver = adapter._voice_receivers.get(guild_id)
                if receiver is not None:
                    receiver.flush_pending()  # realtime mode streamed it; nothing to transcribe
                with suppress(Exception):
                    await bridge.close()
            return await original(adapter, guild_id, *args, **kwargs)

        return leave_voice_channel

    def _wrap_play(self, original):
        @functools.wraps(original)
        async def play_in_voice_channel(adapter, guild_id, *args, **kwargs):
            if guild_id in bridges_of(adapter):
                return False  # the realtime session owns the outgoing audio stream
            return await original(adapter, guild_id, *args, **kwargs)

        return play_in_voice_channel

    def _wrap_should_send_voice_reply(self, original):
        @functools.wraps(original)
        def _should_send_voice_reply(runner, event, *args, **kwargs):
            source = getattr(event, "source", None)
            with suppress(Exception):
                adapter = runner._delivery_adapter_for(source)
                if adapter is not None and active_for_chat(adapter, source.chat_id):
                    return False  # a realtime session speaks for itself; never layer TTS over it
            return original(runner, event, *args, **kwargs)

        return _should_send_voice_reply

    # ── realtime session ────────────────────────────────────────────────

    async def after_join(self, adapter: Any, guild_id: int) -> bool:
        """Replace the STT listen loop with a realtime bridge. False when realtime is not used."""
        s = self.settings
        if not s["realtime_enabled"] or guild_id in bridges_of(adapter):
            return False  # disabled, or a move between channels of an existing session
        vc = adapter._voice_clients.get(guild_id)
        receiver = adapter._voice_receivers.get(guild_id)
        if vc is None or receiver is None:
            return False
        if missing := missing_instance_attrs(receiver, REQUIRED_RECEIVER_ATTRS):
            logger.warning("discord-realtime-voice: VoiceReceiver lacks %s; using STT/TTS", ", ".join(missing))
            return False
        api_key = os.getenv(str(s["api_key_env"]), "").strip()
        if not api_key:
            logger.warning("discord-realtime-voice: realtime_enabled but %s is not set; using STT/TTS",
                           s["api_key_env"])
            return False
        runner = runner_of(adapter)
        if runner is None:
            logger.warning("discord-realtime-voice: gateway runner not found; using STT/TTS")
            return False
        from .live_voice import LiveVoiceBridge
        from .realtime_voice import RealtimeVoiceBridge

        model = str(s["model"])
        # gpt-live-* models are full-duplex and use client delegation instead of function calls.
        kind = "live" if model.startswith("gpt-live") else "realtime"
        bridge_cls = LiveVoiceBridge if kind == "live" else RealtimeVoiceBridge

        async def delegate(task: str) -> str:
            source = adapter._voice_sources.get(guild_id) or {}
            return await delegate_to_agent(runner, adapter, guild_id, int(source.get("user_id") or 0), task)

        # From here on the STT loop is replaced; on failure it is restored.
        listen_task = adapter._voice_listen_tasks.pop(guild_id, None)
        if listen_task is not None:
            listen_task.cancel()
        mixers = getattr(adapter, "_voice_mixers", None)
        if isinstance(mixers, dict) and mixers.pop(guild_id, None) is not None and vc.is_playing():
            vc.stop()  # the ambient mixer would compete with the realtime stream
        try:
            if s["open_thread"]:
                await open_voice_thread(adapter, guild_id)
            instructions = "\n\n".join(part for part in (
                prompts.base_instructions(kind, s["prompt_preset"]),
                str(s["instructions"] or ""),
                voice_context(adapter, guild_id, vc, include_soul=bool(s["include_soul"]),
                              include_memory=bool(s["include_memory"])),
            ) if part)
            bridge = bridge_cls(api_key=api_key, delegate=delegate, model=model,
                                voice=str(s["voice"]), instructions=instructions)
            await bridge.start()
        except Exception:
            adapter._voice_listen_tasks[guild_id] = asyncio.ensure_future(adapter._voice_listen_loop(guild_id))
            raise
        if vc.is_playing():
            vc.stop()
        vc.play(bridge.source)
        bridges_of(adapter)[guild_id] = bridge
        adapter._voice_listen_tasks[guild_id] = asyncio.ensure_future(
            self.stream_loop(adapter, guild_id, bridge))
        return True

    async def stream_loop(self, adapter: Any, guild_id: int, bridge: Any) -> None:
        """Stream the invoking user's audio into the session.

        Discord sends no packets while a user is silent, so gaps are filled with silence to
        keep server-side turn detection working.
        """
        from plugins.platforms.discord.adapter import VoiceReceiver

        receiver = adapter._voice_receivers.get(guild_id)
        if not receiver:
            return
        interval = 0.1
        silence = b"\x00" * int(VoiceReceiver.SAMPLE_RATE * VoiceReceiver.CHANNELS * 2 * interval)
        try:
            while receiver._running and bridge.running:
                await asyncio.sleep(interval)
                source = adapter._voice_sources.get(guild_id) or {}
                pcm = drain_user_audio(receiver, int(source.get("user_id") or 0))
                if pcm:
                    adapter._reset_voice_timeout(guild_id)
                await bridge.send_audio(pcm or silence)
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error("discord-realtime-voice: stream loop error: %s", e, exc_info=True)
        if bridges_of(adapter).get(guild_id) is bridge:
            logger.warning("discord-realtime-voice: session closed; leaving voice channel (guild=%d)", guild_id)
            text_ch_id = adapter._voice_text_channels.get(guild_id)
            # Unregister this task first so leave_voice_channel doesn't cancel it mid-teardown.
            adapter._voice_listen_tasks.pop(guild_id, None)
            await adapter.leave_voice_channel(guild_id)
            notify_disconnect(adapter, text_ch_id)

    # ── voice follow ────────────────────────────────────────────────────

    def follow_target(self, adapter: Any) -> Optional[tuple[int, int]]:
        """(user_id, text_channel_id) to follow, or None when disabled/unresolvable."""
        s = self.settings
        if not s["follow_enabled"]:
            return None
        user_id = str(s["follow_user_id"] or "").strip()
        if not user_id:
            allowed = [u for u in getattr(adapter, "_allowed_user_ids", set()) or () if str(u).isdigit()]
            user_id = str(allowed[0]) if len(allowed) == 1 else ""
        text_ch = str(s["follow_text_channel_id"] or os.getenv("DISCORD_HOME_CHANNEL", "")).strip()
        if not (user_id.isdigit() and text_ch.isdigit()):
            logger.warning(
                "discord-realtime-voice: follow_enabled but the user or text channel could not be "
                "resolved (set follow_user_id/follow_text_channel_id, or use a single "
                "DISCORD_ALLOWED_USERS entry and DISCORD_HOME_CHANNEL)")
            return None
        return int(user_id), int(text_ch)

    async def follow_voice_state(self, adapter: Any, member: Any, before: Any, after: Any) -> None:
        """Join, move or leave voice together with the followed user."""
        target = self.follow_target(adapter)
        if not target or member.id != target[0]:
            return
        _, text_ch_id = target
        guild_id = member.guild.id
        if after.channel is None:
            if guild_id in adapter._voice_clients:
                await adapter.leave_voice_channel(guild_id)
                notify_disconnect(adapter, text_ch_id)
            return
        if before.channel == after.channel:
            return  # mute/deafen/stream toggles
        client = adapter._client
        text_channel = client.get_channel(text_ch_id) if client else None
        import discord

        is_dm = isinstance(text_channel, discord.DMChannel)
        source = adapter.build_source(
            chat_id=str(text_ch_id),
            chat_name="" if is_dm else getattr(text_channel, "name", ""),
            chat_type="dm" if is_dm else "group",
            user_id=str(member.id),
            user_name=member.display_name,
        )
        logger.info("discord-realtime-voice: following user %d into %s", member.id, after.channel.name)
        await adapter.join_voice_channel(after.channel, text_channel_id=text_ch_id, source=source.to_dict())


# ── helpers ─────────────────────────────────────────────────────────────


def _wrap(cls: type, name: str, make_wrapper: Callable[[Callable], Callable]) -> None:
    original = getattr(cls, name)
    if getattr(original, "__discord_realtime_voice__", False):
        return  # already wrapped (plugin reloaded)
    wrapper = make_wrapper(original)
    wrapper.__discord_realtime_voice__ = True
    if isinstance(cls.__dict__.get(name), staticmethod):  # pragma: no cover - none of ours today
        wrapper = staticmethod(wrapper)
    setattr(cls, name, wrapper)


def active_for_chat(adapter: Any, chat_id: Any) -> bool:
    """True when a realtime session is bound to this text channel (or thread)."""
    texts = getattr(adapter, "_voice_text_channels", {}) or {}
    return any(str(texts.get(gid)) == str(chat_id) for gid in bridges_of(adapter))


def notify_disconnect(adapter: Any, text_ch_id: Any) -> None:
    callback = getattr(adapter, "_on_voice_disconnect", None)
    if callback and text_ch_id:
        with suppress(Exception):
            callback(str(text_ch_id))


async def open_voice_thread(adapter: Any, guild_id: int) -> None:
    """Give this call its own thread (and Hermes session) in the bound text channel.

    Delegations then start from a short history instead of the channel's long-running
    session, which keeps every backend turn small and fast.
    """
    text_ch_id = adapter._voice_text_channels.get(guild_id)
    client = adapter._client
    channel = client.get_channel(text_ch_id) if (client and text_ch_id) else None
    if channel is None or not hasattr(channel, "create_thread"):
        return  # DMs and threads cannot hold a new thread
    try:
        import discord
        import hermes_time

        thread = await channel.create_thread(
            name=f"Voice — {hermes_time.now():%m/%d %H:%M}",
            type=discord.ChannelType.public_thread,
            auto_archive_duration=1440,
        )
    except Exception as e:
        logger.warning("discord-realtime-voice: could not open a voice thread in %s: %s", text_ch_id, e)
        return
    source = adapter._voice_sources.get(guild_id)
    if source:
        adapter._voice_sources[guild_id] = {
            **source,
            "chat_id": str(thread.id),
            "chat_name": thread.name,
            "chat_type": "thread",
            "thread_id": str(thread.id),
            "parent_chat_id": str(text_ch_id),
        }
    adapter._voice_text_channels[guild_id] = thread.id


def voice_context(adapter: Any, guild_id: int, vc: Any, *, include_soul: bool, include_memory: bool) -> str:
    """Identity, memory and whereabouts for the voice prompt (what the text agent also sees)."""
    parts: list[str] = []
    if include_soul:
        try:
            from agent.prompt_builder import load_soul_md

            if soul := load_soul_md():
                parts.append(soul)
        except Exception as e:
            logger.debug("discord-realtime-voice: could not load SOUL.md: %s", e)
    if include_memory:
        try:
            from tools.memory_tool import MemoryStore

            store = MemoryStore()
            store.load_from_disk()
            for target in ("user", "memory"):
                if block := store.format_for_system_prompt(target):
                    parts.append(block)
        except Exception as e:
            logger.debug("discord-realtime-voice: could not load memory: %s", e)

    where = []
    with suppress(Exception):
        import hermes_time

        where.append(f"Current time: {hermes_time.now():%Y-%m-%d %H:%M %Z}")
    channel = getattr(vc, "channel", None)
    if channel is not None:
        guild = getattr(channel, "guild", None)
        where.append(f"You are in the Discord voice channel '{channel.name}' on the server "
                     f"'{getattr(guild, 'name', '')}'.")
        text_channels = [c for c in getattr(guild, "text_channels", []) or [] if getattr(c, "name", None)]
        if text_channels:
            listing = ", ".join(f"#{c.name} ({c.id})" for c in text_channels)
            where.append(f"Text channels on this server that your backend (Hermes) can read: {listing}.")
    text_ch_id = adapter._voice_text_channels.get(guild_id)
    client = adapter._client
    text_ch = client.get_channel(text_ch_id) if (client and text_ch_id) else None
    if text_ch is not None and getattr(text_ch, "name", None):
        where.append(f"Your requests to the backend and its replies are logged in the text channel #{text_ch.name}.")
    if where:
        parts.append("\n".join(where))
    return "\n\n".join(parts)


async def delegate_to_agent(runner: Any, adapter: Any, guild_id: int, user_id: int, task: str) -> str:
    """Run a task handed over by the voice layer through the normal agent pipeline.

    The turn runs in the bound text channel's session (the call's thread when one was opened);
    the reply text is returned for the voice model to speak. Task and reply are mirrored to the
    text channel. Authorization follows the same rules as spoken input.
    """
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session_identity import clear_identity

    text_ch_id = adapter._voice_text_channels.get(guild_id)
    if not text_ch_id:
        return "Hermes is not linked to a text channel for this voice session."
    source = runner._voice_input_source(adapter, guild_id, user_id, text_ch_id)
    clear_identity(source)
    if runner._canonicalize(source, transport_profile=getattr(adapter, "_owner_profile", None)) is None:
        return "This voice session is routed to a profile this gateway does not serve."
    roles = getattr(adapter, "_allowed_role_ids", None)
    if isinstance(roles, (set, frozenset)) and roles:
        client = adapter._client
        guild = client.get_guild(guild_id) if client else None
        source.role_authorized = guild is not None and adapter._is_allowed_user(
            str(user_id), guild=guild, is_dm=False) is True
    if not runner._is_user_authorized_for_source(source):
        logger.debug("discord-realtime-voice: unauthorized delegation from user %d", user_id)
        return "This user is not authorized to use Hermes."

    client = adapter._client
    channel = client.get_channel(text_ch_id) if client else None
    safe_task = task[:2000].replace("@everyone", "@​everyone").replace("@here", "@​here")
    with suppress(Exception):
        if channel:
            await channel.send(f"**[Voice → Hermes]** {safe_task}")  # a log line, not a ping

    channel_prompt = None
    if callable(resolver := getattr(adapter, "_resolve_channel_prompt", None)):
        with suppress(Exception):
            resolved = resolver(str(text_ch_id), source.parent_chat_id or None)
            channel_prompt = resolved if isinstance(resolved, str) else None
    event = MessageEvent(
        source=source, text=prompts.DELEGATION_PREAMBLE + task, message_type=MessageType.TEXT,
        raw_message=SimpleNamespace(guild_id=guild_id, guild=None), channel_prompt=channel_prompt)
    response = await runner._handle_message(event) or ""
    if response:
        try:
            await adapter.send(str(text_ch_id), response)
        except Exception:
            logger.debug("discord-realtime-voice: failed to mirror the reply", exc_info=True)
    return response
