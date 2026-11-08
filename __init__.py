"""discord-realtime-voice: talk with your Hermes agent in Discord voice channels through an
OpenAI Realtime / GPT-Live speech-to-speech model that delegates real work to the agent.

See README.md for setup and settings.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register(ctx) -> None:
    from .integration import RealtimeVoice, Settings

    plugin = RealtimeVoice(Settings(ctx.get_config))
    if not plugin.install():
        return  # incompatible Hermes build; install() already logged why
    ctx.register_platform_handler("discord", plugin.on_discord_connect)
