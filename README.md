# discord-realtime-voice

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) plugin for talking with your agent in Discord voice channels through an OpenAI speech-to-speech model.

Hermes' built-in Discord voice mode works turn by turn: speech-to-text, an agent turn, then text-to-speech. This plugin puts a realtime voice model in front of the agent instead.

- **The voice model handles the conversation.** It streams audio both ways, handles barge-in and backchannels, and answers small talk on its own.
- **The full Hermes agent does the work.** When something needs tools, files, channels or the web, the voice model hands a task to the agent and speaks its reply. The task and the full reply are also posted to the bound text channel.

Two kinds of model are supported:

| `model` | Bridge | Turn-taking | Delegation |
|---|---|---|---|
| `gpt-realtime-*` | OpenAI Realtime | server-side semantic VAD, truncation on barge-in | an `ask_hermes` function call |
| `gpt-live-*` | GPT-Live (full duplex) | decided by the model | client delegation; the task is built from the recent transcript |

The plugin can also **follow one user**: it joins, moves and leaves voice channels together with them.

## Requirements

- Hermes Agent with the Discord platform set up and working. Tested against **v0.23.0**.
- An OpenAI API key with access to the model you choose.
- The bot needs the *Connect* and *Speak* permissions in voice channels. To open a thread per call, it also needs *Create Public Threads* in the bound text channel.

## Install

```bash
hermes plugins install shohei81/hermes-discord-realtime-voice --enable
```

Put the API key in `~/.hermes/.env`:

```bash
HERMES_REALTIME_OPENAI_API_KEY=sk-...
```

The plugin reads its own variable on purpose. If it read `OPENAI_API_KEY`, adding a key for voice could silently switch other Hermes voice tools (STT/TTS) onto paid OpenAI endpoints.

Turn realtime voice on, then restart the gateway:

```bash
hermes config set plugins.entries.discord-realtime-voice.settings.realtime_enabled true
hermes gateway restart
```

Join a voice channel as usual (`/voice join`, or let the plugin follow you; see below) and start talking.

## Settings

All settings live under `plugins.entries.discord-realtime-voice.settings` in `config.yaml`. They are read on every join, so changes apply from the next call. Set them with `hermes config set plugins.entries.discord-realtime-voice.settings.<key> <value>`.

| Key | Default | Meaning |
|---|---|---|
| `realtime_enabled` | `false` | Use the realtime model when the bot joins a voice channel. When off, or when the session cannot start, Hermes' own STT/TTS voice mode is used. |
| `model` | `gpt-realtime-2.1-mini` | `gpt-live-*` selects the GPT-Live bridge; any other value selects the Realtime bridge. |
| `voice` | `marin` | Output voice. |
| `prompt_preset` | `en` | Built-in voice prompt: `en` or `ja`. |
| `instructions` | `""` | Extra guidance appended to the built-in prompt, such as speaking style. |
| `include_soul` | `true` | Add `SOUL.md` to the voice prompt, so the voice layer speaks as your agent. |
| `include_memory` | `true` | Add the agent's memory snapshots (user and memory) to the voice prompt. |
| `open_thread` | `true` | Open a thread per call in the bound text channel. Delegations then run in a short, fresh session there instead of the channel's long one. |
| `api_key_env` | `HERMES_REALTIME_OPENAI_API_KEY` | Environment variable that holds the API key. |
| `follow_enabled` | `false` | Join, move and leave voice channels together with one user. |
| `follow_user_id` | `""` | User to follow. Empty means the single entry of `DISCORD_ALLOWED_USERS`. |
| `follow_text_channel_id` | `""` | Text channel bound to followed calls. Empty means `DISCORD_HOME_CHANNEL`. |

Example:

```yaml
plugins:
  entries:
    discord-realtime-voice:
      settings:
        realtime_enabled: true
        model: gpt-live-1
        prompt_preset: ja
        follow_enabled: true
        follow_text_channel_id: "123456789012345678"
```

## What happens on a call

1. The bot joins the voice channel through Hermes' normal path. Hermes binds the call to a text channel.
2. With `realtime_enabled`, the plugin stops Hermes' STT listen loop for that call and opens a realtime session. If the session cannot start, the STT loop is restored and the call falls back to Hermes' voice mode.
3. With `open_thread`, a `Voice — MM/DD HH:MM` thread is opened in the bound text channel, and the call is bound to it.
4. Your audio streams to the model. Discord sends nothing while you are silent, so the plugin fills the gaps with silence to keep turn detection working.
5. When the model delegates, the task is posted as `[Voice → Hermes] …` in the thread. The agent runs it in that session and posts its full reply there, and the model speaks a short version.
6. Hermes' text-to-speech replies are suppressed for a channel while a realtime session speaks in it.

Delegated tasks go through the same checks as spoken input: the profile routing and the allowlists (including role grants) for the speaking user. The agent then acts with its usual tools and approvals. Mentions such as `@everyone` in a task are defused before it is posted.

## Compatibility

Hermes has no public extension point for voice-channel audio. To get there, this plugin wraps four methods of Hermes' built-in Discord adapter and gateway runner, and it reads a few of their internal attributes:

- `DiscordAdapter.join_voice_channel`, `leave_voice_channel` and `play_in_voice_channel`
- `GatewayVoiceMixin._should_send_voice_reply`

All of these names are listed in `integration.py` (`REQUIRED`) and checked when the plugin loads. On a Hermes build where any of them is missing, the plugin logs which ones and stays inactive. Hermes and its own voice mode keep working.

The same wrapping is why this plugin cannot be listed in the Hermes plugin catalog: `hermes plugins validate` reports `no core override` and private-import findings. It installs and runs normally from GitHub.

CI runs the test suite against the tested Hermes release and against the latest stable release every week, so breakage from a Hermes update shows up there.

## Development

The tests need an importable Hermes with the Discord extra. One way to set that up:

```bash
git clone --depth 1 --branch v0.23.0 https://github.com/NousResearch/hermes-agent .hermes-src
UV_PROJECT_ENVIRONMENT=.venv uv sync --project .hermes-src --frozen --extra discord
uv pip install --python .venv numpy websockets pytest pytest-asyncio
.venv/bin/python -m pytest
```

- `realtime_voice.py` and `live_voice.py` speak the OpenAI protocols and know nothing about Hermes.
- `integration.py` holds all the wiring into Hermes.
- `prompts.py` holds the built-in voice prompts.

## License

MIT. Parts of `integration.py` are adapted from Hermes Agent (MIT, © Nous Research).
