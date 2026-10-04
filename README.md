# LocalSTT - Local Live Transcription 

Transcribes a running conversation in real time, **entirely on your PC**,
using multilingual/german STT models, and not requiring a GPU (CPU-only).
(Note that the audio capture toolflow currently only supports Windows).


- **System output** (what you hear in speakers/headphones, via WASAPI loopback) → labeled `SIE`
- **Microphone** → labeled `ICH`

Voice activity detection splits each stream into utterances, and [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (int8, CPU) transcribes them. Typical delay: 1–3 seconds after a speaker pauses.

```
[14:32:07] SIE: Guten Morgen zusammen, wir fangen gleich an.
[14:32:11] ICH: Morgen, ich bin da.
```

## Files

| File | Purpose |
|---|---|
| `live_transcribe.py` | The demo itself |
| `run.bat` | Creates `.venv`, installs dependencies (first run only), starts the demo |
| `README.md` | This file |

## Requirements

- Windows 10/11
- Python 3.10–3.12 from [python.org](https://www.python.org/downloads/) (tick **"Add python.exe to PATH"**)
- ~2 GB free disk space (dependencies + model) and an internet connection **for the first run only**
- No GPU needed

## Installation

1. Put `live_transcribe.py`, `run.bat` and `README.md` in one folder.
2. Double-click `run.bat`.

On the first start it will:
1. create a virtual environment in `.\.venv`,
2. install `faster-whisper`, `PyAudioWPatch`, `soxr`, `numpy` and `onnx-asr[cpu,hub]`
3. start the demo, which downloads the selected model once.

Later starts skip steps 1–2 and begin immediately.

## Usage

Double-click `run.bat`, or run it from a terminal and pass options:

```
run.bat                              # mic + system audio, model "small"
run.bat --list-devices               # show audio devices and their indexes
run.bat --model medium               # more accurate, slower
run.bat --model large-v3-turbo       # best quality, needs a fast CPU
run.bat --no-mic                     # only the other side (loopback)
run.bat --no-loopback                # only your microphone
run.bat --out protokoll.txt          # also save the transcript (UTF-8)
run.bat --offline                    # never contact the network for models
run.bat --file <mp3-file>            # transcribe audio from mp3 file
```

Stop with **Ctrl+C**.

### All options

| Option | Default | Meaning |
|---|---|---|
| `--model` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3-turbo`, `large-v3`, `parakeet` or a local model folder |
| `--language` | `de` | Language code |
| `--beam` | `1` | Beam size; 1 is fastest, 5 slightly more accurate |
| `--threads` | `0` (auto) | CPU threads for the model |
| `--loopback-device N` | default output | Device index from `--list-devices` |
| `--mic-device N` | default input | Device index from `--list-devices` |
| `--no-mic` / `--no-loopback` | off | Disable one stream |
| `--threshold-db` | -48` | Minimum loudness (dBFS) counted as speech |
| `--margin-db` | `10` | Required distance above the noise floor |
| `--silence` | `0.7` | Seconds of silence that end an utterance |
| `--min-speech` | `0.3` | Ignore utterances shorter than this (s) |
| `--max-utterance` | `15` | Force a cut after this many seconds |
| `--out FILE` | – | Append transcript to a file |
| `--offline` | off | Set `HF_HUB_OFFLINE=1` |
| `--file <filename.mp3>` | - | use mp3file for the audio instead of live capture |

## Choosing a model (CPU)

| Model | Speed on CPU | German quality |
|---|---|---|
| `base` | very fast | basic |
| `small` | fast (default) | good |
| `medium` | around real-time on a modern 8-core CPU | very good |
| `large-v3-turbo` | slower, needs a strong CPU | best |
| `parakeet` | uses NVIDIA Parakeet-TDT 0.6B V3 | best |

If the console shows **"Rückstand"** (backlog) warnings, the model is too heavy for your CPU. Use a smaller one. Start with `small` and move up only if it keeps up.

## Model storage 

Per default faster-whisper downloads the models through the Hugging Face Hub, which uses a per-user cache:

```
C:\Users\<YourName>\.cache\huggingface\hub
```

You can open it with `explorer %USERPROFILE%\.cache\huggingface\hub`.

Each model gets its own folder, for example `models--Systran--faster-whisper-small`. The actual files are in `snapshots\<hash>\` inside it. Approximate sizes are:

| Model | Size |
|---|---|
| `base` | ~150 MB |
| `small` | ~500 MB |
| `medium` | ~1.5 GB |
| `large-v3-turbo` | ~1.6 GB |
| `large-v3` | ~3 GB |
| `parakeet` | ~700 MB |

**Changing the location:** set `HF_HOME` (or `HF_HUB_CACHE`) before the script starts. 
In `run.bat`, this line is added at the beginning:

```bat
set "HF_HOME=%~dp0models"
```

so that the models live in `models\` next to the script. That makes the setup portable and easy to delete. 

**Removing a model:** delete its `models--...` folder. Deleting `.venv` does not remove any models.

**Offline use:** with `--offline`, the script only uses what is already in this cache and fails with an error if the model isn't there. Run once without `--offline` to download the model first.



## Privacy

- All processing runs locally. After the first model download, start with `--offline` (or block `python.exe` in the Windows firewall) to guarantee that nothing leaves your PC.
- Audio is kept in RAM only. Nothing is written to disk except the transcript if you use `--out`. Store that file somewhere appropriate (e.g. an encrypted folder) and delete it when no longer needed.
- **Legal note:** transcribing a call or meeting counts as recording in many jurisdictions (including Austria and the EU). Inform the other participants and get their consent.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Python 3 was not found` | Install Python and tick "Add to PATH", then rerun `run.bat` |
| Nothing is transcribed from the other side | Run `run.bat --list-devices` and pass the correct `--loopback-device N`. Make sure audio is actually playing on the default output device |
| Mic and system audio are both picked up twice | You are using speakers. Wear headphones, or the mic will hear the remote voice |
| Nothing from the mic / wrong mic | Use `--mic-device N`, and check the Windows microphone privacy setting ("Let desktop apps access your microphone") |
| Background noise creates phantom text | Raise `--threshold-db` (e.g. `-42`) or `--margin-db` (e.g. `14`) |
| Quiet speech is missed | Lower `--threshold-db` (e.g. `-52`) |
| Sentences are cut in the middle | Increase `--silence` (e.g. `1.0`) |
| Install fails for `PyAudioWPatch` | Use Python 3.10–3.12 (64-bit) |
| Reinstall from scratch | Delete the `.venv` folder and run `run.bat` again |

## Known limitations

- All remote participants share the label `SIE`. Separating them needs speaker diarization (not included).
- Text appears once per utterance, not word by word.
- Whisper can occasionally hallucinate on noise or music. The script filters the most common German phantom phrases.
