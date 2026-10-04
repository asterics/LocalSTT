#!/usr/bin/env python3
"""
Lokale Sprachtranskription (Deutsch) von PC-Audio unter Windows - CPU-only,
komplett offline nach dem ersten Modelldownload.

MODI
====

1. Live-Modus (Standard):
   Lautsprecher/Kopfhörer-Ausgabe (WASAPI-Loopback) -> "SIE"
   Mikrofon                                    -> "ICH"

   Pipeline:
     Capture -> Mono -> Resample 16 kHz -> Energie-VAD
     -> Whisper oder Parakeet

2. Datei-Modus:
   Mit --file DATEI.mp3 wird die Datei statt Mikrofon/Loopback transkribiert.

   In diesem Modus wird NICHT die Live-VAD verwendet.
   Whisper verwendet seine eingebaute VAD/Segmentierung.
   Parakeet verwendet onnx-asr mit Silero-VAD für lange Dateien.
   siehe auch: https://huggingface.co/spaces/nvidia/parakeet-tdt-0.6b-v3

INSTALLATION
============

Live + Whisper:

    pip install faster-whisper PyAudioWPatch soxr numpy

Zusätzlich für Parakeet:

    pip install "onnx-asr[cpu,hub]"

BEISPIELE
=========

Live, Whisper small:

    python live_transcribe.py --model small

Live, Whisper large-v3-turbo:

    python live_transcribe.py --model large-v3-turbo

Live, Parakeet:

    python live_transcribe.py --model parakeet

MP3 mit Whisper:

    python live_transcribe.py --file talk.mp3 --model large-v3-turbo

MP3 mit Parakeet:

    python live_transcribe.py --file talk.mp3 --model parakeet

MP3 mit Ausgabe-Datei:

    python live_transcribe.py --file talk.mp3 --model parakeet --out talk.txt

Nur Mikrofon:

    python live_transcribe.py --no-loopback

Nur Loopback:

    python live_transcribe.py --no-mic

Geräte anzeigen:

    python live_transcribe.py --list-devices

Offline-Modus:

    python live_transcribe.py --file talk.mp3 --model parakeet --offline

Model-Download beim ersten Start
================================

Beim ersten Start muss das jeweilige Modell ggf. aus Hugging Face
heruntergeladen werden. Danach kann --offline verwendet werden.

WHISPER:
    faster-whisper, CPU, INT8

PARAKEET:
    NVIDIA Parakeet-TDT 0.6B V3 über onnx-asr, CPU, INT8

Parakeet V3 ist multilingual und unterstützt Deutsch.


Auswahl der Audio Ein-/Ausgabegeräte
====================================

Anzeige der verfügbaren Geräte:

python live_transcribe.py --list-devices

dies zeigt verfügbare Mikrophone und Ausgabegeräte, z.B.:

Mikrofone / Eingänge (WASAPI):
   12  Headset (WH-1000XM5 Hands-Free AG Audio)
   15  Microphone (....)

Loopback-Geräte (Ausgabe, was du hörst):
   20  Headphones (WH-1000XM5 Stereo)
   21  Headset (WH-1000XM5 Hands-Free AG Audio)

Beim Start können die Gerätenummern explizit angegeben werden, z.B.:

python live_transcribe.py --mic-device 12 --loopback-device 20 --model parakeet


"""

import argparse
import collections
import os
import queue
import sys
import threading
import time
from datetime import datetime


import numpy as np


SR = 16000
FRAME_S = 0.03
FRAME = int(SR * FRAME_S)


# Typische Whisper-Halluzinationen bei Stille/Rauschen
HALLUCINATIONS = (
    "untertitel der amara.org",
    "untertitel im auftrag",
    "untertitelung des zdf",
    "vielen dank fürs zuschauen",
    "danke fürs zuschauen",
    "copyright wdr",
)


# ============================================================================
# VAD / Äußerungssegmentierung
# ============================================================================

class Segmenter:
    def __init__(self, label, out_q, args):
        self.label = label
        self.out_q = out_q

        self.abs_db = args.threshold_db
        self.margin_db = args.margin_db
        self.hangover = args.silence
        self.min_speech = args.min_speech
        self.max_utt = args.max_utterance

        self.pending = np.zeros(0, dtype=np.float32)

        self.preroll = collections.deque(
            maxlen=int(0.3 / FRAME_S)
        )

        self.frames = []
        self.in_speech = False
        self.silence_frames = 0
        self.speech_frames = 0

        self.noise_db = -60.0
        self.start_wall = 0.0
        self.last_feed = time.monotonic()

    def feed(self, samples):
        self.last_feed = time.monotonic()

        self.pending = np.concatenate(
            [self.pending, samples]
        )

        while len(self.pending) >= FRAME:
            frame = self.pending[:FRAME]
            self.pending = self.pending[FRAME:]
            self._frame(frame)

    def flush_if_idle(self):
        """
        WASAPI-Loopback liefert bei Stille teilweise gar keine Daten.
        Daher wird eine laufende Äußerung per Timeout beendet.
        """
        if (
            self.in_speech
            and time.monotonic() - self.last_feed > self.hangover
        ):
            self._finish()

    def flush(self):
        """Aktive Äußerung explizit abschließen."""
        if self.in_speech:
            self._finish()

    def _frame(self, f):
        db = 20 * np.log10(
            np.sqrt(np.mean(f * f)) + 1e-9
        )

        is_speech = db > max(
            self.abs_db,
            self.noise_db + self.margin_db,
        )

        if not self.in_speech:
            if is_speech:
                self.in_speech = True

                self.frames = (
                    list(self.preroll) + [f]
                )

                self.preroll.clear()
                self.silence_frames = 0
                self.speech_frames = 1

                self.start_wall = (
                    time.time()
                    - len(self.frames) * FRAME_S
                )

            else:
                self.preroll.append(f)

                self.noise_db = max(
                    -80.0,
                    0.95 * self.noise_db
                    + 0.05 * db,
                )

            return

        self.frames.append(f)

        if is_speech:
            self.silence_frames = 0
            self.speech_frames += 1
        else:
            self.silence_frames += 1

        dur = len(self.frames) * FRAME_S

        if (
            self.silence_frames * FRAME_S
            >= self.hangover
            or dur >= self.max_utt
        ):
            self._finish()

    def _finish(self):
        if (
            self.speech_frames * FRAME_S
            >= self.min_speech
        ):
            audio = np.concatenate(
                self.frames
            ).astype(np.float32)

            self.out_q.put(
                (
                    self.start_wall,
                    self.label,
                    audio,
                )
            )

        self.frames = []
        self.in_speech = False
        self.silence_frames = 0
        self.speech_frames = 0


# ============================================================================
# Audio-Capture / WASAPI
# ============================================================================

class Capture(threading.Thread):
    def __init__(
        self,
        pa,
        pyaudio,
        dev,
        label,
        out_q,
        args,
    ):
        super().__init__(daemon=True)

        import soxr

        self.label = label
        self.raw_q = queue.Queue()
        self.stop_evt = threading.Event()

        self.rate = int(
            dev["defaultSampleRate"]
        )

        self.ch = int(
            dev["maxInputChannels"]
        )

        self.seg = Segmenter(
            label,
            out_q,
            args,
        )

        self.resampler = (
            None
            if self.rate == SR
            else soxr.ResampleStream(
                self.rate,
                SR,
                1,
                dtype="float32",
            )
        )

        def cb(
            in_data,
            frame_count,
            time_info,
            status,
        ):
            self.raw_q.put(
                np.frombuffer(
                    in_data,
                    dtype=np.float32,
                ).copy()
            )

            return (
                None,
                pyaudio.paContinue,
            )

        self.stream = pa.open(
            format=pyaudio.paFloat32,
            channels=self.ch,
            rate=self.rate,
            input=True,
            input_device_index=dev["index"],
            frames_per_buffer=int(
                self.rate * 0.05
            ),
            stream_callback=cb,
        )

        print(
            f"  [{label}] {dev['name']} "
            f"({self.rate} Hz, {self.ch} ch)"
        )

    def run(self):
        self.stream.start_stream()

        while not self.stop_evt.is_set():
            try:
                x = self.raw_q.get(
                    timeout=0.2
                )

            except queue.Empty:
                self.seg.flush_if_idle()
                continue

            mono = (
                x.reshape(-1, self.ch)
                .mean(axis=1)
                .astype(np.float32)
            )

            if self.resampler is not None:
                mono = self.resampler.resample_chunk(
                    np.ascontiguousarray(mono)
                )

            if len(mono):
                self.seg.feed(mono)

    def stop(self):
        self.stop_evt.set()

        try:
            self.stream.stop_stream()
            self.stream.close()
        except Exception:
            pass


# ============================================================================
# Geräte
# ============================================================================

def list_devices(pa):
    print("\nMikrofone / Eingänge (WASAPI):")

    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)

        if (
            d["maxInputChannels"] > 0
            and not d.get("isLoopbackDevice")
            and pa.get_host_api_info_by_index(
                d["hostApi"]
            )["name"] == "Windows WASAPI"
        ):
            print(
                f"  {d['index']:3d}  {d['name']}"
            )

    print(
        "\nLoopback-Geräte "
        "(Ausgabe, was du hörst):"
    )

    for d in pa.get_loopback_device_info_generator():
        print(
            f"  {d['index']:3d}  {d['name']}"
        )


def default_mic(pa, pyaudio):
    wasapi = pa.get_host_api_info_by_type(
        pyaudio.paWASAPI
    )

    return pa.get_device_info_by_index(
        wasapi["defaultInputDevice"]
    )


# ============================================================================
# Hilfsfunktionen
# ============================================================================

def ts(t):
    """
    Unix timestamp -> HH:MM:SS
    """
    return datetime.fromtimestamp(t).strftime(
        "%H:%M:%S"
    )


def duration_ts(seconds):
    """
    Sekunden -> HH:MM:SS
    """
    seconds = max(0, float(seconds))

    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)

    return f"{h:02d}:{m:02d}:{s:02d}"


def write_line(outfile, line):
    print(line, flush=True)

    if outfile:
        outfile.write(line + "\n")
        outfile.flush()


def contains_hallucination(text):
    lower = text.lower()

    return any(
        h in lower
        for h in HALLUCINATIONS
    )


# ============================================================================
# Modell laden
# ============================================================================

def load_model(args):
    """
    Liefert:
        model
        engine

    engine:
        "whisper"
        "parakeet"
    """

    if args.model.lower() == "parakeet":

        try:
            import onnx_asr
        except ImportError:
            print(
                "\nParakeet benötigt onnx-asr.\n"
                "Installation:\n\n"
                '  pip install "onnx-asr[cpu,hub]"\n'
            )
            raise

        print(
            "Lade NVIDIA Parakeet-TDT "
            "0.6B V3 (CPU, INT8) ..."
        )

        # onnx-asr unterstützt die quantisierte
        # Parakeet-TDT-V3-Version direkt.
        model = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v3",
            quantization="int8",
            providers=[
                "CPUExecutionProvider"
            ],
        )

        return model, "parakeet"

    else:

        from faster_whisper import WhisperModel

        print(
            f"Lade Whisper-Modell "
            f"'{args.model}' "
            "(CPU, INT8) ..."
        )

        model = WhisperModel(
            args.model,
            device="cpu",
            compute_type="int8",
            cpu_threads=args.threads,
        )

        return model, "whisper"


# ============================================================================
# Whisper - Live
# ============================================================================

def transcribe_whisper_live(
    model,
    audio,
    args,
    context,
):
    segments, _ = model.transcribe(
        audio,
        language=args.language,
        beam_size=args.beam,
        temperature=0.0,

        vad_filter=True,

        vad_parameters=dict(
            min_silence_duration_ms=300
        ),

        condition_on_previous_text=False,

        initial_prompt=context,

        no_speech_threshold=0.6,
    )

    text = " ".join(
        s.text.strip()
        for s in segments
    ).strip()

    return text


# ============================================================================
# Parakeet - Live
# ============================================================================

def transcribe_parakeet_live(
    model,
    audio,
):
    """
    Parakeet bekommt bereits durch unsere
    Live-VAD geschnittene Audiosegmente.

    Das Ergebnis von onnx-asr ist ein
    TimestampedResult bzw. ein normales
    Ergebnis, abhängig von der Modellkonfiguration.
    """

    result = model.recognize(
        audio,
        sample_rate=SR,
    )

    # Falls recognize() eine Liste/Iteration
    # zurückgibt, Text daraus zusammensetzen.
    if isinstance(result, str):
        return result.strip()

    try:
        return " ".join(
            str(r.text).strip()
            for r in result
            if getattr(r, "text", None)
        ).strip()
    except TypeError:
        return str(result).strip()


# ============================================================================
# MP3 / Datei - Audio laden
# ============================================================================

def load_audio_16k(path):
    """
    Dekodiert beliebige Audioformate (mp3, m4a, flac, wav, ...)
    nach 16 kHz / mono / float32 (Werte -1..1).

    Warum: onnx-asr (Parakeet) kann Dateien nur als PCM-WAV lesen
    ("file does not start with RIFF id" bei MP3), und faster-whisper
    ruft av.open(..., metadata_errors=...) auf, was PyAV >= 19 nicht
    mehr kennt. Mit eigener Dekodierung via PyAV funktionieren beide
    Engines mit jedem PyAV-Stand.
    """
    import av

    chunks = []

    with av.open(path) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=SR,
        )

        for frame in container.decode(stream):
            for f in resampler.resample(frame):
                chunks.append(f.to_ndarray().reshape(-1))

        # Reste aus dem Resampler herausspülen
        for f in resampler.resample(None):
            chunks.append(f.to_ndarray().reshape(-1))

    if not chunks:
        raise RuntimeError(
            f"Keine Audiodaten in Datei gefunden: {path}"
        )

    return (
        np.concatenate(chunks).astype(np.float32) / 32768.0
    )


# ============================================================================
# MP3 / Datei - Whisper
# ============================================================================

def transcribe_file_whisper(
    model,
    filename,
    args,
    outfile,
):
    print(
        f"\nTranskribiere Datei mit "
        f"Whisper '{args.model}':"
    )

    print(f"  {filename}\n")

    start_time = time.monotonic()

    print("Dekodiere Audio ...")
    audio = load_audio_16k(filename)
    print(f"  Dauer: {duration_ts(len(audio) / SR)}\n")

    segments, info = model.transcribe(
        audio,

        language=args.language,

        beam_size=args.beam,

        temperature=0.0,

        vad_filter=True,

        vad_parameters=dict(
            min_silence_duration_ms=500
        ),

        condition_on_previous_text=True,

        no_speech_threshold=0.6,

        # Für Offline-Dateien sind
        # Wort-Timestamps nützlich.
        word_timestamps=False,
    )

    # WICHTIG:
    # faster-whisper liefert einen Generator.
    # Erst durch Iteration wird die eigentliche
    # Transkription ausgeführt.
    count = 0

    for segment in segments:
        text = segment.text.strip()

        if not text:
            continue

        if contains_hallucination(text):
            continue

        start = segment.start
        end = segment.end

        line = (
            f"[{duration_ts(start)} - "
            f"{duration_ts(end)}] "
            f"{text}"
        )

        write_line(outfile, line)

        count += 1

    elapsed = time.monotonic() - start_time

    print(
        f"\nWhisper fertig. "
        f"Segmente: {count}. "
        f"Rechenzeit: {elapsed:.1f} s."
    )

    if info:
        print(
            f"Erkannte Sprache: "
            f"{info.language} "
            f"({info.language_probability:.2f})"
        )


# ============================================================================
# MP3 / Datei - Parakeet
# ============================================================================

def transcribe_file_parakeet(
    model,
    filename,
    args,
    outfile,
):
    """
    Parakeet über onnx-asr.

    Die Datei wird über onnx-asr verarbeitet.
    Für lange Dateien verwenden wir Silero-VAD,
    damit die 20-30-Sekunden-Grenze einzelner
    Modellfenster kein Problem darstellt.

    Die VAD liefert SegmentResults mit
    Start-/End-Zeitpunkten.
    """

    import onnx_asr

    print(
        "\nTranskribiere Datei mit "
        "Parakeet-TDT 0.6B V3:"
    )

    print(f"  {filename}\n")

    start_time = time.monotonic()

    # Silero-VAD laden.
    print("Lade Silero VAD ...")

    vad = onnx_asr.load_vad(
        "silero"
    )

    # Parakeet mit VAD kombinieren.
    #
    # Dadurch kann onnx-asr lange Dateien
    # in Sprachsegmente aufteilen.
    vad_model = (
        model
        .with_vad(vad)
    )

    # onnx-asr liest als Pfad nur PCM-WAV. Deshalb MP3 & Co.
    # selbst dekodieren und das float32-Array übergeben.
    print("Dekodiere Audio ...")
    audio = load_audio_16k(filename)
    print(f"  Dauer: {duration_ts(len(audio) / SR)}\n")

    results = vad_model.recognize(
        audio,
        sample_rate=SR,
    )

    count = 0

    for result in results:

        text = getattr(
            result,
            "text",
            "",
        )

        if not text:
            continue

        text = str(text).strip()

        if not text:
            continue

        # SegmentResult kann je nach VAD-Version
        # begin/end oder start/end bereitstellen.
        begin = getattr(
            result,
            "start",
            getattr(result, "begin", None),
        )

        end = getattr(
            result,
            "end",
            None,
        )

        if begin is not None and end is not None:
            line = (
                f"[{duration_ts(begin)} - "
                f"{duration_ts(end)}] "
                f"{text}"
            )
        else:
            line = text

        write_line(outfile, line)

        count += 1

    elapsed = time.monotonic() - start_time

    print(
        f"\nParakeet fertig. "
        f"Segmente: {count}. "
        f"Rechenzeit: {elapsed:.1f} s."
    )


# ============================================================================
# Datei-Modus
# ============================================================================

def transcribe_file(
    model,
    engine,
    filename,
    args,
    outfile,
):
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            f"Datei nicht gefunden: {filename}"
        )

    if engine == "whisper":
        transcribe_file_whisper(
            model,
            filename,
            args,
            outfile,
        )

    elif engine == "parakeet":
        transcribe_file_parakeet(
            model,
            filename,
            args,
            outfile,
        )

    else:
        raise RuntimeError(
            f"Unbekannte ASR-Engine: {engine}"
        )


# ============================================================================
# Live-Modus
# ============================================================================

def live_mode(
    model,
    engine,
    pa,
    pyaudio,
    args,
    outfile,
):
    utter_q = queue.Queue()

    captures = []

    print(
        "\nStarte Audio-Capture:"
    )

    try:

        # ---------------------------------------------------------------
        # Loopback = SIE
        # ---------------------------------------------------------------

        if not args.no_loopback:

            dev = (
                pa.get_device_info_by_index(
                    args.loopback_device
                )
                if args.loopback_device is not None
                else pa.get_default_wasapi_loopback()
            )

            captures.append(
                Capture(
                    pa,
                    pyaudio,
                    dev,
                    "SIE",
                    utter_q,
                    args,
                )
            )

        # ---------------------------------------------------------------
        # Mikrofon = ICH
        # ---------------------------------------------------------------

        if not args.no_mic:

            dev = (
                pa.get_device_info_by_index(
                    args.mic_device
                )
                if args.mic_device is not None
                else default_mic(
                    pa,
                    pyaudio,
                )
            )

            captures.append(
                Capture(
                    pa,
                    pyaudio,
                    dev,
                    "ICH",
                    utter_q,
                    args,
                )
            )

    except Exception as e:

        print(
            f"\nFehler beim Öffnen "
            f"der Audiogeräte: {e}"
        )

        print(
            "\nTipp: "
            "--list-devices verwenden."
        )

        pa.terminate()
        return

    if not captures:

        print(
            "Kein Stream aktiv."
        )

        pa.terminate()
        return

    context = {}

    for c in captures:
        c.start()

    print(
        "\nBereit - sprich oder spiele "
        "Audio ab."
    )

    print(
        "Beenden mit Strg+C.\n"
    )

    try:

        while True:

            try:
                (
                    start_wall,
                    label,
                    audio,
                ) = utter_q.get(
                    timeout=0.5
                )

            except queue.Empty:
                continue

            t0 = time.monotonic()

            # -------------------------------------------------------
            # Whisper
            # -------------------------------------------------------

            if engine == "whisper":

                text = (
                    transcribe_whisper_live(
                        model,
                        audio,
                        args,
                        context.get(label),
                    )
                )

            # -------------------------------------------------------
            # Parakeet
            # -------------------------------------------------------

            else:

                text = (
                    transcribe_parakeet_live(
                        model,
                        audio,
                    )
                )

            elapsed = (
                time.monotonic() - t0
            )

            if not text:
                continue

            if contains_hallucination(text):
                continue

            context[label] = text[-200:]

            audio_duration = (
                len(audio) / SR
            )

            lag = (
                time.time()
                - start_wall
                - audio_duration
            )

            line = (
                f"[{ts(start_wall)}] "
                f"{label:3s}: "
                f"{text}"
            )

            write_line(
                outfile,
                line,
            )

            # -------------------------------------------------------
            # Performance information
            # -------------------------------------------------------

            if lag > 20:

                print(
                    f"   (Rückstand ~{lag:.0f}s "
                    f"- kleineres Modell oder "
                    f"--beam 1 verwenden)",
                    file=sys.stderr,
                )

            print(
                f"   ASR: {elapsed:.2f}s "
                f"für {audio_duration:.2f}s Audio "
                f"({audio_duration / max(elapsed, 0.001):.1f}x)"
                ,
                file=sys.stderr,
            )

    except KeyboardInterrupt:

        print(
            "\nBeende ..."
        )

    finally:

        for c in captures:
            c.stop()

        pa.terminate()


# ============================================================================
# MAIN
# ============================================================================

def main():

    ap = argparse.ArgumentParser(
        description=(
            "Lokale deutsche Sprachtranskription "
            "für Windows, CPU-only"
        )
    )

    # -----------------------------------------------------------------------
    # ASR
    # -----------------------------------------------------------------------

    ap.add_argument(
        "--model",
        default="large-v3-turbo",
        help=(
            "Whisper-Modell: "
            "tiny|base|small|medium|"
            "large-v3-turbo|large-v3 "
            "oder 'parakeet' "
            "(Standard: large-v3-turbo)"
        ),
    )

    ap.add_argument(
        "--language",
        default="de",
        help="Whisper-Sprache (Standard: de)",
    )

    ap.add_argument(
        "--beam",
        type=int,
        default=1,
        help=(
            "Whisper Beam-Size "
            "(1 = schnell, höhere Werte "
            "= potentiell genauer)"
        ),
    )

    ap.add_argument(
        "--threads",
        type=int,
        default=0,
        help=(
            "Whisper CPU-Threads "
            "(0 = automatisch)"
        ),
    )

    # -----------------------------------------------------------------------
    # Datei
    # -----------------------------------------------------------------------

    ap.add_argument(
        "--file",
        metavar="DATEI",
        help=(
            "Vorhandene Audio-/MP3-Datei "
            "transkribieren statt Live-Audio"
        ),
    )

    # -----------------------------------------------------------------------
    # Geräte
    # -----------------------------------------------------------------------

    ap.add_argument(
        "--list-devices",
        action="store_true",
    )

    ap.add_argument(
        "--loopback-device",
        type=int,
        help="Geräteindex für Loopback",
    )

    ap.add_argument(
        "--mic-device",
        type=int,
        help="Geräteindex für Mikrofon",
    )

    ap.add_argument(
        "--no-mic",
        action="store_true",
        help="Nur Loopback",
    )

    ap.add_argument(
        "--no-loopback",
        action="store_true",
        help="Nur Mikrofon",
    )

    # -----------------------------------------------------------------------
    # Live-VAD
    # -----------------------------------------------------------------------

    ap.add_argument(
        "--threshold-db",
        type=float,
        default=-48.0,
        help=(
            "Absolute VAD-Schwelle in dBFS"
        ),
    )

    ap.add_argument(
        "--margin-db",
        type=float,
        default=10.0,
        help=(
            "Abstand zum Rauschteppich in dB"
        ),
    )

    ap.add_argument(
        "--silence",
        type=float,
        default=0.7,
        help=(
            "Stille in Sekunden, "
            "die eine Äußerung beendet"
        ),
    )

    ap.add_argument(
        "--min-speech",
        type=float,
        default=0.3,
        help=(
            "Minimale Sprachdauer"
        ),
    )

    ap.add_argument(
        "--max-utterance",
        type=float,
        default=15.0,
        help=(
            "Maximale Äußerungslänge "
            "im Live-Modus"
        ),
    )

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------

    ap.add_argument(
        "--out",
        default="protokoll.txt",
        help=(
            "Transkript zusätzlich in "
            "UTF-8-Datei schreiben. "
            "Mit --out '' deaktivieren."
        ),
    )

    # -----------------------------------------------------------------------
    # Offline
    # -----------------------------------------------------------------------

    ap.add_argument(
        "--offline",
        action="store_true",
        help=(
            "HF_HUB_OFFLINE=1: "
            "kein Netzzugriff zum Nachladen "
            "von Modellen"
        ),
    )

    args = ap.parse_args()

    # -----------------------------------------------------------------------
    # Offline
    # -----------------------------------------------------------------------

    if args.offline:

        os.environ[
            "HF_HUB_OFFLINE"
        ] = "1"

        os.environ[
            "TRANSFORMERS_OFFLINE"
        ] = "1"

    # -----------------------------------------------------------------------
    # UTF-8
    # -----------------------------------------------------------------------

    try:
        sys.stdout.reconfigure(
            encoding="utf-8"
        )
    except Exception:
        pass

    # -----------------------------------------------------------------------
    # Modell laden
    # -----------------------------------------------------------------------

    try:
        model, engine = load_model(args)

    except Exception as e:

        print(
            "\nFehler beim Laden des "
            f"Modells:\n{e}"
        )

        return 1

    print(
        f"\nASR-Engine: {engine}"
    )

    if engine == "whisper":
        print(
            f"Whisper-Modell: "
            f"{args.model}"
        )

    else:
        print(
            "Parakeet-TDT 0.6B V3: "
            "INT8 / CPU"
        )

    # -----------------------------------------------------------------------
    # Output-Datei
    # -----------------------------------------------------------------------

    outfile = None

    if args.out:

        try:

            outfile = open(
                args.out,
                "a",
                encoding="utf-8",
            )

        except Exception as e:

            print(
                f"Fehler beim Öffnen "
                f"der Ausgabedatei: {e}"
            )

            return 1

    try:

        # ================================================================
        # DATEI-MODUS
        # ================================================================

        if args.file:

            print(
                f"\nDatei-Modus: "
                f"{args.file}"
            )

            transcribe_file(
                model,
                engine,
                args.file,
                args,
                outfile,
            )

            return 0

        # ================================================================
        # LIVE-MODUS
        # ================================================================

        # PyAudioWPatch nur laden, wenn
        # tatsächlich Live-Modus verwendet wird.

        import pyaudiowpatch as pyaudio

        pa = pyaudio.PyAudio()

        if args.list_devices:

            list_devices(pa)

            pa.terminate()

            return 0

        live_mode(
            model,
            engine,
            pa,
            pyaudio,
            args,
            outfile,
        )

        return 0

    except KeyboardInterrupt:

        print(
            "\nAbgebrochen."
        )

        return 0

    except Exception as e:

        print(
            "\nFehler:"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        return 1

    finally:

        if outfile:
            outfile.close()


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
