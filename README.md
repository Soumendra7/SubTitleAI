# Odia Audio to SRT Subtitle Generator

A Python command-line tool that converts an Odia audio file (such as `.mp3` or `.mpeg`) into an `.srt` subtitle file.

The script uses an automatic speech recognition (ASR) model to recognize Odia speech, voice activity detection (VAD) to locate likely speech regions, and audio-level heuristics to label some gaps and background sound.

> **Important:** This README documents the current script `subtitle_tag_agent_tags_fixed(2).py`. The timing and sound labels are heuristic, not human-verified word-level alignment. Review the generated SRT before publishing it.

## Contents

- [What the project uses](#what-the-project-uses)
- [What to install](#what-to-install)
- [Project files](#project-files)
- [Setup](#setup)
- [Convert MPEG or MP3 to SRT](#convert-mpeg-or-mp3-to-srt)
- [How the process works](#how-the-process-works)
- [Subtitle tags](#subtitle-tags)
- [Resources](#resources)
- [Troubleshooting](#troubleshooting)
- [Limitations and privacy](#limitations-and-privacy)

## What the project uses

| Component & Purpose 
| Python-> Runs the script and coordinates the processing steps. 
| AI4Bharat Indic Conformer model on Hugging Face-> Recognizes speech and produces Odia text. The script uses model ID `ai4bharat/indic-conformer-600m-multilingual` and language code `or`. |
| PyTorch (`torch`)-> Runs the speech-recognition model. 
| Hugging Face Transformers-> Loads the model and its inference interface. 
| Silero VAD (`silero-vad`)-> Estimates where speech occurs in the audio. 
| FFmpeg-> Converts the source audio into mono, 16 kHz, 16-bit PCM WAV for processing.
| SoundFile (`soundfile`)-> Reads audio samples into Python. 
| NumPy (`numpy`)-> Supports numerical audio processing. 
| SciPy (`scipy`)-> Supplies FFT/audio-analysis functions used by the script.
| Standard Python modules-> Handle command-line arguments, temporary files, paths, text cleanup, and SRT writing.

### Is ChatGPT used when the script runs?

No. ChatGPT may be used as a development or documentation assistant, but it is not called by the Python script at runtime. The speech recognition is performed by the AI4Bharat model loaded through Hugging Face and PyTorch.

## What to install

You need:

1. **Python** — Python 3.10 or 3.11 is a reasonable starting point. The script's dependencies have not been pinned to exact versions, so test the setup in your own environment.
2. **FFmpeg** — required to convert the input audio. Installing a Python package alone does not necessarily install the FFmpeg command-line program.
3. **Python packages** listed below.
4. **Hugging Face access** — the model repository may require you to sign in and accept its access conditions before downloading it.
5. **Internet access for the first model download** — model files are large. Later runs may reuse the local cache.

A compatible CPU can run inference, but processing can be slow. A supported GPU may improve speed if your PyTorch installation is configured for it.

## Project files

Keep the script and your input audio in the same folder for the simplest commands:

odia-audio-to-srt/
├── subtitle_tag_agent_tags_fixed(2).py
├── sample.mpeg
└── sample.srt                 # created after a successful run

The source audio does not have to be in the same folder if you provide its full or relative path.

## Setup

### 1. Install Python

Install Python from the official Python website and make sure the Python command is available in your terminal.

Check:

```bash
python --version
python -m pip --version
```

On some systems, use `python3` instead of `python`.

### 2. Install FFmpeg

Install FFmpeg for your operating system using the official download page:

https://ffmpeg.org/download.html

Open a **new terminal** after installation, then verify:

```bash
ffmpeg -version
```

If the command is not found, FFmpeg may not be installed correctly or its executable directory may not be on your system's `PATH`.

### 3. Create and activate a virtual environment

Run these commands from the project folder.

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

```bash
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

A virtual environment keeps this project's Python packages separate from other projects.

### 4. Install Python dependencies

Create a file named `requirements.txt` with the following contents:

numpy
soundfile
scipy
torch
silero-vad
transformers
huggingface_hub

Then install them:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

These are unpinned package names, not a tested lockfile. If installation fails, check the package's current installation instructions and your Python/OS compatibility. For PyTorch, follow the official selector if you need a particular CPU or GPU build.

### 5. Get access to the model

1. Create or sign in to a Hugging Face account.
2. Open the model page: https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual
3. Read and accept any access conditions shown on the model page.
4. If the repository requires authentication, sign in through the Hugging Face CLI:

```bash
hf auth login
```

Paste your access token into the CLI prompt. **Do not put the token in a public repository, README, screenshot, or shared code.**

The script also supports an optional `--hf-token` argument and the `HF_TOKEN` environment variable. Environment variables are generally preferable to putting a token directly in a command that may be saved in shell history.

## Convert MPEG or MP3 to SRT

### Basic command

Put the audio file and script in the current folder, activate the virtual environment, and run:

```bash
python "subtitle_tag_agent_tags_fixed(2).py" "sample.mpeg"
```

For an MP3 file:

```bash
python "subtitle_tag_agent_tags_fixed(2).py" "sample.mp3"
```

If the script completes successfully, it writes an SRT file using the input filename with the `.srt` extension. For example:

```text
sample.mpeg  ->  sample.srt
```

### Choose an output filename

```bash
python "subtitle_tag_agent_tags_fixed(2).py" "sample.mpeg" --output "odia_subtitles.srt"
```

### Use a file from another folder

Windows example:

```powershell
python "subtitle_tag_agent_tags_fixed(2).py" "C:\Audio\recording.mpeg" --output "C:\Subtitles\recording.srt"
```

macOS / Linux example:

```bash
python "subtitle_tag_agent_tags_fixed(2).py" "/home/user/audio/recording.mpeg" --output "/home/user/subtitles/recording.srt"
```

Replace these example paths with the actual paths on your computer.

### What happens during the first run?

The script may download the speech-recognition model and related files. The model is about 600 million parameters, so the download can be large and the first run can take longer than later runs. Runtime depends on audio length, CPU/GPU, available memory, and model-loading/download time.

## How the process works

1. **Input:** You provide an audio file such as MP3 or MPEG.
2. **Audio conversion:** FFmpeg converts it to mono, 16 kHz, 16-bit PCM WAV.
3. **Load audio:** SoundFile reads the WAV samples as floating-point audio.
4. **Find speech regions:** Silero VAD estimates which regions likely contain speech.
5. **Recognize speech:** The AI4Bharat Indic Conformer model processes speech chunks using Odia language code `or` and the script's CTC inference path.
6. **Clean the recognized text:** Unicode normalization and text cleanup remove or replace punctuation and unsupported symbols.
7. **Analyze gaps/background:** Audio-level and spectral heuristics classify some gaps as silence or noise and estimate whether speech has background noise/music.
8. **Create subtitle cues:** The script assigns time ranges and applies its available tags.
9. **Write SRT:** The script saves numbered subtitle cues with timestamps in `HH:MM:SS,mmm` format.

See [PROCESS_FLOW.md](PROCESS_FLOW.md) for flowcharts.

## Subtitle tags

The current script can produce these forms:

| Output & Intended meaning in the current implementation 

| `recognized Odia text`-> Speech classified as clean speech. 
| `<NOISE>recognized Odia text</NOISE>`-> Speech detected with possible background noise/music. 
| `<NOISE></NOISE>`-> A gap classified as noise rather than silence. 
| `<SILENT></SILENT>`-> A gap classified as silence by the script's audio thresholds.
| `<FIL>text</FIL>`-> The script's current filler/mumble-text path.

**Implementation note:** The current script's `mb` cue kind is rendered as `<FIL>...</FIL>`. It does not implement a reliable, general detector for every inaudible word and should not be described as automatically identifying all inaudible speech as `<MB></MB>`. Also, silence classification is based on amplitude thresholds; it does not prove that the waveform is perfectly flat.

## Resources

- **Project script:** `subtitle_tag_agent_tags_fixed(2).py` — the script this README describes.
- **AI4Bharat Indic Conformer model:** https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual
- **Hugging Face Transformers installation guide:** https://huggingface.co/docs/transformers/installation
- **Hugging Face CLI / authentication:** https://huggingface.co/docs/huggingface_hub/guides/cli
- **Silero VAD project:** https://github.com/snakers4/silero-vad
- **FFmpeg downloads and installation:** https://ffmpeg.org/download.html
- **PyTorch installation guide:** https://pytorch.org/get-started/locally/
- **Python downloads:** https://www.python.org/downloads/

### Licences and access conditions

Check the licence and access conditions for the model, each dependency, and FFmpeg before redistribution or commercial use. The model repository's current terms are shown on its Hugging Face page. Do not assume that every component has the same licence.

## Troubleshooting

### `ffmpeg` is not recognized

Install FFmpeg and ensure the FFmpeg executable directory is added to your system `PATH`. Close and reopen the terminal, then run `ffmpeg -version`.

### Model access or download error

Open the model page while signed in, accept any access conditions, then authenticate with `hf auth login`. Confirm that the account used by the terminal has access.

### Python package installation fails

- Confirm the active Python version with `python --version`.
- Upgrade pip with `python -m pip install --upgrade pip`.
- Use a virtual environment.
- For PyTorch-specific errors, use the official PyTorch installation selector.
- Read the complete error message; do not assume every install failure has the same cause.

### The output is missing or the script crashes

- Read the final traceback/error message in the terminal.
- Check that the input path exists and the file can be opened.
- Check that FFmpeg works independently.
- Confirm there is enough free disk space and memory for the model.
- Try a short sample file to isolate whether the issue is related to the input length or file format.

### Recognition or timing is inaccurate

The ASR model may misrecognize words, and the script does not perform precise word-level forced alignment. Its duration enforcement can split longer text proportionally across words, so a subtitle boundary may not match the actual spoken word boundary. Review and correct the SRT before relying on it.

## Limitations and privacy

- The script is designed for Odia speech using language code `or`; do not assume it supports every language correctly.
- It is a local command-line script, not a public upload website.
- Model download requires internet access if the model is not already cached.
- Processing speed and memory use depend on your hardware.
- Automatic speech recognition and sound classification can make mistakes.
- Before uploading private or sensitive recordings to any third-party service, understand that service's data-handling rules. In this workflow, the script is run locally, but model files are downloaded from Hugging Face and no claim is made here about every external dependency's telemetry or behaviour.
- Review the output and verify dependency/model licences before public distribution.

## Suggested workflow

1. Install Python and FFmpeg.
2. Create a virtual environment and install dependencies.
3. Sign in to Hugging Face if the model requires it.
4. Run the script on a short sample first.
5. Open the `.srt` in a subtitle editor and review text, timestamps, and tags.
6. Keep the script, dependency list, model link, and documentation together when publishing the project.
