README — Odia Speech-to-Subtitle Tagging Script
=================================================

OVERVIEW
--------
This Python script converts an audio or video file containing Odia speech into a Subtitle Edit-compatible SRT subtitle file. It uses AI4Bharat's Indic Conformer model for speech recognition and Silero VAD to detect speech regions.

The output is a timestamped .srt file with text and tags for silence, background noise/music, and filler or mumbling sounds. This is not a plain-text transcription-only script.

FEATURES
--------
- Transcribes speech in Odia script.
- Converts input media to mono, 16 kHz PCM WAV using FFmpeg.
- Detects speech intervals with Silero Voice Activity Detection (VAD).
- Splits speech into chunks for recognition.
- Creates timestamped SRT subtitles.
- Uses the following tags:
  <SILENT></SILENT>   — detected silence.
  <NOISE></NOISE>     — detected background noise or music without recognized speech.
  <FIL>...</FIL>      — recognized filler/mumbling sounds.
  <NOISE>speech</NOISE> — speech classified as having background noise/music.
- Writes output as UTF-8 text.

The silence and background-noise/music checks are heuristic estimates and can be wrong. Speech recognition may also make mistakes, so review the output when accuracy matters.

REQUIREMENTS
------------
- Python 3.10 or 3.11 recommended.
- FFmpeg installed and available from the command line (PATH).
- Internet connection for the initial model download.
- Enough RAM to load the ASR model. The script loads the model on CPU, which may be slow.

PYTHON PACKAGES
---------------
The script imports:
- numpy
- soundfile
- scipy
- torch
- silero-vad
- transformers

WINDOWS SETUP (PowerShell)
--------------------------
1. Open a terminal in your project folder.
2. Create and activate a virtual environment:

   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1

3. Upgrade pip and install dependencies:

   python -m pip install --upgrade pip
   python -m pip install numpy soundfile scipy torch silero-vad transformers

If PowerShell blocks activation, use Command Prompt and run:
   .venv\Scripts\activate.bat

MACOS / LINUX SETUP
-------------------
Create and activate a virtual environment, then install packages:

   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install numpy soundfile scipy torch silero-vad transformers

For a specific CPU/CUDA PyTorch build, use the official PyTorch installation selector.

INSTALL FFMPEG
--------------
FFmpeg is a separate system program; installing Python packages does not install it.
Install FFmpeg for your operating system, ensure its executable is on PATH, and verify:

   ffmpeg -version

If the command is not found, update PATH and reopen your terminal.

HUGGING FACE MODEL CONFIGURATION
--------------------------------
Default model:
   ai4bharat/indic-conformer-600m-multilingual

The script uses language code "or" for Odia and loads the model with trust_remote_code=True.

1. Sign in to or create a Hugging Face account.
2. Visit the model page:
   https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual
3. Review and accept any access conditions displayed on the model page.
4. If authentication is required, create a Hugging Face Read token.
5. Log in in the terminal:

   hf auth login

Paste the token when prompted. If the "hf" command is unavailable, update Hugging Face Hub:

   python -m pip install --upgrade huggingface_hub

Alternatively, set the HF_TOKEN environment variable or pass --hf-token to the script. Keep tokens private; do not hard-code them in source code or publish them.

RUN THE SCRIPT
--------------
Save the Python script as subtitle_tag_agent_tags_fixed.py, or use the filename you already have.

Basic usage (output defaults to the same basename with .srt extension):

   python subtitle_tag_agent_tags_fixed.py input_audio.wav

Choose an output path:

   python subtitle_tag_agent_tags_fixed.py input_audio.mp3 --output odia_subtitles.srt

Specify the model explicitly:

   python subtitle_tag_agent_tags_fixed.py input_audio.wav --model ai4bharat/indic-conformer-600m-multilingual

Provide a Hugging Face token explicitly:

   python subtitle_tag_agent_tags_fixed.py input_audio.wav --hf-token YOUR_READ_TOKEN

For security, prefer `hf auth login` or the HF_TOKEN environment variable over placing a real token in shell history.

Show available command-line options:

   python subtitle_tag_agent_tags_fixed.py --help

The input can be an audio or video format supported by your FFmpeg installation.

PROCESSING STEPS
----------------
1. Loads the AI4Bharat ASR model.
2. Converts the input media to 16 kHz mono WAV.
3. Loads the waveform and estimates its duration and ambient noise floor.
4. Detects speech using Silero VAD.
5. Merges nearby speech intervals and splits long regions into chunks.
6. Transcribes chunks using the model in CTC mode.
7. Classifies possible filler sounds and estimates background noise/music.
8. Fills timeline gaps and adjusts subtitle cue durations.
9. Writes the tagged SRT file.

TROUBLESHOOTING
---------------
"ffmpeg was not found"
- Install FFmpeg, add it to PATH, reopen the terminal, and test with `ffmpeg -version`.

Hugging Face 401, 403, or gated-model error
- Open the model page and accept any required conditions.
- Run `hf auth login` with a valid Read token.

Model download/network error
- Check your internet connection and retry. The first download can take time because the model is large.

Missing Python module
- Activate the correct virtual environment and install the missing package, for example:
  python -m pip install scipy

Incorrect transcription or tags
- Results depend on audio quality, accents, overlapping speakers, and background sound. The tag detector is heuristic and can misclassify audio. Review the SRT manually for important use cases.

PRIVACY AND SECURITY
--------------------
The script runs inference locally in your Python process after the model is downloaded; it does not call a hosted transcription API. Model files are downloaded from Hugging Face as needed. Keep access tokens private and never commit them to a public repository.

REFERENCES
----------
AI4Bharat model:
https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual

FFmpeg:
https://ffmpeg.org/

Python:
https://www.python.org/

PyTorch:
https://pytorch.org/

Silero VAD:
https://github.com/snakers4/silero-vad
