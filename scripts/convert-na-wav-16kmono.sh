# 1. Download Na recordings from Lacito Pangloss.
# 2. Convert mp3 to WAV using Audacity or something similar.
# 3. Convert to 16k Hz
ffmpeg -i $1 -ar 16000 16k$1
# 4. Convert from stereo to mono
ffmpeg -i 16k$1 -ac 1 mono16k$1
