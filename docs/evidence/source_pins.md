# Download pins for the public speech sources (`xtts_trainer/speech_data.py`)

Every source file is fetched from an immutable URL and checked against a SHA-1 before use.

## LibriVox, The Art of War (archive.org)

The SHA-1 values are the ones archive.org publishes in the item metadata:
`curl -sL https://archive.org/metadata/art_of_war_librivox` -> `files[].sha1` for the seven `*_64kb.mp3` files.
License: `metadata.licenseurl` = `http://creativecommons.org/licenses/publicdomain/`; single reader (Moira Fogarty,
per the item description). The first file was downloaded and checked: 8.45 min, mono, 22050 Hz, SHA-1 matches.

## LJSpeech sample and JFK (GitHub raw)

These were downloaded from moving branches (`coqui-ai/TTS@dev`, `openai/whisper@main`). They are now pinned to
commits, and their SHA-1 values were computed from downloads of those exact URLs:

```
$ gh api repos/coqui-ai/TTS/commits/dev --jq '.sha'
dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e
$ gh api repos/openai/whisper/commits/main --jq '.sha'
86098128c0b4f24f0e2aa2994de830614b474227

$ LJ=https://raw.githubusercontent.com/coqui-ai/TTS/dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e/tests/data/ljspeech
$ for f in metadata.csv wavs/LJ001-000{1..6}.wav; do curl -sfL -o "$(basename $f)" "$LJ/$f"; done; shasum -a 1 ...
247ef0626c1ec7f4d01e8bc8599c4b070a396209  metadata.csv        (1655 bytes)
a4f9d63b0cfbd36eb1c23b7b122f213bc587ecc4  LJ001-0001.wav      (425830)
6249e7e3edd982e8cb79a8b8deef4c907374e1fc  LJ001-0002.wav      (83814)
fc4d535b7b0a1ba2ea09dca3aad2fc9e2816a19c  LJ001-0003.wav      (426342)
41a5fe1e72f423534158d54cc9bfca920921c045  LJ001-0004.wav      (226662)
ccef55335f89ed4f21c4e0091a1aaf34b2bcbe11  LJ001-0005.wav      (357734)
6fafc6a2bb5dedf373cdace60bd6746a89548153  LJ001-0006.wav      (250726)

$ curl -sfL -o jfk.flac https://raw.githubusercontent.com/openai/whisper/86098128c0b4f24f0e2aa2994de830614b474227/tests/jfk.flac
$ shasum -a 1 jfk.flac
537ddf75dff344f8d31c1cd6ebee23ee8362d921  jfk.flac            (1152693 bytes)
```

The same JFK hash came from three separate downloads (the pinned URL twice, and `main`). One earlier download,
piped straight into `shasum` with a 60 s timeout, gave `e69fe392...`: a truncated transfer, which is the failure
a pinned hash turns into an error instead of silently bad training data.

Licenses: LJ Speech Dataset is public domain; the JFK clip is the 1961 inaugural address, a US government work
(public domain).
