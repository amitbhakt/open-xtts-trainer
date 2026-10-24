# Download pins for the public speech sources (`xtts_trainer/speech_data.py`)

Every downloaded file is checked against a SHA-1 before use; a file without a pin is refused. The GitHub raw
URLs are commit-pinned, so their content can't change. The archive.org URLs are not revision-specific (an item
can be edited), so for them the hash is the guarantee: changed content fails the check instead of silently
entering a dataset.

## LibriVox, The Art of War (archive.org)

The SHA-1 values are the ones archive.org publishes in the item metadata. Fetched 2026-10-25:

```
$ curl -sfL https://archive.org/metadata/art_of_war_librivox | python3 -c '
import json, sys
d = json.load(sys.stdin)
print("licenseurl", d["metadata"].get("licenseurl"))
for f in sorted((f for f in d["files"] if f.get("format") == "64Kbps MP3"), key=lambda f: f["name"]):
    print(f["sha1"], f["name"], f["size"], f["length"])'
licenseurl http://creativecommons.org/licenses/publicdomain/
9b9e7c673fe27d50c703f92efcabeb695d9722f7 art_of_war_01-02_sun_tzu_64kb.mp3 4055167 08:26
bfda68aff1a6f6a7a55f62cf11ee3ee652b3f699 art_of_war_03-04_sun_tzu_64kb.mp3 3723859 07:45
9bede3f68665ca5a7cdd0140d599ee529cb879bb art_of_war_05-06_sun_tzu_64kb.mp3 5094335 10:36
787463abd4ef57d25c64547708c7c5aa3514c6a0 art_of_war_07-08_sun_tzu_64kb.mp3 4246508 08:50
2b4e476b0f39b6ab055263d64bad8846d3b3c2be art_of_war_09-10_sun_tzu_64kb.mp3 7059580 14:42
ab6d413e9c793381b0f3e539bc65385f22c91fec art_of_war_11_sun_tzu_64kb.mp3 6213090 12:56
e59b738e5399332bed637039838059d38adc0137 art_of_war_12-13_sun_tzu_64kb.mp3 4312968 08:59
```

These equal `ART_OF_WAR` in `xtts_trainer/speech_data.py` (checked by diffing that list against this output).
Single reader: Moira Fogarty, per the item description. The first file was downloaded and verified: 8.45 min,
mono, 22050 Hz, SHA-1 matches.

## LJSpeech sample and JFK (GitHub raw)

These were downloaded from moving branches (`coqui-ai/TTS@dev`, `openai/whisper@main`). They are now pinned to
commits, and their SHA-1 values were computed from downloads of those exact URLs:

```
$ gh api repos/coqui-ai/TTS/commits/dev --jq '.sha'
dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e
$ gh api repos/openai/whisper/commits/main --jq '.sha'
86098128c0b4f24f0e2aa2994de830614b474227

$ LJ=https://raw.githubusercontent.com/coqui-ai/TTS/dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e/tests/data/ljspeech
$ for f in metadata.csv wavs/LJ001-000{1..6}.wav; do curl -sfL -o "$(basename $f)" "$LJ/$f"; done
$ shasum -a 1 metadata.csv LJ001-000*.wav
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
