# Download pins for the public speech sources (`xtts_trainer/speech_data.py`)

Every downloaded file is checked against a SHA-256 before use. A file without a pin is refused, and so is a pin
that isn't 64 hex digits. The GitHub raw URLs are commit-pinned, so their content can't change. The archive.org
URLs are not revision-specific (an item can be edited), so for them the hash is the guarantee: changed content
fails the check instead of silently entering a dataset.

Why SHA-256: the pins were first SHA-1, because archive.org publishes SHA-1 and MD5 but not SHA-256. SHA-1 is no
longer considered collision-resistant, so the pins were moved to SHA-256 (2026-10-25). Each SHA-256 was taken
from a file whose SHA-1 first matched the earlier pin: archive.org's published value, or the value computed at
pinning for the GitHub files. That carries the provenance over, and the check itself no longer relies on SHA-1.

## LibriVox, The Art of War (archive.org)

The SHA-1 values archive.org publishes in the item metadata, fetched 2026-10-25:

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

Single reader: Moira Fogarty, per the item description.

## LJSpeech sample and JFK (GitHub raw)

These were downloaded from moving branches (`coqui-ai/TTS@dev`, `openai/whisper@main`) until they were pinned
to the commits those branches pointed at:

```
$ gh api repos/coqui-ai/TTS/commits/dev --jq '.sha'
dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e
$ gh api repos/openai/whisper/commits/main --jq '.sha'
86098128c0b4f24f0e2aa2994de830614b474227
```

Their SHA-1 values were computed at pinning, from downloads of those exact URLs (sizes in bytes):

```
247ef0626c1ec7f4d01e8bc8599c4b070a396209  metadata.csv        (1655)
a4f9d63b0cfbd36eb1c23b7b122f213bc587ecc4  LJ001-0001.wav      (425830)
6249e7e3edd982e8cb79a8b8deef4c907374e1fc  LJ001-0002.wav      (83814)
fc4d535b7b0a1ba2ea09dca3aad2fc9e2816a19c  LJ001-0003.wav      (426342)
41a5fe1e72f423534158d54cc9bfca920921c045  LJ001-0004.wav      (226662)
ccef55335f89ed4f21c4e0091a1aaf34b2bcbe11  LJ001-0005.wav      (357734)
6fafc6a2bb5dedf373cdace60bd6746a89548153  LJ001-0006.wav      (250726)
537ddf75dff344f8d31c1cd6ebee23ee8362d921  jfk.flac            (1152693)
```

The same JFK hash came from three separate downloads (the pinned URL twice, and `main`). One earlier download,
piped straight into `shasum` with a 60 s timeout, gave `e69fe392...`: a truncated transfer, which is the failure
a pinned hash turns into an error instead of silently bad training data.

## SHA-256 pins (2026-10-25)

All 15 files downloaded again from the pinned URLs. Every SHA-1 equals the value above, and the SHA-256 values
are the pins in `speech_data.py`. A separate earlier download of all 15 gave the same SHA-256 values.

```
$ LJ=https://raw.githubusercontent.com/coqui-ai/TTS/dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e/tests/data/ljspeech
$ mkdir -p ljspeech jfk art_of_war
$ (cd ljspeech && for f in metadata.csv wavs/LJ001-000{1..6}.wav; do curl -sfL -o "$(basename $f)" "$LJ/$f"; done)
$ (cd jfk && curl -sfL -o jfk.flac https://raw.githubusercontent.com/openai/whisper/86098128c0b4f24f0e2aa2994de830614b474227/tests/jfk.flac)
$ (cd art_of_war && for n in 01-02 03-04 05-06 07-08 09-10 11 12-13; do
     curl -sSfL --retry 3 -O "https://archive.org/download/art_of_war_librivox/art_of_war_${n}_sun_tzu_64kb.mp3"; done)
$ shasum -a 1 ljspeech/* jfk/* art_of_war/*
a4f9d63b0cfbd36eb1c23b7b122f213bc587ecc4  ljspeech/LJ001-0001.wav
6249e7e3edd982e8cb79a8b8deef4c907374e1fc  ljspeech/LJ001-0002.wav
fc4d535b7b0a1ba2ea09dca3aad2fc9e2816a19c  ljspeech/LJ001-0003.wav
41a5fe1e72f423534158d54cc9bfca920921c045  ljspeech/LJ001-0004.wav
ccef55335f89ed4f21c4e0091a1aaf34b2bcbe11  ljspeech/LJ001-0005.wav
6fafc6a2bb5dedf373cdace60bd6746a89548153  ljspeech/LJ001-0006.wav
247ef0626c1ec7f4d01e8bc8599c4b070a396209  ljspeech/metadata.csv
537ddf75dff344f8d31c1cd6ebee23ee8362d921  jfk/jfk.flac
9b9e7c673fe27d50c703f92efcabeb695d9722f7  art_of_war/art_of_war_01-02_sun_tzu_64kb.mp3
bfda68aff1a6f6a7a55f62cf11ee3ee652b3f699  art_of_war/art_of_war_03-04_sun_tzu_64kb.mp3
9bede3f68665ca5a7cdd0140d599ee529cb879bb  art_of_war/art_of_war_05-06_sun_tzu_64kb.mp3
787463abd4ef57d25c64547708c7c5aa3514c6a0  art_of_war/art_of_war_07-08_sun_tzu_64kb.mp3
2b4e476b0f39b6ab055263d64bad8846d3b3c2be  art_of_war/art_of_war_09-10_sun_tzu_64kb.mp3
ab6d413e9c793381b0f3e539bc65385f22c91fec  art_of_war/art_of_war_11_sun_tzu_64kb.mp3
e59b738e5399332bed637039838059d38adc0137  art_of_war/art_of_war_12-13_sun_tzu_64kb.mp3
$ shasum -a 256 ljspeech/* jfk/* art_of_war/*
4d2bd83ff0f7fe33491b03193abdf623f1b4cc2d8103972b33638d3607d86dc5  ljspeech/LJ001-0001.wav
6563390fa42121eeeab15f49fa91fd26afe000022bfdaaa882f06224ad549599  ljspeech/LJ001-0002.wav
ad97822fa512b4897a01557423ffff7cc1fc763c9b9d5d502a5a1bc566f7440b  ljspeech/LJ001-0003.wav
3c9cce186f26b6690b1d96a6a640c3f1bdb10e1f206738344190fa264955a7af  ljspeech/LJ001-0004.wav
0d95df8cd780f0ed4f71d74559d42b33cc98ae9e7b398e0eceb17428dfacb9ca  ljspeech/LJ001-0005.wav
4b866eebe1e7cb6ded4bcb27caa0638b8ade67a6a8a2f7132952b716dc7006db  ljspeech/LJ001-0006.wav
2727a6428189a0213d480b2de5eceb159877c2b99b0027b22115d076c868938d  ljspeech/metadata.csv
63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715  jfk/jfk.flac
6cb289fa9a13cfc94b18e8a89159da687c60db88c2b6377311f9bc5664d5492f  art_of_war/art_of_war_01-02_sun_tzu_64kb.mp3
57e879fe7822578fd22010f1280185799a119a3482e025d9a71f49878ff45ba7  art_of_war/art_of_war_03-04_sun_tzu_64kb.mp3
ace85bbe5e784c9dc622b9ddb346b513afa7cdb4f9176c45d2fd9df16794bb58  art_of_war/art_of_war_05-06_sun_tzu_64kb.mp3
23bf3869fc869942cd68812417927fc350dee33e114bbe445f41d5157d559aad  art_of_war/art_of_war_07-08_sun_tzu_64kb.mp3
49dcdde816c1e1152c1ff1721171ebe5817d0db5ffd2c8970c2c20511b4043ad  art_of_war/art_of_war_09-10_sun_tzu_64kb.mp3
031b2426f8a67f7cda7e3a38ed13b061876f7610e078fb8395962669803c27d8  art_of_war/art_of_war_11_sun_tzu_64kb.mp3
61acc0e274eef3f740ec4ad4c5c4a18907e35832d0581d1bf9dfdc3b031afafa  art_of_war/art_of_war_12-13_sun_tzu_64kb.mp3
```

The first loop dropped two archive.org files without an error (plain `curl -sfL` inside a `for` loop); they
were fetched again with `-sS --retry 3`, which is the command shown.

Licenses: LJ Speech Dataset is public domain; the JFK clip is the 1961 inaugural address, a US government work
(public domain); LibriVox recordings are public-domain dedications.
