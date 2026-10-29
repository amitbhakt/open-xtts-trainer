# License of the LibriVox held-out recordings

LibriVox recordings are released into the public domain (https://librivox.org/pages/public-domain/). For each
item used here, the license is what archive.org publishes in the item's metadata. Fetched 2026-10-27:

```
$ for item in fables_lafontaine_01_librivox grimm_maerchen_1_librivox; do
    curl -sSfL "https://archive.org/metadata/$item" | python3 -c 'import json, sys; m = json.load(sys.stdin)["metadata"]; print(m.get("licenseurl"), "|", m.get("title"))'; done
http://creativecommons.org/licenses/publicdomain/ | Fables de La Fontaine, livre 01
http://creativecommons.org/licenses/publicdomain/ | Märchen
```

| Item | File used | licenseurl |
|---|---|---|
| `fables_lafontaine_01_librivox` | `fables_01_00_lafontaine_64kb.mp3` | `http://creativecommons.org/licenses/publicdomain/` |
| `grimm_maerchen_1_librivox` | `grimm_214_spindel_weberschiffchen_und_nadel_64kb.mp3` | `http://creativecommons.org/licenses/publicdomain/` |

Checked again 2026-10-29, on macOS (`md5 -q` and `shasum` as shipped there; on Linux use `md5sum "$2" | cut -d' ' -f1`, `sha256sum "$2"` and `sha1sum "$2"`): each file is listed in its item's `files` metadata, and the downloaded bytes match the
SHA-1, MD5 and size archive.org publishes there, and the SHA-256 pinned in `../source_pins.json`:

```
$ check() {  # item, file
    curl -sSfL "https://archive.org/metadata/$1" -o "$1.json"
    python3 -I -c 'import json, sys; m = json.load(open(sys.argv[1])); f = [x for x in m["files"] if x["name"] == sys.argv[2]]; print(sys.argv[2], "| licenseurl", m["metadata"].get("licenseurl"), "| listed", bool(f), "| sha1", f and f[0].get("sha1"), "| md5", f and f[0].get("md5"), "| size", f and f[0].get("size"))' "$1.json" "$2"
    curl -sSfL "https://archive.org/download/$1/$2" -o "$2"; shasum -a 256 "$2"; shasum -a 1 "$2"; md5 -q "$2"; wc -c < "$2"; }
$ check fables_lafontaine_01_librivox fables_01_00_lafontaine_64kb.mp3
fables_01_00_lafontaine_64kb.mp3 | licenseurl http://creativecommons.org/licenses/publicdomain/ | listed True | sha1 bec7a9eddb99290e963a494e88bf3c43f7e9c330 | md5 64e6208300f93c72b7e739058152a284 | size 582095
be04604cf7b37bcf47ef3026392cebe5b2f38d9564b394bb9bed36781d039b0b  fables_01_00_lafontaine_64kb.mp3
bec7a9eddb99290e963a494e88bf3c43f7e9c330  fables_01_00_lafontaine_64kb.mp3
64e6208300f93c72b7e739058152a284
  582095
$ check grimm_maerchen_1_librivox grimm_214_spindel_weberschiffchen_und_nadel_64kb.mp3
grimm_214_spindel_weberschiffchen_und_nadel_64kb.mp3 | licenseurl http://creativecommons.org/licenses/publicdomain/ | listed True | sha1 bfff29f0c90ab89e8d111ecd9927e4717b798747 | md5 113717964b423ce3237a10cffef5a0f1 | size 3502545
e7e4ab3352316bfc0e3409ad9dbe8534e938497e99cba0613fb2f3fe3ae75c85  grimm_214_spindel_weberschiffchen_und_nadel_64kb.mp3
bfff29f0c90ab89e8d111ecd9927e4717b798747  grimm_214_spindel_weberschiffchen_und_nadel_64kb.mp3
113717964b423ce3237a10cffef5a0f1
 3502545
```

The SHA-256 lines equal the pins in `../source_pins.json`, which run 1 recorded from its first download. So for these
two files the pins are confirmed independently of that download.
