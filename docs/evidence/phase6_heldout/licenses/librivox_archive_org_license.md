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
| `grimm_maerchen_1_librivox` | `grimm_114_erbsenprobe_64kb.mp3` | `http://creativecommons.org/licenses/publicdomain/` |

Both files are listed in their items' metadata. The audio itself is pinned by SHA-256 in `../source_pins.json`
after the first run.
