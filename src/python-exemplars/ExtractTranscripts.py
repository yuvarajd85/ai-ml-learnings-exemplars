"""
Bulk YouTube transcript extractor for the "Machine Learning Mathematics" series.

Run this LOCALLY (not in Claude's sandbox — it has no network path to youtube.com).

Setup:
    pip install youtube-transcript-api

Usage:
    python extract_transcripts.py

Output:
    One .txt file per video in ./transcripts/, named <video_id>.txt
    A manifest.json mapping video_id -> title (fetched via oEmbed, no API key needed)
    Any failures (no captions available, etc.) are logged to failures.txt instead of
    silently skipped — check that file before assuming you got everything.
"""

import json
import os
import ssl
import time
import urllib.request

import certifi
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

VIDEO_IDS = [
    "wVUdZ6N7_fc", "CiBL4zEE1Lo", "AmMritqNAiA", "q8unKzqY8No", "Bmljc5h3itc",
    "VgSEoPXlQds", "SjNait8gv6Y", "MBPRCM1xlPs", "eXbVgB7UQro", "lvp2tkyiqEQ",
    "_PaN-yl7y94", "dZpe3qgWFW4", "on9kiAUV3jk", "HGMcpWkxqcM", "1nKYYIOi4rQ",
    "0Dwb7y82FMc", "4_WM_zJeJt4", "HPNc7OsM_oI", "5D4sAWxL9fo", "_ljthhcBpVA",
    "ayFQxzsU6EE", "TWtQUb5Z-HU", "-g9PNutZ-J8", "lc_NDUwDOnM", "tl535M5-kgc",
    "Mx-6vR5OCPo", "J83SusYlB10", "TVEnJGo4pjY", "bAJEs18BOfY", "W6QFRP29HUU",
    "lx1uhgxh-C0", "ng_2W-jWUqk", "ueD4J3XNcGg", "BSpUP78YV1A", "ZLmF902sfpo",
    "AJS_EUe-Fx0", "LMP1hwaxNvI", "JCQO7MSbgec", "hQ1sgJTMcuA", "RQnVoeyq2Jg",
    "h8E1foWdim4", "4_8fdutjlYs", "uHs_kzy_Byg",
]
# de-duplicate while preserving order
VIDEO_IDS = list(dict.fromkeys(VIDEO_IDS))

OUT_DIR = "transcripts"
os.makedirs(OUT_DIR, exist_ok=True)

manifest = {}
failures = []
ytt_api = YouTubeTranscriptApi()

for i, vid in enumerate(VIDEO_IDS, 1):
    print(f"[{i}/{len(VIDEO_IDS)}] {vid} ...", end=" ")

    # Get title via oEmbed (no API key required)
    title = vid
    try:
        with urllib.request.urlopen(
                f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={vid}&format=json",
                timeout=10,
                context=_SSL_CTX,
        ) as resp:
            title = json.loads(resp.read())["title"]
    except Exception as e:
        print(f"(title lookup failed: {e})", end=" ")

    manifest[vid] = title

    try:
        transcript = ytt_api.fetch(vid)
        text = "\n".join(seg.text for seg in transcript)
        with open(os.path.join(OUT_DIR, f"{vid}.txt"), "w", encoding="utf-8") as f:
            f.write(f"# {title}\n# https://youtu.be/{vid}\n\n{text}")
        print("OK")
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        failures.append((vid, title, str(e)))
        print(f"FAILED: {e}")
    except Exception as e:
        failures.append((vid, title, str(e)))
        print(f"FAILED: {e}")

    time.sleep(1)  # be polite, avoid rate limiting

with open("manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

if failures:
    with open("failures.txt", "w", encoding="utf-8") as f:
        for vid, title, err in failures:
            f.write(f"{vid}\t{title}\t{err}\n")
    print(f"\n{len(failures)} video(s) failed — see failures.txt")
else:
    print("\nAll transcripts extracted successfully.")

print(f"Done. {len(VIDEO_IDS) - len(failures)}/{len(VIDEO_IDS)} transcripts written to ./{OUT_DIR}/")