import glob, os, re, subprocess
import numpy as np
import yt_dlp

LIVE_SECONDS = 600      # live streams: record the first 10 minutes
MAX_MINUTES = 180       # refuse videos longer than this
HYPE = ["omg", "wow", "no way", "insane", "crazy", "let's go", "lets go",
        "holy", "bro", "yooo", "huge", "sick", "oh my god", "what the"]


def download(url, folder):
    base = {"quiet": True, "noplaylist": True}
    with yt_dlp.YoutubeDL(base) as y:
        info = y.extract_info(url, download=False)
    if not info.get("is_live") and (info.get("duration") or 0) > MAX_MINUTES * 60:
        raise ValueError("Video is longer than %d minutes." % MAX_MINUTES)
    opts = dict(base, outtmpl=os.path.join(folder, "source.%(ext)s"),
                format="bestvideo[height<=720]+bestaudio/best[height<=720]/best",
                merge_output_format="mp4")
    if info.get("is_live"):
        opts.update(downloader="ffmpeg", downloader_args={"ffmpeg_o": ["-t", str(LIVE_SECONDS)]})
    with yt_dlp.YoutubeDL(opts) as y:
        y.download([url])
    files = sorted(glob.glob(os.path.join(folder, "source.*")))
    if not files:
        raise RuntimeError("Download failed.")
    return files[0]


def energy(path):
    """Loudness (RMS) for every second of audio."""
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    n = len(a) // 16000
    if n < 5:
        raise ValueError("Video has no usable audio.")
    return np.sqrt((a[:n * 16000].reshape(n, 16000) ** 2).mean(axis=1))


def transcribe(path):
    """Optional captions + hype words. Returns [] if faster-whisper is missing."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return []
    model = WhisperModel("base", compute_type="int8")
    segs, _ = model.transcribe(path, vad_filter=True)
    return [(s.start, s.end, s.text.strip()) for s in segs]


def pick(en, segs, length, count):
    n = len(en)
    z = (en - en.mean()) / (en.std() + 1e-6)
    for s, e, t in segs:
        if any(w in t.lower() for w in HYPE):
            z[int(s):min(n, int(e) + 1)] += 1.0
    sm = np.convolve(z, np.ones(8) / 8, mode="same")
    length = min(length, n)
    chosen = []
    for t in np.argsort(sm)[::-1]:
        start = int(max(0, min(t - length * 0.5, n - length)))
        end = start + length
        if any(start < e2 and end > s2 for s2, e2, _ in chosen):
            continue
        chosen.append((start, end, float(sm[t])))
        if len(chosen) == count:
            break
    top = max(c[2] for c in chosen)
    low = min(float(sm.min()), 0)
    return sorted([(s, e, round(10 * (v - low) / (top - low + 1e-6), 1)) for s, e, v in chosen])


def stamp(x):
    h, m, s, ms = int(x // 3600), int(x % 3600 // 60), int(x % 60), int(x % 1 * 1000)
    return "%02d:%02d:%02d,%03d" % (h, m, s, ms)


def cut(src, start, end, out, segs):
    src, out = os.path.abspath(src), os.path.abspath(out)
    folder = os.path.dirname(out)
    srt = os.path.join(folder, os.path.basename(out) + ".srt")
    lines, k = [], 1
    for s, e, t in segs:
        if e > start and s < end and t:
            lines.append("%d\n%s --> %s\n%s\n" % (k, stamp(max(s, start) - start), stamp(min(e, end) - start), t))
            k += 1
    vf = "crop='min(iw,ih*9/16)':ih,scale=1080:1920"
    if lines:
        open(srt, "w", encoding="utf-8").write("\n".join(lines))
        vf += ",subtitles=%s:force_style='FontSize=14,Alignment=2,MarginV=90,Outline=2'" % os.path.basename(srt)
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-i", src, "-t", str(end - start),
           "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-c:a", "aac", "-movflags", "+faststart", os.path.basename(out)]
    subprocess.run(cmd, check=True, cwd=folder)


def make_clips(url, folder, length, count, step):
    step("Fetching the video", 8)
    src = download(url, folder)
    step("Analyzing the audio", 35)
    en = energy(src)
    step("Transcribing (for captions)", 50)
    segs = transcribe(src)
    step("Finding the best moments", 65)
    moments = pick(en, segs, length, count)
    clips = []
    for i, (s, e, sc) in enumerate(moments, 1):
        step("Cutting clip %d of %d" % (i, len(moments)), 65 + int(30 * i / len(moments)))
        name = "clip%d.mp4" % i
        cut(src, s, e, os.path.join(folder, name), segs)
        clips.append({"file": "/output/%s/%s" % (os.path.basename(folder), name),
                      "start": s, "end": e, "score": sc})
    os.remove(src)
    return clips
