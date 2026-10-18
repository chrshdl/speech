"""
A person's review of the training clips whose transcript Whisper does not
hear, as examples/commonvoice/whisper_check.py finds them: for each, they
listen and decide whether Whisper misheard it, OK, and the clip is kept, or
Whisper was right that the transcript does not match, NOK, and the clip is
left out of training. train.py, with a "review" in its config's data,
pauses after each epoch for it, and leaves the NOK clips out of the next.

The files of a dataset <stem>.json, next to it:

    <stem>-whisper.jsonl  Whisper's transcript of each clip and its character
                          error rate against the clip's, from whisper_check.py
    <stem>-review.json    the verdicts so far, {audio path: "ok" or "nok"}

A clip is to review when Whisper's CER is over MIN_CER, which lets
spellings differ in a long sentence but not a single word.
"""

import http.server
import json
import os
import threading
import urllib.parse

MIN_CER = 0.15
VERDICTS = ("ok", "nok")


def files(dataset):
    """The Whisper results and the verdicts of a dataset json."""
    stem = os.path.splitext(dataset)[0]
    return f"{stem}-whisper.jsonl", f"{stem}-review.json"


def load_verdicts(dataset):
    path = files(dataset)[1]
    if not os.path.exists(path):
        return {}
    with open(path) as fid:
        return json.load(fid)


def save_verdicts(dataset, verdicts):
    path = files(dataset)[1]
    with open(path + ".tmp", "w") as fid:
        json.dump(verdicts, fid, indent=0, sort_keys=True)
    os.replace(path + ".tmp", path)


def queue(datasets, min_cer=MIN_CER):
    """
    The clips to review of the datasets with Whisper results, each with
    its dataset and verdict, the highest CER of each dataset first.
    """
    items = []
    for dataset in datasets:
        whisper = files(dataset)[0]
        if not os.path.exists(whisper):
            continue
        verdicts = load_verdicts(dataset)
        with open(whisper) as fid:
            for line in fid:
                row = json.loads(line)
                if row["CER"] > min_cer:
                    row["dataset"] = dataset
                    row["verdict"] = verdicts.get(row["audio"])
                    items.append(row)
    items.sort(key=lambda r: (datasets.index(r["dataset"]), -r["CER"]))
    return items


def excluded(datasets):
    """The audio paths of the clips reviewed as NOK, to leave out of training."""
    return {
        audio
        for dataset in datasets
        for audio, verdict in load_verdicts(dataset).items()
        if verdict == "nok"
    }


def serve(datasets, port=8765, min_cer=MIN_CER, title="Review"):
    """
    Serves the review of the datasets' clips on 127.0.0.1:port until the
    page's Continue is pressed, saving each verdict as it is given.
    Returns how many clips are reviewed NOK.
    """
    items = queue(datasets, min_cer)
    lock = threading.Lock()
    done = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, body, content_type="application/json", status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urllib.parse.urlparse(self.path).path
            if path == "/":
                self.reply(
                    PAGE.replace("{title}", title).encode(), "text/html; charset=utf-8"
                )
            elif path == "/api/items":
                with lock:
                    rows = [
                        {
                            "text": r["text"],
                            "whisper": r["whisper"],
                            "cer": r["CER"],
                            "duration": r["duration"],
                            "dataset": r["dataset"],
                            "verdict": r["verdict"],
                        }
                        for r in items
                    ]
                self.reply(json.dumps(rows).encode())
            elif (
                path.startswith("/audio/")
                and path[7:].isdigit()
                and int(path[7:]) < len(items)
            ):
                # Only the clips of the queue are served.
                audio = items[int(path[7:])]["audio"]
                kind = "audio/flac" if audio.endswith(".flac") else "audio/wav"
                with open(audio, "rb") as fid:
                    self.reply(fid.read(), kind)
            else:
                self.reply(b"Not found", "text/plain", 404)

        def do_POST(self):
            path = urllib.parse.urlparse(self.path).path
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            if path == "/api/verdict":
                i, verdict = body.get("i"), body.get("verdict")
                if (
                    not isinstance(i, int)
                    or not 0 <= i < len(items)
                    or verdict not in (*VERDICTS, None)
                ):
                    self.reply(b'{"error": "bad verdict"}', status=400)
                    return
                with lock:
                    item = items[i]
                    item["verdict"] = verdict
                    verdicts = load_verdicts(item["dataset"])
                    if verdict is None:
                        verdicts.pop(item["audio"], None)
                    else:
                        verdicts[item["audio"]] = verdict
                    save_verdicts(item["dataset"], verdicts)
                self.reply(b"{}")
            elif path == "/api/continue":
                self.reply(b"{}")
                done.set()
            else:
                self.reply(b"Not found", "text/plain", 404)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    reviewed = sum(r["verdict"] is not None for r in items)
    print(
        f"Review {len(items) - reviewed} new of {len(items)} clips at "
        f"http://127.0.0.1:{server.server_address[1]}/ , then press Continue there.",
        flush=True,
    )
    try:
        done.wait()
    finally:
        server.shutdown()
        server.server_close()
    nok = sum(r["verdict"] == "nok" for r in items)
    print(
        f"Reviewed: {sum(r['verdict'] is not None for r in items)} of {len(items)}, {nok} NOK."
    )
    return nok


PAGE = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root { --bg: #f6f5f9; --card: #fff; --ink: #24203a; --muted: #6b6680;
    --ok: #1f9d55; --nok: #d64545; --accent: #6c5ce7; --line: #e4e1ee; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #17151f; --card: #221f2e; --ink: #ece9f6; --muted: #a29db6;
      --line: #34304a; }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
    font: 16px/1.45 system-ui, -apple-system, sans-serif; }
  main { max-width: 760px; margin: 0 auto; padding: 28px 16px 48px; }
  header { display: flex; align-items: center; justify-content: space-between;
    gap: 12px; flex-wrap: wrap; margin-bottom: 18px; }
  h1 { font-size: 20px; margin: 0; }
  .progress { flex: 1 1 100%; height: 6px; background: var(--line);
    border-radius: 3px; overflow: hidden; }
  .progress div { height: 100%; background: var(--accent); width: 0; }
  .counts { color: var(--muted); font-size: 14px; }
  .card { background: var(--card); border: 1px solid var(--line);
    border-radius: 14px; padding: 22px; }
  .meta { color: var(--muted); font-size: 13px; display: flex; gap: 14px;
    flex-wrap: wrap; margin-bottom: 14px; }
  .label { font-size: 13px; color: var(--muted); text-transform: uppercase;
    letter-spacing: .04em; margin-top: 10px; }
  .text { font-size: 26px; font-weight: 600; word-break: break-word; }
  .heard { font-size: 22px; font-style: italic; word-break: break-word; }
  audio { width: 100%; margin: 18px 0 6px; }
  .buttons { display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
    margin-top: 14px; }
  button { font: inherit; border-radius: 10px; border: 2px solid var(--line);
    background: var(--card); color: var(--ink); padding: 12px 14px;
    cursor: pointer; }
  button.ok { border-color: var(--ok); }
  button.nok { border-color: var(--nok); }
  button.ok.on { background: var(--ok); color: #fff; }
  button.nok.on { background: var(--nok); color: #fff; }
  button small { display: block; color: var(--muted); font-size: 13px; }
  button.on small { color: rgba(255, 255, 255, .85); }
  .nav { display: flex; justify-content: space-between; gap: 12px;
    margin-top: 14px; flex-wrap: wrap; }
  .keys { color: var(--muted); font-size: 13px; margin-top: 18px; }
  kbd { border: 1px solid var(--line); border-radius: 4px; padding: 0 5px;
    font-size: 12px; }
  #continue { border-color: var(--accent); }
  #continue.done { background: var(--accent); color: #fff; }
  .empty { text-align: center; color: var(--muted); padding: 40px 0; }
</style>
<main>
  <header>
    <h1>{title}</h1>
    <span class="counts" id="counts"></span>
    <div class="progress"><div id="bar"></div></div>
  </header>
  <section class="card" id="card">
    <div class="meta"><span id="position"></span><span id="dataset"></span>
      <span id="cer"></span><span id="duration"></span></div>
    <div class="label">Transcript</div>
    <div class="text" id="text"></div>
    <div class="label">Whisper heard</div>
    <div class="heard" id="heard"></div>
    <audio id="audio" controls preload="auto"></audio>
    <div class="buttons">
      <button class="ok" id="ok">OK, keep <small>Whisper misheard it (O)</small></button>
      <button class="nok" id="nok">NOK, drop <small>Whisper is right (N)</small></button>
    </div>
    <div class="nav">
      <button id="prev">&larr; Previous</button>
      <button id="next-new">Next unreviewed</button>
      <button id="next">Next &rarr;</button>
    </div>
  </section>
  <div class="nav"><span></span><button id="continue">Continue training</button></div>
  <p class="keys"><kbd>O</kbd> OK &nbsp; <kbd>N</kbd> NOK &nbsp; <kbd>Space</kbd> play again
    &nbsp; <kbd>&larr;</kbd> <kbd>&rarr;</kbd> previous, next. A verdict is saved at once
    and moves on to the next clip.</p>
</main>
<script>
let items = [], at = 0;
const $ = (id) => document.getElementById(id);

async function load() {
  items = await (await fetch("/api/items")).json();
  if (!items.length) {
    $("card").innerHTML = '<p class="empty">Nothing to review.</p>';
    counts();
    return;
  }
  const first = items.findIndex((r) => !r.verdict);
  show(first < 0 ? 0 : first);
}

function counts() {
  const reviewed = items.filter((r) => r.verdict).length;
  const nok = items.filter((r) => r.verdict === "nok").length;
  $("counts").textContent = `${reviewed} of ${items.length} reviewed, ${nok} NOK`;
  $("bar").style.width = items.length ? `${(100 * reviewed) / items.length}%` : "100%";
  $("continue").classList.toggle("done", reviewed === items.length);
}

function show(i) {
  at = Math.max(0, Math.min(items.length - 1, i));
  const r = items[at];
  $("position").textContent = `${at + 1} / ${items.length}`;
  $("dataset").textContent = r.dataset;
  $("cer").textContent = `CER ${r.cer.toFixed(2)}`;
  $("duration").textContent = `${r.duration.toFixed(1)} s`;
  $("text").textContent = r.text;
  $("heard").textContent = r.whisper || "(nothing)";
  $("ok").classList.toggle("on", r.verdict === "ok");
  $("nok").classList.toggle("on", r.verdict === "nok");
  $("audio").src = `/audio/${at}`;
  $("audio").play().catch(() => {});
  counts();
}

async function decide(verdict) {
  const r = items[at];
  r.verdict = r.verdict === verdict ? null : verdict;
  await fetch("/api/verdict", { method: "POST", body: JSON.stringify({ i: at, verdict: r.verdict }) });
  if (r.verdict && at < items.length - 1) show(at + 1); else show(at);
}

function nextNew() {
  const after = items.findIndex((r, i) => i > at && !r.verdict);
  const any = items.findIndex((r) => !r.verdict);
  if (after >= 0 || any >= 0) show(after >= 0 ? after : any);
}

$("ok").onclick = () => decide("ok");
$("nok").onclick = () => decide("nok");
$("prev").onclick = () => show(at - 1);
$("next").onclick = () => show(at + 1);
$("next-new").onclick = nextNew;
$("continue").onclick = async () => {
  await fetch("/api/continue", { method: "POST", body: "{}" });
  document.querySelector("main").innerHTML =
    '<p class="empty">Saved. Training continues; this page can be closed.</p>';
};
document.addEventListener("keydown", (e) => {
  if (!items.length || e.metaKey || e.ctrlKey || e.altKey) return;
  const key = e.key.toLowerCase();
  if (key === "o") decide("ok");
  else if (key === "n") decide("nok");
  else if (key === " ") { e.preventDefault(); $("audio").currentTime = 0; $("audio").play(); }
  else if (key === "arrowleft") show(at - 1);
  else if (key === "arrowright") show(at + 1);
  else return;
  e.preventDefault();
});
load();
</script>
</html>
"""
