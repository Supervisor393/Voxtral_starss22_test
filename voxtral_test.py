import os
import random
import io
import json
import re
import soundfile as sf
import pandas as pd
import requests
from typing import List, Dict, Any, Optional, Tuple
from mistral_common.protocol.instruct.messages import TextChunk, AudioChunk, UserMessage
from mistral_common.audio import Audio

# ========== CONFIG ==========
VLLM_ENDPOINT = "http://127.0.0.1:8003/v1/chat/completions"
WIN_LEN = 120.0      # seconds
TAU_EVENTS_CSV  = "/data/user/jzt/crd/audioLLM/train_events/tau.csv"
SONY_EVENTS_CSV = "/data/user/jzt/crd/audioLLM/train_events/sony.csv"

# model gen params
TEMPERATURE = 0.2
TOP_P = 0.95
MAX_TOKENS = 4096

# ---------------- Robust JSON -> list[{"class":int, "start":float}] ----------------

def extract_json_array(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if m:
        candidate = m.group(1)
        arr = _find_top_level_array(candidate)
        if arr is not None:
            return arr
    arr = _find_top_level_array(text)
    return arr

def _find_top_level_array(s: str) -> Optional[str]:
    start = s.find('[')
    while start != -1:
        i, depth, in_str, esc = start, 0, False, False
        while i < len(s):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == '[':
                    depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0:
                        return s[start:i+1]
            i += 1
        start = s.find('[', start + 1)
    return None

def parse_class_start_list(raw_reply: str) -> List[Dict[str, Any]]:
    """Parse [{"class":ID,"start":seconds}, ...] (start is RELATIVE to the window)."""
    arr_text = extract_json_array(raw_reply)
    if not arr_text:
        return []
    try:
        data = json.loads(arr_text)
    except Exception:
        return []
    out = []
    if isinstance(data, list):
        for e in data:
            if not isinstance(e, dict):
                continue
            c = e.get("class")
            st = e.get("start")
            try:
                c_int = int(c)
                st_f = float(st)
            except Exception:
                continue
            if c_int < 0 or c_int > 12:
                continue
            if not (0.0 <= st_f < WIN_LEN + 1e-6):
                continue
            out.append({"class": c_int, "start": st_f})
    return out

# ---------------- audio <-> AudioChunk (via temp file path) ----------------

def array_to_audiochunk(arr, sr) -> AudioChunk:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        sf.write(tmp_path, arr, sr, format="wav")
        audio = Audio.from_file(tmp_path, strict=False)
        return AudioChunk.from_audio(audio)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

def call_model_on_window(audio_arr, sr, present_classes: List[int]) -> str:
    """Prompt dynamically with present_classes and return raw content string."""
    cls_list_str = "[" + ", ".join(str(x) for x in sorted(set(present_classes))) + "]"
    prompt = (
        f"You will be given a {WIN_LEN:.0f}-second audio segment.\n"
        f"The following target class IDs are present in THIS segment: {cls_list_str}.\n"
        "Task: For EACH listed class ID, report the EARLIEST start time within THIS segment.\n"
        "Return ONLY a strict JSON array with EXACTLY one object per listed class, sorted by class.\n"
        "{\"class\": ID, \"start\": seconds} for each object.\n"
        f"Constraints: 0 <= start < {WIN_LEN:.1f}. Do NOT include any class not in the list. No extra text."
    )

    audio_chunk = array_to_audiochunk(audio_arr, sr)
    text_chunk = TextChunk(text=prompt)
    user_msg = UserMessage(content=[audio_chunk, text_chunk]).to_openai()

    payload = {
        "model": "voxtral-mini-3b",
        "messages": [user_msg],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
    }
    r = requests.post(VLLM_ENDPOINT, json=payload, timeout=300)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
    return r.json()["choices"][0]["message"]["content"]

# ---------------- windowing and GT computation ----------------

def split_fixed_10s(wav_path: str):
    """
    Return (wav, sr, windows)
    windows = list of (i0, i1, t0, t1) with exact ts each; last tail <ts is dropped.
    """
    wav, sr = sf.read(wav_path)
    n = len(wav)
    samples = int(WIN_LEN * sr)
    wins = []
    i = 0
    while i + samples <= n:
        i0, i1 = i, i + samples
        t0, t1 = i0 / sr, i1 / sr
        wins.append((i0, i1, t0, t1))
        i += samples
    return wav, sr, wins

def load_events_csv(path: str) -> pd.DataFrame:
    """
    Expect columns: file,onset,offset,class
    'file' should match basename of wav file.
    """
    df = pd.read_csv(path)
    need = {"file", "onset", "offset", "class"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Events CSV missing columns: {missing} in {path}")
    return df

def is_sony_path(p: str) -> bool:
    return "sony" in p.lower()

def is_tau_path(p: str) -> bool:
    return "tau" in p.lower()

def earliest_class_abs_time_in_window(df_events: pd.DataFrame, basename: str, t0: float, t1: float) -> Dict[int, float]:
    """
    对窗口 [t0, t1)：
      - 过滤出与窗口重叠的事件: offset > t0 && onset < t1
      - 对每个类别取窗口内最早出现的“绝对时间”： earliest_abs = min( max(onset, t0) )
    返回 {class_id: earliest_abs_time}（注意是绝对时间，不减 t0）
    """
    df = df_events[df_events["file"] == basename]
    if df.empty:
        return {}

    mask = (df["offset"] > t0) & (df["onset"] < t1)
    sub = df.loc[mask, ["class", "onset", "offset"]]
    if sub.empty:
        return {}

    out: Dict[int, float] = {}
    for cls_id, grp in sub.groupby("class"):
        try:
            cid = int(cls_id)
        except Exception:
            continue
        # 窗口内最早“绝对时间”：事件若跨入窗口，则为 t0
        earliest_abs = min(max(float(row.onset), t0) for _, row in grp.iterrows())
        out[cid] = float(earliest_abs)
    return out

# ---------------- append writers (no dedupe) ----------------

def append_rows(csv_path: str, rows: List[Dict[str, Any]]):
    """
    追加写：file, win_start, win_end, class, gt_start, pred_start
    此处的 gt_start / pred_start 都是“绝对时间”（相对整条音频）。
    """
    if not rows:
        return
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    cols = ["file", "win_start", "win_end", "class", "gt_start", "pred_start"]
    df = pd.DataFrame(rows)[cols]
    exists = os.path.exists(csv_path)
    df.to_csv(csv_path, mode=("a" if exists else "w"), header=not exists, index=False)
    print(f"[append] {len(df)} rows -> {csv_path}")

# ---------------- main per-file process ----------------

def process_file_windows(wav_path: str,
                         events_df: pd.DataFrame,
                         out_csv_path: str,
                         replies_jsonl_path: str):
    """
    对每个 ts 窗口：
      - 由 GT CSV 计算“每类在该窗口的最早绝对时间”（gt_start，绝对时间）
      - 提示模型仅返回这些类的“最早开始相对时间” -> 转为 pred_start = win_start + 相对时间（绝对时间）
      - 逐类写入一行：file, win_start, win_end, class, gt_start, pred_start （都绝对时间）
      - 原始模型回复以追加方式写入 replies_jsonl
    """
    wav, sr, wins = split_fixed_10s(wav_path)
    base = os.path.basename(wav_path)

    os.makedirs(os.path.dirname(replies_jsonl_path) or ".", exist_ok=True)
    append_buffer: List[Dict[str, Any]] = []

    with open(replies_jsonl_path, "a", encoding="utf-8") as fout:
        for (i0, i1, t0, t1) in wins:
            # 计算该窗口的 GT：字典 {class_id: earliest_abs_time}
            gt_abs_map = earliest_class_abs_time_in_window(events_df, base, t0, t1)
            if not gt_abs_map:
                continue

            clip = wav[i0:i1]
            present_classes = sorted(gt_abs_map.keys())
            # 如果类别大于4，随机选4个类别
            if len(present_classes) > 4:
                present_classes = random.sample(present_classes, 4)

            # 调模型
            raw = call_model_on_window(clip, sr, present_classes)

            # 保存原始回复（按窗口）
            fout.write(json.dumps({
                "file": base,
                "win_start": t0,
                "win_end": t1,
                "classes": present_classes,
                "reply": raw
            }, ensure_ascii=False) + "\n")

            # 解析模型相对开始时间
            pred_list = parse_class_start_list(raw)  # [{"class":ID,"start":rel}, ...]
            pred_rel_map = {int(d["class"]): float(d["start"]) for d in pred_list}

            # 汇总行：pred_start = t0 + rel_start   ;   gt_start 已是绝对时间
            for cid in present_classes:
                gt_abs = float(gt_abs_map[cid])             # 绝对时间
                pred_abs = t0 + pred_rel_map[cid] if cid in pred_rel_map else ""  # 若缺失则空
                append_buffer.append({
                    "file": base,
                    "win_start": t0,
                    "win_end": t1,
                    "class": int(cid),
                    "gt_start": gt_abs,     # 绝对时间
                    "pred_start": pred_abs  # 绝对时间（= 窗口起点 + 相对时间）
                })

    append_rows(out_csv_path, append_buffer)

# ---------------- entry ----------------

def is_sony_path(p: str) -> bool:
    return "sony" in p.lower()

def is_tau_path(p: str) -> bool:
    return "tau" in p.lower()

if __name__ == "__main__":
    files = [
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix001.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix002.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix003.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix004.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix005.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix006.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix007.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix008.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix009.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix010.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix011.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix012.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix013.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix014.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix015.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix016.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix017.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix018.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix019.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix020.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix021.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix022.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix023.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix024.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix025.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix026.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix027.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix028.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix029.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix001.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix002.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix003.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix004.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix005.wav",   
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix006.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix007.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix008.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix009.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix010.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix011.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room4_mix001.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room4_mix004.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room4_mix005.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room4_mix006.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room4_mix007.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room4_mix008.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room6_mix001.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room6_mix002.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room6_mix003.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room6_mix004.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room6_mix005.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room6_mix006.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room6_mix007.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room7_mix001.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room7_mix002.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room7_mix003.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room7_mix004.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room7_mix005.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room7_mix006.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room9_mix001.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room9_mix002.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room9_mix003.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room9_mix004.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room9_mix005.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room9_mix006.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room9_mix007.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room9_mix008.wav",
    ]

    df_tau  = load_events_csv(TAU_EVENTS_CSV)  if os.path.exists(TAU_EVENTS_CSV)  else None
    df_sony = load_events_csv(SONY_EVENTS_CSV) if os.path.exists(SONY_EVENTS_CSV) else None

    for wav_path in files:
        replies_jsonl = wav_path + ".win120.4limit.replies.jsonl"

        if is_tau_path(wav_path):
            if df_tau is None:
                raise FileNotFoundError(f"TAU events CSV not found: {TAU_EVENTS_CSV}")
            out_csv = "tau.win120_earliest.csv"
            process_file_windows(wav_path, df_tau, out_csv, replies_jsonl)

        elif is_sony_path(wav_path):
            if df_sony is None:
                raise FileNotFoundError(f"SONY events CSV not found: {SONY_EVENTS_CSV}")
            out_csv = "sony.win120_earliest.csv"
            process_file_windows(wav_path, df_sony, out_csv, replies_jsonl)

        else:
            print(f"[WARN] Unknown domain (neither sony nor tau): {wav_path}")
