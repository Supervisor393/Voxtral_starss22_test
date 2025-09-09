import os
import random
import json
import re
import soundfile as sf
import pandas as pd
import requests
from typing import List, Dict, Any, Optional, Tuple
from mistral_common.protocol.instruct.messages import TextChunk, AudioChunk, UserMessage
from mistral_common.audio import Audio

# ========== CONFIG ==========
VLLM_ENDPOINT = "http://127.0.0.1:8011/v1/chat/completions"

# 一次跑多个窗口长度（秒）
WIN_LENS = [5, 10, 20, 30, 40, 50, 60, 90, 120]

TAU_EVENTS_CSV  = "/data/user/jzt/crd/audioLLM/train_events/tau.csv"
SONY_EVENTS_CSV = "/data/user/jzt/crd/audioLLM/train_events/sony.csv"

# model gen params
TEMPERATURE = 0.2
TOP_P = 0.95
MAX_TOKENS = 4096

# ====== CLASS ID → NAME MAPPING ======
CLASS_ID_TO_NAME = {
    0: "Female speech, woman speaking",
    1: "Male speech, man speaking",
    2: "Clapping",
    3: "Telephone",
    4: "Laughter",
    5: "Domestic sounds",
    6: "Walk, footsteps",
    7: "Door, open or close",
    8: "Music",
    9: "Musical instrument",
    10: "Water tap, faucet",
    11: "Bell",
    12: "Knock",
}

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

def parse_class_start_list(raw_reply: str, win_len: float) -> List[Dict[str, Any]]:
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
            if not (0.0 <= st_f < win_len + 1e-6):
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

# ---------------- prompt & model call ----------------

def build_prompt(present_classes: List[int], win_len: float) -> str:
    # 生成 “ID: 含义” 列表字符串
    cls_list_str = "[" + ", ".join(
        f"{cid}: {CLASS_ID_TO_NAME.get(cid, 'Unknown')}"
        for cid in sorted(set(present_classes))
    ) + "]"
    # 提示词：强调相对时间 & 禁止多余文本/代码块
    prompt = (
        f"You will be given a {win_len:.0f}-second audio segment.\n"
        f"The following target class IDs are present in THIS segment (ID: meaning): {cls_list_str}.\n"
        "Task: For EACH listed class ID, report the EARLIEST start time within THIS segment.\n"
        "Return ONLY a strict JSON array with EXACTLY one object per listed class, sorted by class id.\n"
        "{\"class\": ID, \"start\": seconds} for each object.\n"
        f"Constraints: 0 <= start < {win_len:.1f}. 'start' must be RELATIVE to THIS segment (0 means the segment start).\n"
        "Do NOT include any class not in the list. No extra text. No markdown. No code fences."
    )
    return prompt

def call_model_on_window(audio_arr, sr, present_classes: List[int], win_len: float) -> str:
    """Prompt dynamically with present_classes (ID + meaning) and return raw content string."""
    prompt = build_prompt(present_classes, win_len)

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

def split_fixed_win(wav_path: str, win_len: float):
    """
    Return (wav, sr, windows)
    windows = list of (i0, i1, t0, t1) with exact ts each; last tail (<win_len) is dropped.
    """
    wav, sr = sf.read(wav_path)
    # 若是多通道，转单通道（均值）
    if hasattr(wav, "ndim") and wav.ndim > 1:
        import numpy as np
        wav = np.mean(wav, axis=1)
    n = len(wav)
    samples = int(win_len * sr)
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
                         replies_jsonl_path: str,
                         win_len: float):
    """
    对每个窗口：
      - 由 GT CSV 计算“每类在该窗口的最早绝对时间”（gt_start，绝对时间）
      - 提示模型仅返回这些类的“最早开始相对时间” -> 转为 pred_start = win_start + 相对时间（绝对时间）
      - 逐类写入一行：file, win_start, win_end, class, gt_start, pred_start （都绝对时间）
      - 原始模型回复以追加方式写入 replies_jsonl
    """
    wav, sr, wins = split_fixed_win(wav_path, win_len)
    base = os.path.basename(wav_path)

    os.makedirs(os.path.dirname(replies_jsonl_path) or ".", exist_ok=True)
    append_buffer: List[Dict[str, Any]] = []

    with open(replies_jsonl_path, "a", encoding="utf-8") as fout:
        for (i0, i1, t0, t1) in wins:
            gt_abs_map = earliest_class_abs_time_in_window(events_df, base, t0, t1)
            if not gt_abs_map:
                continue

            clip = wav[i0:i1]
            present_classes = sorted(gt_abs_map.keys())
            if len(present_classes) > 4:
                present_classes = random.sample(present_classes, 4)

            # 调模型（带 win_len）
            raw = call_model_on_window(clip, sr, present_classes, win_len)

            # 保存原始回复（按窗口）
            fout.write(json.dumps({
                "file": base,
                "win_start": t0,
                "win_end": t1,
                "classes": present_classes,
                "reply": raw
            }, ensure_ascii=False) + "\n")

            # 解析模型相对开始时间（带 win_len 约束）
            pred_list = parse_class_start_list(raw, win_len)  # [{"class":ID,"start":rel}, ...]
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
    ]

    # 读取事件 CSV
    df_tau  = load_events_csv(TAU_EVENTS_CSV)  if os.path.exists(TAU_EVENTS_CSV)  else None
    df_sony = load_events_csv(SONY_EVENTS_CSV) if os.path.exists(SONY_EVENTS_CSV) else None

    for wav_path in files:
        # 针对每个窗口长度分别评测 & 产出文件
        for win_len in WIN_LENS:
            # 每个 win_len 独立的输出文件名（便于区分）
            win_tag = f"win{int(win_len):02d}"  # 5->win05, 120->win120
            replies_jsonl = f"{wav_path}.{win_tag}.4limit.replies.jsonl"

            if is_tau_path(wav_path):
                if df_tau is None:
                    raise FileNotFoundError(f"TAU events CSV not found: {TAU_EVENTS_CSV}")
                out_csv = f"tau.{win_tag}_earliest.csv"
                process_file_windows(wav_path, df_tau, out_csv, replies_jsonl, win_len)

            elif is_sony_path(wav_path):
                if df_sony is None:
                    raise FileNotFoundError(f"SONY events CSV not found: {SONY_EVENTS_CSV}")
                out_csv = f"sony.{win_tag}_earliest.csv"
                process_file_windows(wav_path, df_sony, out_csv, replies_jsonl, win_len)

            else:
                print(f"[WARN] Unknown domain (neither sony nor tau): {wav_path}")
