import os
import random
import json
import re
import soundfile as sf
import pandas as pd
import librosa
import torch
from typing import List, Dict, Any, Optional
from transformers import Qwen2AudioForConditionalGeneration, Qwen2AudioProcessor

# ========== MULTI-WINDOW CONFIG ==========
WIN_LENS = [5, 10, 20, 30, 40, 50, 60, 90, 120]  # seconds; 一次跑多个窗口长度

TAU_EVENTS_CSV  = "/data/user/jzt/crd/audioLLM/train_events/tau.csv"
SONY_EVENTS_CSV = "/data/user/jzt/crd/audioLLM/train_events/sony.csv"

# 生成参数（与原脚本一致）
TEMPERATURE = 0.2
TOP_P = 0.95

# ========== QWEN2-AUDIO CONFIG ==========
# 如果你通过 ModelScope/HF 已经把权重下到了本地，把路径改成你的本地目录
QWEN_LOCAL_DIR = "/data/user/jzt/.cache/modelscope/hub/models/Qwen/Qwen2-Audio-7B-Instruct"
QWEN_MAX_NEW_TOKENS = 4096   # 对“只输出 JSON”已足够；需要的话可降到 512
QWEN_DO_SAMPLE = True        # 若想更严格 JSON，可改为 False（贪心解码）

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
    """从文本中提取 JSON 数组（处理转义字符和单引号）。"""
    if not text:
        return None
    # 处理转义字符：确保反斜杠 \ 不会影响 JSON 格式
    text = text.replace(r'\"', '"')  # 转换转义的引号
    text = text.replace(r"\'", "'")  # 转换转义的单引号（如果有）
    text = text.replace(r'\\', '\\')  # 保持反斜杠不变

    # 匹配 JSON 数组
    m = re.search(r"\[.*\]", text, flags=re.S | re.I)
    if m:
        return m.group(0)
    return None

def _find_top_level_array(s: str) -> Optional[str]:
    """递归查找顶层的 JSON 数组。"""
    start = s.find('[')
    while start != -1:
        i, depth, in_str, esc, quote_char = start, 0, False, False, None
        while i < len(s):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == quote_char:
                    in_str = False
                    quote_char = None
            else:
                if ch == '"' or ch == "'":
                    in_str = True
                    quote_char = ch
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
    """解析 JSON 字符串并提取 class 和 start 字段。"""
    arr_text = extract_json_array(raw_reply)
    if not arr_text:
        return []
    try:
        # 将单引号转换为双引号，以兼容 JSON 标准
        json_text = arr_text.replace("'", '"')
        data = json.loads(json_text)  # 解析 JSON 数据
    except json.JSONDecodeError:
        return []  # 如果解析失败，返回空列表

    out = []
    if isinstance(data, list):
        for e in data:
            if isinstance(e, dict):
                # 处理 class 和 start
                c = e.get("class")
                st = e.get("start")
                try:
                    c_int = int(c)  # 将 class 转为整数
                    st_f = float(st)  # 将 start 转为浮动类型
                    # 检查有效范围
                    if c_int < 0 or c_int > 12 or not (0.0 <= st_f < win_len + 1e-6):
                        continue
                    out.append({"class": c_int, "start": st_f})
                except (ValueError, TypeError):
                    continue  # 如果解析出错则跳过该项
    return out

# ---------------- 模型加载（一次性） ----------------

_qwen_processor = Qwen2AudioProcessor.from_pretrained(QWEN_LOCAL_DIR, sampling_rate=16000)  # 添加 sampling_rate
_qwen_model = Qwen2AudioForConditionalGeneration.from_pretrained(
    QWEN_LOCAL_DIR, device_map={"": 0}, dtype=torch.float16
).eval()
_QWEN_TARGET_SR = _qwen_processor.feature_extractor.sampling_rate  # 通常 16000

# ---------------- 提示词生成 + 模型调用 ----------------

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
    """
    用 Qwen2-Audio-7B-Instruct 对当前窗口推理：
    - 提示词包含 ID->含义
    - 输入是“窗口内的 numpy 音频数组 + 文本指令”
    - 返回原始字符串（供 parse_class_start_list 解析）
    """
    prompt = build_prompt(present_classes, win_len)

    # Qwen 要求采样率与处理器一致（通常 16k）
    if sr != _QWEN_TARGET_SR:
        audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=_QWEN_TARGET_SR)

    # 按 ChatML 构造带 audio+text 的会话
    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": None},  # 本地数组，不用 URL
            {"type": "text", "text": prompt},
        ]},
    ]
    text = _qwen_processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    # 打包输入并移动到模型设备
    inputs = _qwen_processor(text=text, audio=audio_arr, return_tensors="pt", padding=True)
    inputs = {k: v.to(_qwen_model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    # 生成
    with torch.no_grad():
        gen_ids = _qwen_model.generate(
            **inputs,
            do_sample=QWEN_DO_SAMPLE,            # 更严格 JSON 可改为 False
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_new_tokens=QWEN_MAX_NEW_TOKENS,
        )
    # 只保留新生成部分
    gen_ids = gen_ids[:, inputs["input_ids"].size(1):]

    # 解码为字符串
    response = _qwen_processor.batch_decode(
        gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response

# ---------------- windowing and GT computation ----------------

def split_fixed_win(wav_path: str, win_len: float):
    """
    Return (wav, sr, windows)
    windows = list of (i0, i1, t0, t1) with exact timestamps each; last tail (<win_len) is dropped.
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
      - 由 GT CSV 计算“每类在该窗口的最早绝对时间”（gt_start，绝对）
      - 让模型仅返回这些类在窗口内的“最早开始相对时间” -> pred_start = win_start + 相对时间（绝对）
      - 逐类写入：file, win_start, win_end, class, gt_start, pred_start
      - 原始模型回复逐窗口写入 replies_jsonl
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
            # 如果类别大于4，随机选4个类别（如需全类，请删除这一段）
            if len(present_classes) > 4:
                present_classes = random.sample(present_classes, 4)

            # 调模型（Qwen2-Audio）
            raw = call_model_on_window(clip, sr, present_classes, win_len)
            
            # 保存原始回复（按窗口）
            fout.write(json.dumps({
                "file": base,
                "win_start": t0,
                "win_end": t1,
                "classes": present_classes,
                "reply": raw
            }, ensure_ascii=False) + "\n")

            # 解析模型相对开始时间
            pred_list = parse_class_start_list(raw, win_len)  # [{"class":ID,"start":rel}, ...]
            pred_rel_map = {int(d["class"]): float(d["start"]) for d in pred_list}

            # 汇总行：pred_start = t0 + rel_start   ;   gt_start 已是绝对时间
            for cid in present_classes:
                gt_abs = float(gt_abs_map[cid])                       # 绝对时间
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

    # 读取事件 CSV
    df_tau  = load_events_csv(TAU_EVENTS_CSV)  if os.path.exists(TAU_EVENTS_CSV)  else None
    df_sony = load_events_csv(SONY_EVENTS_CSV) if os.path.exists(SONY_EVENTS_CSV) else None

    for wav_path in files:
        # 针对每个窗口长度分别评测 & 产出文件
        for win_len in WIN_LENS:
            win_tag = f"win{int(win_len):02d}"  # 5->win05, 120->win120
            replies_jsonl = f"{wav_path}.{win_tag}.4limit.replies.Qwen.jsonl"

            if is_tau_path(wav_path):
                if df_tau is None:
                    raise FileNotFoundError(f"TAU events CSV not found: {TAU_EVENTS_CSV}")
                out_csv = f"Qwen_tau.{win_tag}_earliest.csv"
                process_file_windows(wav_path, df_tau, out_csv, replies_jsonl, win_len)

            elif is_sony_path(wav_path):
                if df_sony is None:
                    raise FileNotFoundError(f"SONY events CSV not found: {SONY_EVENTS_CSV}")
                out_csv = f"Qwen_sony.{win_tag}_earliest.csv"
                process_file_windows(wav_path, df_sony, out_csv, replies_jsonl, win_len)

            else:
                print(f"[WARN] Unknown domain (neither sony nor tau): {wav_path}")
