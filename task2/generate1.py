import pandas as pd
import os
from pydub import AudioSegment

# 文件路径
A_PATH = '/data/user/jzt/crd/audioLLM/task2/fillwave/music/fill_front_5.wav'
B_PATH = '/data/user/jzt/crd/audioLLM/task2/fillwave/music/fill_back_5.wav'
C_PATH = '/data/user/jzt/crd/audioLLM/task2/fillwave/music/fill_mid.wav'
A = AudioSegment.from_wav(A_PATH)  # 读取音频A
B = AudioSegment.from_wav(B_PATH)  # 读取音频B
C = AudioSegment.from_wav(C_PATH)  # 读取音频C

# 输入和输出路径
INPUT_CSV = '/data/user/jzt/crd/audioLLM/task2/tau_event.csv'
OUTPUT_AUDIO_DIR = 'tau11/'
MANIFEST_CSV = 'tau11/output_manifest.csv'
SOURCE_AUDIO_DIR = '/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/'

# 目标类别集合
TARGET_CLASSES = {0,1}

def process_audio_events(df: pd.DataFrame):
    manifest_data = []
    
    # 遍历每一行数据
    for _, row in df.iterrows():
        file_path = row['file']
        class_label = row['class']
        onset = row['onset']
        offset = row['offset']

        # 只处理目标类别
        if class_label in TARGET_CLASSES:
            # 读取原始音频文件
            audio_file_path = os.path.join(SOURCE_AUDIO_DIR, file_path)
            original_audio = AudioSegment.from_wav(audio_file_path)
            
            # 获取事件的音频片段
            event_audio = original_audio[int(onset * 1000):int(offset * 1000)]  # 转为毫秒
            
            # 计算需要填充的音频C时长
            event_duration = len(event_audio) / 1000  # 转为秒
            mid_duration = (10 - event_duration) / 2  # C音频长度，确保总长度为10秒
            
            if mid_duration > 0:
                mid_audio = C[:int(mid_duration * 1000)]  # 取出C的部分
            else:
                mid_audio = AudioSegment.silent(duration=0)  # 若无需填充C，则为空音频

            # 拼接音频：前A、事件音频、中C、后B
            final_audio = A + mid_audio + event_audio + mid_audio + B

            # 确保最终音频为20秒
            if len(final_audio) < 20000:
                final_audio = final_audio + AudioSegment.silent(duration=20000 - len(final_audio))

            # 计算事件音频在 final_audio 中的相对开始和结束时间
            relative_onset_event_audio = 5 + mid_duration  # 事件音频的相对开始时间
            relative_offset_event_audio = 5 + mid_duration + event_duration   # 事件音频的相对结束时间

            # 生成新的文件名，包含原始时间信息
            base_filename = os.path.basename(file_path)
            new_filename = f"{base_filename.split('.')[0]}_class_{class_label}_onset_{onset}_offset_{offset}.wav"
            output_file_path = os.path.join(OUTPUT_AUDIO_DIR, new_filename)

            # 保存最终的音频文件
            final_audio.export(output_file_path, format="wav")

            # 记录manifest数据
            manifest_data.append({
                'file': output_file_path,
                'class': class_label,
                'onset': onset,
                'offset': offset,
                'relative_onset_event_audio': relative_onset_event_audio,  # 事件音频的相对开始时间
                'relative_offset_event_audio': relative_offset_event_audio  # 事件音频的相对结束时间
            })
    
    # 保存manifest文件
    manifest_df = pd.DataFrame(manifest_data)
    manifest_df.to_csv(MANIFEST_CSV, index=False)
    print(f"已保存 {len(manifest_data)} 条数据到 {MANIFEST_CSV}")

def main():
    # 读取原始的CSV数据
    df = pd.read_csv(INPUT_CSV)
    
    # 清洗和过滤数据
    required_cols = {"file", "onset", "offset", "class"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"输入CSV缺少列：{missing}")
    
    # 只保留目标类别的数据
    df = df[df['class'].isin(TARGET_CLASSES)]
    
    # 处理音频事件并生成manifest
    process_audio_events(df)

if __name__ == "__main__":
    main()
