from pydub import AudioSegment

def crop_audio(input_path, start_time, end_time, output_path):
    """
    从音频文件中裁剪出指定时间段的音频并保存。
    :param input_path: 输入音频文件路径
    :param start_time: 裁剪开始时间（秒）
    :param end_time: 裁剪结束时间（秒）
    :param output_path: 输出音频文件路径
    """
    audio = AudioSegment.from_wav(input_path)
    
    # 将秒转换为毫秒
    start_ms = start_time * 1000
    end_ms = end_time * 1000
    
    # 裁剪音频
    cropped_audio = audio[start_ms:end_ms]
    
    # 导出裁剪后的音频
    cropped_audio.export(output_path, format="wav")
    print(f"已保存裁剪音频: {output_path}")

def repeat_audio(input_path, repeat_times, output_path):
    """
    将音频重复指定次数并保存。
    :param input_path: 输入音频文件路径
    :param repeat_times: 重复次数
    :param output_path: 输出音频文件路径
    """
    audio = AudioSegment.from_wav(input_path)
    
    # 重复音频
    repeated_audio = audio * repeat_times
    
    # 导出重复后的音频
    repeated_audio.export(output_path, format="wav")
    print(f"已保存重复音频: {output_path}")

def main():
    # 1. 裁剪 8.0 到 12.0 秒的音频并重复 10 次
    crop_audio("/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix001.wav", 4.0, 13.0, "man_speech.wav")

    # 2. 裁剪 38.0 到 43.0 秒的音频并保存
    crop_audio("/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix021.wav", 60.0, 80.0, "60-80.wav")
    crop_audio("/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix008.wav", 17.0, 27.0, "17-27.wav")
    crop_audio("/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix008.wav", 50.0, 61.0, "50-61.wav")

if __name__ == "__main__":
    main()
