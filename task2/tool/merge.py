from pydub import AudioSegment

# 设置三段音频的路径
audio1_path = "/data/user/jzt/crd/audioLLM/task2/man_speech.wav"
audio2_path = "/data/user/jzt/crd/audioLLM/task2/60-87.wav"
audio3_path = "/data/user/jzt/crd/audioLLM/task2/17-29.wav"
audio4_path = "/data/user/jzt/crd/audioLLM/task2/50-62.wav"

# 读取音频（pydub 会根据后缀自动识别格式）
audio1 = AudioSegment.from_file(audio1_path)
audio2 = AudioSegment.from_file(audio2_path)
audio3 = AudioSegment.from_file(audio3_path)
audio4 = AudioSegment.from_file(audio4_path)
# 按顺序拼接
combined = audio2  + audio3+ audio4+audio1

# 导出合并后的文件
combined.export("60_man_back.wav", format="wav")

