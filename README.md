### 说明

- `data_process`：处理大模型回答和真实标注的代码
- `metadata_dev`：真实标注
- `test_data`和`test_event`：最初使用sony和tau的测试数据得的结果
- `train_data`和`train_event`：之后使用sony和tau的训练数据得到的结果
- **`Voxtral_result.xlsx`：整合后的结果**
- `voxtral_test.py`：评测大模型的代码
- `frames_to_events_batch.py`：处理帧级的真实标注的代码

## task2 长短事件音频

- 'voxtral.py'：评测voxtral的代码
- 'qwen'：评测qwen代码
- 'select_events.py'：从真实标注中挑选符合要求的片段
- 'sony_event.csv'：select_events.py得到的sony事件
- 'tau_event.csv'：select_events.py得到的tau事件
- 'generate1.py&generate2.py'：构造评测音频
- 'cal_mae.py'：统计结果
- 'fillwave'：供剪辑使用的音频
- 'tool'：切割音频工具
- 'voxtral_preds.csv'：voxtral的评测结果
- 'qwen_preds.csv'：qwen的评测结果

## task3 不同位置音频

- 'old'：垃圾桶
- 'window_candidates'：每个窗口大小供后续生成测评片段的记录
- 'analyze_segment_mae.py'：统计结果
- 'collect_window_candidates.py'：得到window_candidates的代码
- 'cut_window'：生成评测数据的代码
- 'plot_segment_mae.py'：以曲线显示结果的代码
- 'voxtral.py'：评测voxtral的代码
- 'qwen'：评测qwen代码
- 'voxtral_results.csv'：voxtral的评测结果
- 'qwen_results.csv'：qwen的评测结果
