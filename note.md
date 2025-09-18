# proxy
git config --global --unset http.proxy
git config --global --unset https.proxy
git config --global http.proxy http://127.0.0.1:9090
git config --global https.proxy http://127.0.0.1:9090
PIP_INDEX_URL=https://pypi.org/simple 

# https proxy
export http_proxy=http://127.0.0.1:9090
export https_proxy=http://127.0.0.1:9090

# nvidia/voxtral-mini-3b usage
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

python -m vllm.entrypoints.openai.api_server \
  --model /data/user/jzt/crd/audioLLM/Voxtral-Mini-3B-2507 \
  --served-model-name voxtral-mini-3b \
  --tokenizer_mode mistral \
  --config_format mistral \
  --load_format mistral \
  --dtype auto \
  --max-model-len 10000 \
  --gpu-memory-utilization 0.20 \
  --host 127.0.0.1 --port 8011

## pretrainedSED audio class contrast
⭕ 0: "Female speech, woman speaking",
⭕ 1: "Male speech, man speaking",
⭕ 2: "Clapping",
❓ 3: "Telephone"->"Wind chime",
⭕ 4: "Laughter",
❌ 5: "Domestic sounds",
⭕ 6: "Walk, footsteps",
❓ 7: "Door, open or close"->Generic impact sounds,
⭕ 8: "Music",
⭕ 9: "Musical instrument"->"Music",
⭕ 10: "Water tap, faucet",
❓ 11: "Bell"->"Wind chime",
⭕ 12: "Knock",

# train_data sony/tau
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

# test_data sony/tau
files = [
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-tau/fold4_room2_mix002.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-tau/fold4_room2_mix003.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-tau/fold4_room2_mix004.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-tau/fold4_room2_mix005.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-tau/fold4_room2_mix006.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-tau/fold4_room8_mix001.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-tau/fold4_room8_mix002.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-tau/fold4_room8_mix003.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-tau/fold4_room8_mix004.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-tau/fold4_room8_mix005.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-tau/fold4_room8_mix006.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-tau/fold4_room8_mix007.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-tau/fold4_room8_mix008.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-tau/fold4_room8_mix009.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-tau/fold4_room10_mix001.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-tau/fold4_room10_mix002.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-tau/fold4_room10_mix003.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-tau/fold4_room10_mix004.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-tau/fold4_room10_mix005.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-tau/fold4_room10_mix006.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-tau/fold4_room10_mix007.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-tau/fold4_room10_mix008.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-tau/fold4_room10_mix009.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room23_mix001.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room23_mix001.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room23_mix002.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room23_mix003.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room23_mix004.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room23_mix005.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room23_mix006.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room23_mix007.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room23_mix008.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room23_mix009.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room23_mix010.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room23_mix011.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room23_mix012.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room23_mix013.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room23_mix014.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room24_mix001.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room24_mix002.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room24_mix003.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room24_mix004.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room24_mix005.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room24_mix006.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room24_mix007.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room24_mix008.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room24_mix009.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room24_mix010.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room24_mix011.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room24_mix012.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room24_mix013.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room24_mix014.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room24_mix015.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-test-sony/fold4_room24_mix016.wav",
    ]