#!/usr/bin/env bash
# v92 今晚（規劃端補充指令 v92 今晚一顆）：轉格式（舊＋第四包，同時渲染）→ 組 m1 → 保留探針那 24 筆 → F-1 v92-a（1 epoch）→ 挑 checkpoint
#   → HF 探針 → 合併＋轉檔＋量化 → 合併後探針 → 匯入 v92-a:q4km（G6 渲染＋回一句有 ACT）→ 量化後探針 → 保留報告 → 卸模型、停 Ollama。
# ＝finetune-v91/run_f_v91.sh 的 run_one（沒有截止估法：補充指令第 3 點「超過 7 小時照樣做完」）。Ollama（ollama serve）由呼叫端先起好。
#   Start-Process pwsh -File finetune-v92\keepawake_run_v92.ps1 -Log finetune-v92\logs\run-night.out "C:\Program Files\Git\bin\bash.exe" finetune-v92/run_night_v92.sh
# 每一步印「== 時間 …」；結束一定印「RUN-NIGHT ALL-END」（失敗也會印，前面有 FAILED）。
set -u
cd "$(dirname "$0")"
TPY="../finetune-v68/.venv-train/Scripts/python.exe"
export PYTHONIOENCODING=utf-8
mkdir -p logs results
log() { echo "== $(date '+%F %T') $*"; }
unload_all() {
  "$TPY" -c "
import json, urllib.request
try:
    ps = json.loads(urllib.request.urlopen('http://127.0.0.1:11434/api/ps', timeout=10).read())
    for m in ps.get('models', []):
        urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:11434/api/generate', data=json.dumps({'model': m['name'], 'keep_alive': 0}).encode(), headers={'Content-Type': 'application/json'}), timeout=120).read()
        print('unloaded', m['name'])
    print('ollama models loaded now:', len(json.loads(urllib.request.urlopen('http://127.0.0.1:11434/api/ps', timeout=10).read()).get('models', [])))
except Exception as e:
    print('ollama not reachable:', e)
"
}
gpu() { nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; }
stop_ollama() {
  powershell.exe -NoProfile -Command "foreach(\$n in 'llama-server','ollama'){ Get-Process -Name \$n -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id \$_.Id -Force; \"stopped \$n \$(\$_.Id)\" } }"
}
finish() { unload_all; stop_ollama; log "gpu at end: $(gpu)"; log "RUN-NIGHT ALL-END $1"; exit 0; }

FAM=qwen; DATA=m1; TAG=qwen-a; NAME=v92-a:q4km; Q=Q4_K_M; q=q4km
log "RUN-NIGHT start; gpu $(gpu)"
log "convert old + p4 (parallel)"
"$TPY" convert_v92.py old > logs/convert-old.out 2>&1 & P1=$!
"$TPY" convert_v92.py p4 > logs/convert-p4.out 2>&1 & P2=$!
wait $P1; R1=$?
wait $P2; R2=$?
tail -3 logs/convert-old.out; tail -3 logs/convert-p4.out
if [ $R1 -ne 0 ] || [ $R2 -ne 0 ]; then log "CONVERT-FAILED old=$R1 p4=$R2"; finish FAILED; fi
"$TPY" convert_v92.py mix > logs/convert-mix.out 2>&1 || { log "MIX-FAILED"; tail -5 logs/convert-mix.out; finish FAILED; }
tail -2 logs/convert-mix.out
log "probe build"
"$TPY" probe_v92.py build --family qwen > logs/probe-build.out 2>&1 || log "PROBE-BUILD-FAILED"
tail -1 logs/probe-build.out
unload_all
log "gpu before train: $(gpu)"
log "train start (1 epoch)"
"$TPY" train_v92.py --family "$FAM" --data "$DATA" --tag "$TAG" --epochs 1 > "logs/train-$TAG.out" 2>&1 || { log "TRAIN-FAILED"; tail -8 "logs/train-$TAG.out"; finish FAILED; }
tail -2 "logs/train-$TAG.out"
log "pick"
"$TPY" eval_ckpt_v92.py pick --family "$FAM" --data "$DATA" --tag "$TAG" > "logs/pick-$TAG.out" 2>&1 || { log "PICK-FAILED"; tail -8 "logs/pick-$TAG.out"; finish FAILED; }
grep -v "^\[transformers\]\|Warning\|warn" "logs/pick-$TAG.out" | tail -12
CK=$("$TPY" -c "import json;print(json.load(open('results/pick-$TAG.json',encoding='utf-8'))['pick'])")
log "picked $CK"
log "probe hf (base/adapter)"
"$TPY" probe_v92.py hf --family "$FAM" --tag "$TAG" --adapter "adapters/$TAG/$CK" > "logs/probe-hf-$TAG.out" 2>&1 || log "PROBE-HF-FAILED"
tail -2 "logs/probe-hf-$TAG.out"
log "chain"
bash chain_v92.sh "$FAM" "$TAG" "$CK" "$Q" > "logs/chain-$TAG.out" 2>&1 || { log "CHAIN-FAILED"; tail -8 "logs/chain-$TAG.out"; rm -rf "merged/$TAG-$CK"; finish FAILED; }
tail -3 "logs/chain-$TAG.out"
log "probe merged"
"$TPY" probe_v92.py merged --family "$FAM" --tag "$TAG" --path "merged/$TAG-$CK" > "logs/probe-merged-$TAG.out" 2>&1 || log "PROBE-MERGED-FAILED"
tail -1 "logs/probe-merged-$TAG.out"
rm -rf "merged/$TAG-$CK"
log "import $NAME"
"$TPY" ollama_side_v92.py import --family "$FAM" --gguf "gguf/$TAG-$CK-$q.gguf" --name "$NAME" > "logs/import-$TAG.out" 2>&1 || log "IMPORT-NOT-OK"
tail -2 "logs/import-$TAG.out"
log "probe ollama"
"$TPY" probe_v92.py ollama --family "$FAM" --tag "$TAG" --name "$NAME" > "logs/probe-ollama-$TAG.out" 2>&1 || log "PROBE-OLLAMA-FAILED"
tail -1 "logs/probe-ollama-$TAG.out"
"$TPY" probe_v92.py report --family "$FAM" --tag "$TAG" > "logs/retention-$TAG.out" 2>&1 && log "RETENTION-OK" || log "RETENTION-NOT-OK"
tail -1 "logs/retention-$TAG.out"
finish DONE
