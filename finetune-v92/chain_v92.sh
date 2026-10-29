#!/usr/bin/env bash
# v90：adapter → 合併 → bf16 GGUF → 一種量化 → 刪 bf16 GGUF（一次只留一種量化；§8 v85 磁碟）。合併資料夾留著給探針（呼叫端刪）。
#   bash finetune-v90/chain_v90.sh qwen  <tag> <ckpt> Q4_K_M     4B：finetune-v87\merge_qwen_v87.py（G2）→ .venv 轉檔 --no-mtp（v69 G3）
#   bash finetune-v90/chain_v90.sh gemma <tag> <ckpt> Q5_K_M     Gemma：finetune-v88\merge_gemma_v88.py --mode deq（做法一）→ .venv-train 轉檔（§8 v85）
set -u
FAM="${1:?family}"; TAG="${2:?tag}"; CK="${3:?ckpt}"; Q="${4:?quant}"
cd "$(dirname "$0")"                     # v91：＝finetune-v90/chain_v90.sh，只換資料夾
TPY="../finetune-v68/.venv-train/Scripts/python.exe"
CPY="../finetune-v68/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8
mkdir -p gguf merged results
L="$TAG-$CK"
q=$(echo "$Q" | tr 'A-Z' 'a-z' | tr -d '_')
echo "== $(date +%T) merge $FAM $L"
if [ "$FAM" = qwen ]; then
  "$TPY" ../finetune-v87/merge_qwen_v87.py --label "$CK" --adapter "adapters/$TAG/$CK" --out "merged/$L" --result "results/merge-$L.json" 2>&1 | grep -v "^\[transformers\]" | tail -3
  [ -f "merged/$L/model.safetensors" ] || { echo "no merged model"; exit 1; }
  echo "== $(date +%T) convert bf16 (--no-mtp)"
  "$CPY" ../finetune-v68/llama.cpp/convert_hf_to_gguf.py "merged/$L" --outfile "gguf/$L-bf16.gguf" --outtype bf16 --no-mtp 2>&1 | grep -v "Writing:\|^INFO:hf-to-gguf:gguf: \|blk\.\|^INFO:gguf.gguf_writer:" | tail -3
else
  "$TPY" ../finetune-v88/merge_gemma_v88.py --mode deq --adapter "adapters/$TAG/$CK" --out "merged/$L" --result "results/merge-$L.json" 2>&1 | grep -v "flop_counter" | tail -2
  [ -f "merged/$L/model.safetensors.index.json" ] || { echo "no merged model"; exit 1; }
  echo "== $(date +%T) convert bf16"
  "$TPY" ../finetune-v68/llama.cpp/convert_hf_to_gguf.py "merged/$L" --outfile "gguf/$L-bf16.gguf" --outtype bf16 2>&1 | grep -v "Writing:\|^INFO:hf-to-gguf:gguf: \|blk\.\|^INFO:gguf.gguf_writer:" | tail -3
fi
[ -f "gguf/$L-bf16.gguf" ] || { echo "no bf16 gguf"; exit 1; }
echo "== $(date +%T) quantize $Q"
../finetune-v68/llama-bin/llama-quantize.exe "gguf/$L-bf16.gguf" "gguf/$L-$q.gguf" "$Q" 2>&1 | tail -2
[ -f "gguf/$L-$q.gguf" ] || { echo "no quantized gguf"; exit 1; }
rm -f "gguf/$L-bf16.gguf"
ls -la gguf | grep "$L"
echo "== $(date +%T) CHAIN DONE $L $Q"
