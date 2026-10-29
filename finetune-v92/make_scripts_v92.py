# -*- coding: utf-8 -*-
r"""v92 F：從 finetune-v91 複製訓練／挑選／探針／匯入／合併的腳本（只換路徑、來源 p4、卡 D；每一處斷言剛好一次）。

  python make_scripts_v92.py
"""
KAMIE_ROOT = __import__('os').environ.get('KAMIE_ROOT', __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))).replace(chr(92), '/')
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
V91 = os.path.join(ROOT, 'finetune-v91')
JOBS = [
    ('train_v91.py', 'train_v92.py', [
        ('r"""v91 F：', 'r"""v92 F：＝finetune-v91/train_v91.py，只改驗證 loss 的來源多一個 p4（第四包）；1 epoch 由呼叫端 --epochs 1 給（規劃端補充指令 v92 第 2 點）。以下 v91 原文：\nv91 F：'),
        ("assert set(vgrp) <= {'p1', 'p2', 'p3'}, set(vgrp)", "assert set(vgrp) <= {'p1', 'p2', 'p3', 'p4'}, set(vgrp)"),
    ]),
    ('eval_ckpt_v91.py', 'eval_ckpt_v92.py', [
        ('r"""v91：＝finetune-v90', 'r"""v92：＝finetune-v91/eval_ckpt_v91.py 逐字（路徑跟著檔案在 finetune-v92）。以下 v91 原文：\nv91：＝finetune-v90'),
    ]),
    ('probe_v91.py', 'probe_v92.py', [
        ('r"""v91：＝finetune-v90', 'r"""v92：＝finetune-v91/probe_v91.py，只改探針來源＝第四包驗證集（data/qwen-p4-val.jsonl，種子 8707 抽 24 筆）。以下 v91 原文：\nv91：＝finetune-v90'),
        ("'%s-p3-val.jsonl' % a.family", "'%s-p4-val.jsonl' % a.family"),
    ]),
    ('ollama_side_v91.py', 'ollama_side_v92.py', [
        ('r"""v91：＝finetune-v90', 'r"""v92：＝finetune-v91/ollama_side_v91.py，只改「回一句」的 system＝卡 D（mainline-v92/train-system-cardD.json）。以下 v91 原文：\nv91：＝finetune-v90'),
        ("os.path.join(ROOT, 'mainline-v91', 'train-system-cardC.json')", "os.path.join(ROOT, 'mainline-v92', 'train-system-cardD.json')"),
    ]),
    ('chain_v91.sh', 'chain_v92.sh', [
        ('cd "%s/finetune-v91"' % KAMIE_ROOT, 'cd "%s/finetune-v92"' % KAMIE_ROOT),
    ]),
    ('keepawake_run_v91.ps1', 'keepawake_run_v92.ps1', []),
]
for src, dst, reps in JOBS:
    enc = 'utf-8-sig' if src.endswith('.ps1') else 'utf-8'
    raw = open(os.path.join(V91, src), 'rb').read()
    s = raw.decode(enc)
    for a, b in reps:
        assert s.count(a) == 1, (src, a, s.count(a))
        s = s.replace(a, b)
    nl = '\r\n' if b'\r\n' in raw else '\n'
    s = s.replace('\r\n', '\n').replace('\n', nl)
    open(os.path.join(HERE, dst), 'wb').write(s.encode(enc))
    print('寫入', dst)
