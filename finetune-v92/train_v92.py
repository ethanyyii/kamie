# -*- coding: utf-8 -*-
r"""v92 F：＝finetune-v91/train_v91.py，只改驗證 loss 的來源多一個 p4（第四包）；1 epoch 由呼叫端 --epochs 1 給（規劃端補充指令 v92 第 2 點）。以下 v91 原文：
v91 F：＝finetune-v90\train_v90.py 逐字複製，只改兩處：① --data m1|m2 ② 驗證 loss 按 src 分三個來源（p1／p2／p3）。路徑跟著檔案在 finetune-v91。
以下是 v90 原文：
v90 C：＝finetune-v87\train_v87.py 逐字複製，只改三處：① --data p2|mix（資料＝convert_v90.py 的 data\<family>-<data>-<split>.jsonl）
② 驗證 loss 另外按來源分開記（p2＝第二包、p1＝第一包；val_curve_by_src）③ 路徑在 finetune-v90。超參數、種子、挑選規則都沒動（plan-v90 §六）。
以下是 v87 原文：
v87 C-2／C-3／C-4：訓練（兩顆同一支）。資料＝convert_v87.py 的 data\<family>-<split>.jsonl（提示＝HF 模板、已對 Ollama 逐位元組）。

  qwen ：bf16 LoRA r16 α16 all-linear（v82／v86 同）
  gemma：QLoRA（NF4＋double quant、bf16 計算）、LoRA r16 α16 只掛語言模型七種線性層、softcap（v85 同）
  共同：只對目標 token 算 loss（分段算 logits）、GC、batch 1＋梯度累積、AdamW、warmup＋cosine、顯存上限＝開跑時實際空著的（§8 v68：Windows 不加會無聲溢出）
  checkpoint：每 1/4 epoch 存一次 adapter，同時在驗證集算 loss（目標 token 平均 NLL）；每步記時間、loss；結束寫 results\train-<tag>.json
  --sentinel：拋棄式哨兵（data\<family>-sentinel.jsonl、不累積、lr 2e-4、2 epoch、只存最後）

  ..\finetune-v68\.venv-train\Scripts\python.exe train_v87.py --family qwen --tag qwen-formal [--epochs 2 --lr 1e-4 --accum 8]
"""
import argparse, io, json, math, os, random, sys, time, traceback

ap = argparse.ArgumentParser()
ap.add_argument('--family', required=True, choices=['qwen', 'gemma'])
ap.add_argument('--tag', required=True)
ap.add_argument('--data', required=True, choices=['m1', 'm2'])                # v91：m1＝舊＋新、m2＝舊＋新×2（convert_v91.py）
ap.add_argument('--sentinel', action='store_true')
ap.add_argument('--epochs', type=float, default=2)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--accum', type=int, default=8)
ap.add_argument('--warmup', type=int, default=20, help='optimizer steps')
ap.add_argument('--ckpt-per-epoch', type=int, default=4)
ap.add_argument('--seed', type=int, default=8705)
ap.add_argument('--max-steps', type=int, default=0, help='>0：只跑這麼多個 optimizer step（煙霧測試）')
args = ap.parse_args()
if args.sentinel:
    args.lr, args.accum, args.warmup, args.ckpt_per_epoch = 2e-4, 1, 0, 0
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HF = {'qwen': os.path.join(ROOT, 'finetune-v68', 'hf', 'Qwen3.5-4B'), 'gemma': os.path.join(ROOT, 'finetune-v85', 'hf', 'gemma-4-12B-it-qat-unq')}[args.family]
ADIR = os.path.join(HERE, 'adapters', args.tag)
OUT = os.path.join(HERE, 'results', 'train-%s.json' % args.tag)
os.makedirs(ADIR, exist_ok=True)


def rows(split):
    return [json.loads(l) for l in io.open(os.path.join(HERE, 'data', '%s-%s-%s.jsonl' % (args.family, args.data, split)), encoding='utf-8')]


res = {'args': vars(args), 'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'hf': HF}
try:
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF)
    train = rows('sentinel' if args.sentinel else 'train')
    val = [] if args.sentinel else rows('val')

    def encode(r):
        p = tok(r['prompt'], add_special_tokens=False)['input_ids']
        t = tok(r['target'], add_special_tokens=False)['input_ids']
        return p + t, [-100] * len(p) + t
    enc = [encode(r) for r in train]
    venc = [encode(r) for r in val]
    vgrp = [r['src'] for r in val]                                         # v91：驗證 loss 按來源分開記（p1＝第一包、p2＝第二包、p3＝第三包）
    assert set(vgrp) <= {'p1', 'p2', 'p3', 'p4'}, set(vgrp)
    dec = tok.decode([x for x in enc[0][1] if x != -100])
    assert dec == train[0]['target'], ('🔴 G0', dec, train[0]['target'])
    if args.family == 'gemma':
        assert tok.decode(enc[0][0]).startswith('<bos>') and enc[0][0].count(2) == 1
    res['G0'] = {'trains_example': dec, 'n_train': len(enc), 'n_val': len(venc), 'seq_max': max(len(i) for i, _ in enc),
                 'target_tokens_train': sum(sum(1 for x in l if x != -100) for _, l in enc)}
    print('G0 ok：train %d／val %d、最長 %d token' % (len(enc), len(venc), res['G0']['seq_max']), flush=True)

    free, total = torch.cuda.mem_get_info()
    frac = max(0.05, (free - 256 * 2**20) / total)
    torch.cuda.set_per_process_memory_fraction(frac)
    res['cap_mib'] = round(frac * total / 2**20)
    from peft import LoraConfig, get_peft_model
    if args.family == 'qwen':
        from transformers import Qwen3_5ForCausalLM
        model, info = Qwen3_5ForCausalLM.from_pretrained(HF, dtype=torch.bfloat16, device_map='cuda', output_loading_info=True)
        target_modules = 'all-linear'
    else:
        from transformers import BitsAndBytesConfig, Gemma4UnifiedForConditionalGeneration
        model, info = Gemma4UnifiedForConditionalGeneration.from_pretrained(HF, dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa', output_loading_info=True,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_storage=torch.bfloat16))
        target_modules = r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$'
    assert not info['missing_keys'] and not info['mismatched_keys'], 'weights were re-initialised'
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, target_modules=target_modules, task_type='CAUSAL_LM'))
    res['params_trainable'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    inner = model.get_base_model()
    if args.family == 'qwen':
        body, head, softcap = inner.model, inner.lm_head, None
    else:
        body, head, softcap = inner.model.language_model, inner.lm_head, inner.config.get_text_config().final_logit_softcapping

    def loss_of(ids, labels):
        x = torch.tensor([ids]).cuda(); y = torch.tensor([labels]).cuda()
        h = body(input_ids=x).last_hidden_state[:, :-1]
        tgt = y[:, 1:]; sel = tgt[0] != -100
        logits = head(h[0][sel]).float()
        if softcap:
            logits = torch.tanh(logits / softcap) * softcap
        return torch.nn.functional.cross_entropy(logits, tgt[0][sel], reduction='sum'), int(sel.sum())

    def val_loss(label):
        if not venc:
            return None
        model.eval()
        tot, n = 0.0, 0
        grp = {}
        with torch.no_grad():
            for (ids, lab), g in zip(venc, vgrp):
                l, k = loss_of(ids, lab)
                tot += l.item(); n += k
                a = grp.setdefault(g, [0.0, 0]); a[0] += l.item(); a[1] += k
        model.train()
        res.setdefault('val_curve_by_src', {})[label] = {g: round(v[0] / v[1], 5) for g, v in grp.items()}
        return round(tot / n, 5)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    n_micro = int(round(len(enc) * args.epochs))
    n_opt = math.ceil(n_micro / args.accum)
    if args.max_steps:
        n_opt = min(n_opt, args.max_steps)
        n_micro = n_opt * args.accum

    def lr_at(s):                                                            # s＝已完成的 optimizer step 數
        if args.sentinel:
            return args.lr
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        prog = (s - args.warmup) / max(1, n_opt - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))
    ck_every = 0 if not args.ckpt_per_epoch else max(1, round(len(enc) / args.accum / args.ckpt_per_epoch))
    res.update({'n_micro': n_micro, 'n_opt': n_opt, 'ckpt_every_opt_steps': ck_every})
    print('micro %d、optimizer step %d、每 %d 步一個 checkpoint、可訓練 %d 參數、上限 %d MiB' % (n_micro, n_opt, ck_every, res['params_trainable'], res['cap_mib']), flush=True)
    res['val_curve'] = {'step0': val_loss('step0')}
    print('val loss step0 %s' % res['val_curve']['step0'], flush=True)
    rnd = random.Random(args.seed)
    order = []
    while len(order) < n_micro:
        o = list(range(len(enc))); rnd.shuffle(o); order += o
    order = order[:n_micro]
    model.train()
    times, losses = [], []
    torch.cuda.reset_peak_memory_stats()
    step, acc_l, acc_n = 0, 0.0, 0
    for i_micro, k in enumerate(order):
        t = time.time()
        ids, labels = enc[k]
        l, nk = loss_of(ids, labels)
        (l / nk / args.accum).backward()
        acc_l += l.item(); acc_n += nk
        if (i_micro + 1) % args.accum == 0 or i_micro + 1 == n_micro:
            for g in opt.param_groups:
                g['lr'] = lr_at(step)
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            losses.append(round(acc_l / acc_n, 4)); acc_l, acc_n = 0.0, 0
            if ck_every and step % ck_every == 0:
                model.save_pretrained(os.path.join(ADIR, 'step%d' % step))
                vl = val_loss('step%d' % step)
                res['val_curve']['step%d' % step] = vl
                json.dump(res | {'losses': losses, 'partial': True}, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
                print('ckpt step %d  train %.4f  val %.5f  lr %.2e  reserved %.0f MiB' % (step, losses[-1], vl, lr_at(step - 1), torch.cuda.max_memory_reserved() / 2**20), flush=True)
            elif step % 25 == 0:
                w = sorted(times[-200:])
                print('step %d/%d  train %.4f  micro median %.2f s  reserved %.0f MiB' % (step, n_opt, losses[-1], w[len(w) // 2], torch.cuda.max_memory_reserved() / 2**20), flush=True)
        torch.cuda.synchronize(); times.append(time.time() - t)
    model.save_pretrained(os.path.join(ADIR, 'step%d' % step))
    res['val_curve']['step%d' % step] = val_loss('step%d' % step)
    w = sorted(times[3:])
    res.update({'ok': True, 'steps': step, 'losses': losses, 'peak_reserved_mib': round(torch.cuda.max_memory_reserved() / 2**20),
                'micro_s': {'n': len(w), 'p10': round(w[int(len(w) * .1)], 3), 'median': round(w[len(w) // 2], 3), 'p90': round(w[int(len(w) * .9)], 3), 'max': round(w[-1], 3)},
                'checkpoints': sorted(os.listdir(ADIR))})
    print('done: %d optimizer steps, reserved %d MiB, micro median %.2f s, val %s' % (step, res['peak_reserved_mib'], res['micro_s']['median'], res['val_curve']), flush=True)
except Exception as e:
    res.update({'ok': False, 'error_type': type(e).__name__, 'error': str(e)[:2000], 'traceback_tail': traceback.format_exc()[-3000:]})
    print('🔴 FAILED', type(e).__name__, str(e)[:800], flush=True)
finally:
    res['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    res.pop('partial', None)
    json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
sys.exit(0 if res.get('ok') else 1)
