# -*- coding: utf-8 -*-
r"""v92 轉格式（plan-v92-E §四）。🔴 全部訓練資料（舊＋新）的 system＝mainline-v92\train-system-cardD.json（sha256 斷言）。
＝finetune-v91\convert_v91.py 的做法（函式照抄），來源換成這一輪的：

  old：v91 轉好的訓練目標（finetune-v91\data\qwen-old-{train,val}.jsonl＝第一、二包（src p1／p2）、qwen-p3-{train,val}.jsonl＝第三包（p3）；system＝卡 C）
       ① E-2：拿掉 mainline-v92\e\card\judged.json 的 remove_old（宣布嘴＋卡上事實矛盾＋講設定）
       ② 每一筆的 prompt 拆回 messages；**用卡 C（train-system-cardC）以 HF 模板重組＝原 prompt 逐位元組**（證明拆法對）
       ③ system 換成卡 D → HF 模板重組 → 問 Ollama `_debug_render_only`（qwen3.5:4b、think:false）逐位元組
       → data\qwen-old-{train,val}.jsonl
  p4：第四包定稿（mainline-v92\e\b3\final-v92.json）＝convert_v91 new 那一套（他：`[YYYY-MM-DD HH:MM] User：`＋原文，種子 9203；
       她過去的話拿掉表情標記、開頭一個空白；目標 `<|ACT {…}|> 台詞`＋`<|im_end|>`；不當目標的句子不出一筆、照樣留在後面的歷史）
       → data\qwen-p4-{train,val}.jsonl
  mix：data\qwen-m1-{train,val}.jsonl＝舊＋第四包（F-1 v92-a）
  mix2（第二晚，plan-v92-E3 §三）：data\qwen-m2-{train,val}.jsonl＝舊（再拿掉 E-3 下評語＝是：mainline-v92\ruler\old-removed.json）＋第四包（F-2 v92-b）
       拿掉的鍵＝(src, seg, line, 去標記後的台詞)＋清單上的 file／idx 兩頭都要對；斷言拿掉的筆數＝清單筆數；那一筆不出，其他筆（含後面的歷史）逐字不動
  結果 → results\convert-v92-<old|p4|mix|mix2>.json

  ..\finetune-v68\.venv-train\Scripts\python.exe convert_v92.py old|p4|mix|mix2 [--no-render]
"""
import os as _os_un
USER_NAME = _os_un.environ.get('KAMIE_USER_NAME', 'User')
import argparse, hashlib, io, json, os, random, re, sys, time, urllib.request

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
ap = argparse.ArgumentParser()
ap.add_argument('stage', choices=['old', 'p4', 'mix', 'mix2'])
ap.add_argument('--no-render', action='store_true')
args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'mainline-v85'))
sys.path.insert(0, os.path.join(ROOT, 'mainline-v91'))

OL = 'http://127.0.0.1:11434'
LIB = 'qwen3.5:4b'
HF = os.path.join(ROOT, 'finetune-v68', 'hf', 'Qwen3.5-4B')
END = '<|im_end|>'
TCC = json.load(open(os.path.join(ROOT, 'mainline-v91', 'train-system-cardC.json'), encoding='utf-8'))
TCD = json.load(open(os.path.join(ROOT, 'mainline-v92', 'train-system-cardD.json'), encoding='utf-8'))
SYS_C, SYS_D = TCC['system'], TCD['system']
assert hashlib.sha256(SYS_C.encode('utf-8')).hexdigest() == TCC['sha256'], 'train-system-cardC sha256 對不上'
assert hashlib.sha256(SYS_D.encode('utf-8')).hexdigest() == TCD['sha256'], 'train-system-cardD sha256 對不上'
assert TCD['equals_offline_construction'] and '嘴' not in SYS_D
ACT = re.compile(r'<\|ACT\s*\{.*?\}\s*\|>')
MARK = re.compile(r'<\|[^|]*?\|>')
LAH = re.compile(r'[啦齁吼]')
GEN_TAIL = '<|im_start|>assistant\n<think>\n\n</think>\n\n'
TURN = re.compile(r'<\|im_start\|>(user|assistant)\n(.*?)<\|im_end\|>\n', re.S)
OUTD = os.path.join(HERE, 'data')
os.makedirs(OUTD, exist_ok=True)
os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)

from transformers import AutoTokenizer                                     # noqa: E402
tok = AutoTokenizer.from_pretrained(HF)


def post(path, body):
    r = urllib.request.Request(OL + path, data=json.dumps(body, ensure_ascii=False).encode('utf-8'), headers={'Content-Type': 'application/json; charset=utf-8'})
    return json.loads(urllib.request.urlopen(r, timeout=600).read())


def hf_render(msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def parse_prompt(p, sys_text):
    head = '<|im_start|>system\n' + sys_text + END + '\n'
    assert p.startswith(head) and p.endswith(GEN_TAIL), 'prompt 開頭／結尾不是預期的形狀'
    body = p[len(head):-len(GEN_TAIL)]
    msgs = [{'role': 'system', 'content': sys_text}]
    pos = 0
    for m in TURN.finditer(body):
        assert m.start() == pos, ('拆不乾淨', pos)
        role, content = m.group(1), m.group(2)
        msgs.append({'role': role, 'content': (' ' + content) if role == 'assistant' else content})
        pos = m.end()
    assert pos == len(body), '拆不乾淨（尾巴）'
    assert msgs[-1]['role'] == 'user'
    return msgs


def line_text(target):
    return MARK.sub('', target.replace(END, '')).strip()


def finish(recs, label):
    mism, lens, out = [], [], []
    t0 = time.time()
    for i, r in enumerate(recs):
        if not args.no_render:
            rend = post('/api/chat', {'model': LIB, 'messages': r['messages'], 'think': False, 'stream': False, '_debug_render_only': True})['_debug_info']['rendered_template']
            if rend != r['prompt']:
                j = next((k for k, (x, y) in enumerate(zip(rend, r['prompt'])) if x != y), min(len(rend), len(r['prompt'])))
                mism.append({'src': r['src'], 'seg': r['seg'], 'line': r['line'], 'at': j, 'ollama': rend[max(0, j - 40):j + 60], 'hf': r['prompt'][max(0, j - 40):j + 60]})
        n = len(tok(r['prompt'], add_special_tokens=False)['input_ids']) + len(tok(r['target'], add_special_tokens=False)['input_ids'])
        lens.append(n)
        out.append({'src': r['src'], 'seg': r['seg'], 'line': r['line'], 'prompt': r['prompt'], 'target': r['target'], 'tokens': n})
        if (i + 1) % 500 == 0:
            print('  %s %d/%d（%.0f s）' % (label, i + 1, len(recs), time.time() - t0), flush=True)
    L = sorted(lens) or [0]
    q = lambda p: L[min(len(L) - 1, int(p * len(L)))]
    st = {'n': len(out), 'render_equal': None if args.no_render else '%d/%d' % (len(out) - len(mism), len(out)), 'n_mismatch': len(mism), 'mismatch_examples': mism[:5],
          'tokens': {'min': L[0], 'median': q(.5), 'p90': q(.9), 'max': L[-1], 'sum': sum(L), 'mean': round(sum(L) / max(1, len(L)), 1)},
          'by_src': {s: sum(1 for x in out if x['src'] == s) for s in sorted({x['src'] for x in out})}, 'secs': round(time.time() - t0, 1)}
    return out, st


def write(name, rows):
    with io.open(os.path.join(OUTD, name), 'w', encoding='utf-8', newline='\n') as fo:
        for x in rows:
            fo.write(json.dumps(x, ensure_ascii=False) + '\n')


def stage_old():
    J = json.load(open(os.path.join(ROOT, 'mainline-v92', 'e', 'card', 'judged.json'), encoding='utf-8'))
    # 鍵含句子原文：第一包的小批與放大有同名段（同 src／seg／line、不同句子），只用三欄會撞號、多拿掉別的句子（23:55 實查：train 拿掉 56 > 清單 53）
    REMOVE = {(x['src'], x['seg'], x['line'], x['her']): x['why'] for x in J['remove_old']}
    assert len(REMOVE) == len(J['remove_old'])
    res = {'what': 'v92 舊資料（v91 轉好的第一、二、三包）：E-2 拿掉宣布嘴＋兩個新錯誤型＋system 換卡 D、重新渲染', 'at': time.strftime('%Y-%m-%d %H:%M:%S'),
           'system_d_sha256': TCD['sha256'], 'system_d_chars': len(SYS_D), 'sources': {}, 'removed_e2': {}, 'splits': {}}
    srcs = [os.path.join(ROOT, 'finetune-v91', 'data', 'qwen-old-%s.jsonl'), os.path.join(ROOT, 'finetune-v91', 'data', 'qwen-p3-%s.jsonl')]
    hit = set()
    for split in ('train', 'val'):
        recs, removed = [], []
        for pat in srcs:
            lines = io.open(pat % split, encoding='utf-8').read().splitlines()
            res['sources'][os.path.basename(pat % split)] = len(lines)
            for l in lines:
                r = json.loads(l)
                assert r['target'].endswith(END) and not LAH.search(line_text(r['target']))
                k = (r['src'], r['seg'], r['line'], line_text(r['target']))
                if k in REMOVE:
                    hit.add(k)
                    removed.append({'src': r['src'], 'seg': r['seg'], 'line': r['line'], 'why': REMOVE[k], 'text': line_text(r['target'])})
                    continue
                msgs = parse_prompt(r['prompt'], SYS_C)
                assert hf_render(msgs) == r['prompt'], ('🔴 卡 C 重組≠原 prompt', r['src'], r['seg'], r['line'])
                msgs[0] = {'role': 'system', 'content': SYS_D}
                recs.append({'src': r['src'], 'seg': r['seg'], 'line': r['line'], 'messages': msgs, 'prompt': hf_render(msgs), 'target': r['target']})
        res['removed_e2'][split] = {'n': len(removed), 'by_src': {s: sum(1 for x in removed if x['src'] == s) for s in ('p1', 'p2', 'p3')}, 'rows': removed}
        print('old %s：%d 筆（E-2 拿掉 %d）；卡 C 重組全部＝原 prompt' % (split, len(recs), len(removed)), flush=True)
        out, st = finish(recs, 'old-' + split)
        res['splits'][split] = st
        write('qwen-old-%s.jsonl' % split, out)
        print('old %s：渲染 %s、token 中位 %d、平均 %.0f、最長 %d（%.0f s）' % (split, st['render_equal'], st['tokens']['median'], st['tokens']['mean'], st['tokens']['max'], st['secs']), flush=True)
    assert hit == set(REMOVE), ('E-2 清單裡有對不到的筆', len(set(REMOVE) - hit))
    n_rm = sum(v['n'] for v in res['removed_e2'].values())
    assert n_rm == len(REMOVE), ('E-2 拿掉的筆數≠清單', n_rm, len(REMOVE))
    if not args.no_render:
        post('/api/generate', {'model': LIB, 'keep_alive': 0})
    json.dump(res, open(os.path.join(HERE, 'results', 'convert-v92-old.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return all(v['n_mismatch'] == 0 for v in res['splits'].values()) or args.no_render


def target_text(raw):
    m = ACT.match(raw.strip())
    assert m, raw
    return m.group(0) + ' ' + raw.strip()[m.end():].lstrip()


def history_text(raw):
    return ' ' + ACT.sub('', raw).strip()


def stage_p4():
    import e1_v91 as E1
    FIN = json.load(open(os.path.join(ROOT, 'mainline-v92', 'e', 'b3', 'final-v92.json'), encoding='utf-8'))
    segs = dict(E1.load_pack(os.path.join(ROOT, 'mainline-v92', '給你-DeepSeek-第四包', '回收'))[0])
    assert len(segs) == 216
    tgt = {(r['key'], r['line']): r for r in FIN['rows']}
    val_segs = {x['key'] for x in FIN['segments'] if x['val']}
    rng = random.Random(9203)
    recs = {'train': [], 'val': []}
    multi = 0
    for key, s in sorted(segs.items()):
        day, hh, mm = rng.randint(1, 28), rng.randint(7, 22), rng.randint(0, 50)
        t = hh * 60 + mm
        msgs = [{'role': 'system', 'content': SYS_D}]
        split = 'val' if key in val_segs else 'train'
        for i, tr in enumerate(s['turns']):
            if tr['who'] == '他':
                t += rng.randint(0, 2)
                msgs.append({'role': 'user', 'content': '[2026-10-%02d %02d:%02d] %s：%s' % (day, (t // 60) % 24, t % 60, USER_NAME, tr['text'])})
                continue
            r = tgt[(key, i + 1)]
            if r['target']:
                if len(ACT.findall(tr['raw'])) > 1:
                    multi += 1
                m = [dict(x) for x in msgs]
                recs[split].append({'src': 'p4', 'seg': key, 'line': i + 1, 'messages': m, 'prompt': hf_render(m), 'target': target_text(tr['raw']) + END})
            msgs.append({'role': 'assistant', 'content': history_text(tr['raw'])})
    n_tgt = {k: sum(1 for r in FIN['rows'] if r['target'] and (r['key'] in val_segs) == (k == 'val')) for k in ('train', 'val')}
    assert {k: len(v) for k, v in recs.items()} == n_tgt, ({k: len(v) for k, v in recs.items()}, n_tgt)
    res = {'what': 'v92 第四包轉格式', 'at': time.strftime('%Y-%m-%d %H:%M:%S'), 'system_d_sha256': TCD['sha256'], 'multi_marker_lines': multi, 'splits': {}}
    for split in ('train', 'val'):
        out, st = finish(recs[split], 'p4-' + split)
        res['splits'][split] = st
        write('qwen-p4-%s.jsonl' % split, out)
        print('p4 %s：%d 筆、渲染 %s、token 中位 %d、平均 %.0f、最長 %d' % (split, st['n'], st['render_equal'], st['tokens']['median'], st['tokens']['mean'], st['tokens']['max']), flush=True)
    json.dump(res, open(os.path.join(HERE, 'results', 'convert-v92-p4.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return all(v['n_mismatch'] == 0 for v in res['splits'].values()) or args.no_render


def stage_mix():
    res = {'what': 'v92 組 m1（F-1 v92-a＝E-2 的舊資料＋第四包）', 'at': time.strftime('%Y-%m-%d %H:%M:%S')}
    for part in ('old', 'p4'):
        r = json.load(open(os.path.join(HERE, 'results', 'convert-v92-%s.json' % part), encoding='utf-8'))
        assert all(v['n_mismatch'] == 0 for v in r['splits'].values()) and all(v['render_equal'] for v in r['splits'].values()), '🔴 %s 那一步沒有乾淨做完' % part
    old = {sp: io.open(os.path.join(OUTD, 'qwen-old-%s.jsonl' % sp), encoding='utf-8').read().splitlines() for sp in ('train', 'val')}
    new = {sp: io.open(os.path.join(OUTD, 'qwen-p4-%s.jsonl' % sp), encoding='utf-8').read().splitlines() for sp in ('train', 'val')}
    for name, lines in (('qwen-m1-train.jsonl', old['train'] + new['train']), ('qwen-m1-val.jsonl', old['val'] + new['val'])):
        with io.open(os.path.join(OUTD, name), 'w', encoding='utf-8', newline='\n') as fo:
            for l in lines:
                fo.write(l + '\n')
        res.setdefault('mixes', {})[name] = len(lines)
    print('組好：%s' % res['mixes'], flush=True)
    json.dump(res, open(os.path.join(HERE, 'results', 'convert-v92-mix.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return True


def stage_mix2():
    R = json.load(open(os.path.join(ROOT, 'mainline-v92', 'ruler', 'old-removed.json'), encoding='utf-8'))
    J = json.load(open(os.path.join(ROOT, 'mainline-v92', 'ruler', 'old-judged.json'), encoding='utf-8'))
    REMOVE = {(x['file'], x['idx']): x['key'] for x in R['removed']}
    assert len(REMOVE) == len(R['removed']) == R['n']
    res = {'what': 'v92 組 m2（F-2 v92-b＝E-3 之後的舊資料＋第四包）', 'at': time.strftime('%Y-%m-%d %H:%M:%S'),
           'e3_judged': J['n_judged'], 'e3_items': J['n_items'], 'e3_chunks': J['chunks'], 'removed_e3': {}, 'mixes': {}}
    for part in ('old', 'p4'):
        r = json.load(open(os.path.join(HERE, 'results', 'convert-v92-%s.json' % part), encoding='utf-8'))
        assert all(v['n_mismatch'] == 0 for v in r['splits'].values()) and all(v['render_equal'] for v in r['splits'].values()), '🔴 %s 那一步沒有乾淨做完' % part
    hit = set()
    for sp in ('train', 'val'):
        old = io.open(os.path.join(OUTD, 'qwen-old-%s.jsonl' % sp), encoding='utf-8').read().splitlines()
        new = io.open(os.path.join(OUTD, 'qwen-p4-%s.jsonl' % sp), encoding='utf-8').read().splitlines()
        keep, gone = [], []
        for idx, l in enumerate(old):
            r = json.loads(l)
            k = '%s|%s|%d|%s' % (r['src'], r['seg'], r['line'], line_text(r['target']))
            if (sp, idx) in REMOVE:
                assert REMOVE[(sp, idx)] == k, ('🔴 清單的 file／idx 跟句子對不上', sp, idx)
                hit.add((sp, idx))
                gone.append(r['src'])
                continue
            keep.append(l)
        name = 'qwen-m2-%s.jsonl' % sp
        with io.open(os.path.join(OUTD, name), 'w', encoding='utf-8', newline='\n') as fo:
            for l in keep + new:
                fo.write(l + '\n')
        res['removed_e3'][sp] = {'n': len(gone), 'by_src': {s: gone.count(s) for s in ('p1', 'p2', 'p3')}}
        res['mixes'][name] = {'old_kept': len(keep), 'p4': len(new), 'total': len(keep) + len(new)}
    assert hit == set(REMOVE), ('E-3 清單裡有對不到的筆', len(set(REMOVE) - hit))
    assert sum(v['n'] for v in res['removed_e3'].values()) == R['n']
    print('組好：%s；E-3 拿掉 %s' % (res['mixes'], res['removed_e3']), flush=True)
    json.dump(res, open(os.path.join(HERE, 'results', 'convert-v92-mix2.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return True


ok = {'old': stage_old, 'p4': stage_p4, 'mix': stage_mix, 'mix2': stage_mix2}[args.stage]()
print('OK' if ok else '🔴 渲染不一致')
sys.exit(0 if ok else 1)
