"""OpenAI-compatible pass-through in front of Ollama that can actually turn qwen3's
reasoning off. Port 11435.

Why this exists
---------------
AIRI's Ollama provider has a "思考模式" setting, and it does send `"think": false`
in the request body — that part was verified from the real payload. But AIRI's
Base URL ends in `/v1/`, so the request lands on Ollama's OpenAI-compatible
endpoint, and **that endpoint ignores `think` entirely**. Measured on
`/v1/chat/completions`, `think:false` / `think:true` / no-think-field all reason
for ~300 characters and all take ~1.0s to the first visible token.

The one switch that does work there is OpenAI's own `reasoning_effort: "none"`
(0.150s to first token, 0 reasoning characters). AIRI has no UI for extra body
fields — its advanced settings only expose HTTP headers — so the field has to be
injected somewhere between AIRI and Ollama. That is all this process does.

Same shape as the TTS adapter: AIRI thinks it is talking to Ollama, and one hop
in the middle adds a field. No AIRI source change; the only thing the user
touches is the Base URL in settings.

Streaming
---------
The body is forwarded chunk by chunk and the response is streamed straight back.
This matters more than it looks: AIRI's first TTS request fires on the first
punctuation in the token stream, so a proxy that buffered the response would
convert a 0.15s first token into a full-response wait and make latency *worse*
than doing nothing. `httpx.AsyncClient.stream` plus `StreamingResponse` keeps it
incremental end to end.

Settings
--------
  LLM_PROXY_PORT       default 11435
  LLM_PROXY_UPSTREAM   default http://localhost:11434
  LLM_PROXY_THINKING   "off" (default) injects reasoning_effort=none;
                       "on" forwards untouched, i.e. the model's own default

`GET/POST /config` flips thinking at runtime, so both sides can be compared
inside AIRI without a restart.

Timing log
----------
`timing.jsonl` records four wall-clock instants per chat request — received,
forwarded upstream, first upstream byte, response finished — so the proxy's own
overhead can be separated from Ollama's work. Instants are anchored once per
request and derived from `perf_counter` deltas, because `time.time()` on Windows
only ticks every ~15.6ms.

Call stack:

  POST /v1/chat/completions
    -> {@link chat_completions}
      -> {@link _forward_stream}
        -> Ollama /v1/chat/completions
"""

import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from marker_fix import SseMarkerRewriter   # v73b task 3

HERE = Path(__file__).parent
UPSTREAM = os.environ.get("LLM_PROXY_UPSTREAM", "http://localhost:11434").rstrip("/")
LOG_PATH = Path(os.environ.get("LLM_PROXY_LOG", str(HERE / "timing.jsonl")))

# v17: `t_first_byte` alone could not answer "where did 0.86s go", because the
# first SSE frame is not necessarily the first thing the user can read. The two
# extra instants below split it: `t_first_visible` is the first frame carrying
# non-empty `delta.content`, `t_first_reasoning` the first carrying reasoning.
# If they differ, the gap is decode of invisible tokens; if they coincide, the
# time is upstream of generation. Frames are inspected *after* being yielded and
# only until the first visible one, so the streamed path is not slowed down.
CAPTURE_DIR = os.environ.get("LLM_PROXY_CAPTURE_DIR", "").strip()

app = FastAPI(title="Ollama reasoning proxy")

# AIRI calls this from the browser, so it is a cross-origin XHR like the TTS adapter.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_settings = {
    "thinking": os.environ.get("LLM_PROXY_THINKING", "off"),
    # AIRI attaches four tool definitions to every request and offers no way to
    # turn them off — they are registered at runtime by `llm-tools.ts`, not by a
    # setting. Measured on the real path they cost 1260 prompt tokens, 31% of a
    # 4096 context. Two of them (`builtIn_mcpListTools`, `builtIn_mcpCallTool`)
    # can only ever return an empty list while no MCP server is configured, and
    # a third is `builtIn_debugRandomNumber`.
    #
    # v53 (planner ruling 2026-09-09): ON by default, and the number above is stale --
    # re-measured on qwen3.5:4b the 7 definitions are 2622 of the 4952 tokens production
    # was sending (53%), `builtIn_emitSparkCommand` alone 1505. The old "off by default,
    # it changes what the model can do" reasoning was overturned by measuring it: six
    # 20-turn runs per arm on the real desktop path gave 38/120 third-person turns and 5
    # invented tool calls with the tools forwarded, 5/120 and 0 with them stripped, at the
    # same per-turn latency (they live in the cached prefix). Nothing can use them today:
    # no MCP server is configured and no frontend module is connected.
    #   🔴 Recorded with the ruling: production carried that defect for every round up to
    #   v52 and nobody knew, because every v50-v52 measurement ran with strip_tools=on --
    #   the arm we measured was never the arm we shipped. A silent production defect.
    #   ⚠️ Deferred cost: this is `body.pop("tools")`, all-or-nothing (v33 read the code),
    #   so when a real module connects, per-tool filtering has to exist first --
    #   `builtIn_emitSparkCommand` is the Minecraft line's channel. ENVIRONMENT §10 item 20.
    # `_note_tools` runs before the strip, so startup check 15 still sees the real list.
    "strip_tools": os.environ.get("LLM_PROXY_STRIP_TOOLS", "on"),
    # v24 E3: lift the few-shot block out of the system prompt and replay it as
    # alternating user/assistant messages. AIRI has no path that sends
    # mes_example as messages (messageExample is parsed on import and then
    # never used), so the conversion happens here.
    #
    # v57 (planner ruling 2026-09-09): ON by default. It was off, and that meant
    # the v30 W2 decision -- "few-shot travels as messages+boundary", taken
    # 2026-08-22 and recorded in ENVIRONMENT §9 as 現行基準 -- had never once
    # run in production. `fewshot_mode` only does anything when this switch is
    # on; with it off the block stays untagged inside the system prompt, which
    # is the pre-v24 arrangement. Every measurement from v50 to v54 ran with it
    # on. The arm we measured was not the arm we shipped, for the second time
    # (v53 was the tools).
    #   What the flip buys, measured single-variable on the real desktop path in
    #   v56 (arm F, 20 turns): the leading <|ACT ...|> marker drops from 51
    #   characters to 27 and from 212 ms to 103, ACT coverage goes from 23/40
    #   turns to 20/20, replies lost entirely go from 2/40 to 0/20, end to end
    #   2557.5 -> 2391.1 ms. The ACT coverage is the point -- that is acceptance
    #   condition 3 (expressions), not a latency tweak.
    #   ⚠️ The cost is that she sees the eight demo pairs as sixteen real
    #   conversation messages. On 4B that has never produced the v29 symptom
    #   (Gemma/Qwen3.5/27B all answered as if the examples had really happened:
    #   「你又問一次」); if the base model changes, re-check that first.
    #   🔴 And it makes 情境測試清單 contamination worse rather than better,
    #   which is why v57 fixed the list before flipping this.
    "fewshot_messages": os.environ.get("LLM_PROXY_FEWSHOT_MESSAGES", "on"),
    # v30 W2: how the lifted pairs are transported when fewshot_messages=on.
    #   messages          -- v24 E3: 16 alternating turns (stronger models read
    #                        them as real history: 「你又問一次」)
    #   messages+boundary -- same turns + a closing system message
    #                        ← DEFAULT since v30 close (planner decision
    #                        2026-08-22): hallucinated references 0 on both
    #                        protocols; the 4q verbatim-copy number measures
    #                        the same confusion from the other side because
    #                        3 of the 4 questions ARE example user lines, and
    #                        the 60 script turns copied 0.
    #   single_user       -- all pairs in one labelled user message + 「好。」
    #   system_tagged     -- v30 research rank 1: block stays in the system
    #                        prompt, pairs wrapped in <example> tags + scaffold
    #   system_lines      -- v30 research rank 2 (SillyTavern layout): each
    #                        example line as its own system message + boundaries
    #                        (rejected in v30: she starts calling the user User)
    #   messages+boundary_user -- v64 E-2: the boundary as a USER message (the
    #                        default's layout is outside Qwen3.5's official
    #                        format: a system message that is not first)
    #   messages+boundary_head -- v64 E-2: the boundary folded into the first
    #                        system message, reworded; nothing mid-conversation
    #                        Both measured in v64, neither is the default.
    #   drop              -- v81 task E: lift the demo block out of the system
    #                        prompt and insert NOTHING (no turns, no boundary).
    #                        This is the only transport that produces, on the
    #                        real desktop path, the shape the pilot was trained
    #                        on (system[0] == training string, then the real
    #                        turns). fewshot_messages=off does NOT do that: it
    #                        leaves the block inside the system text (v80 §8).
    #                        Measured only; the default stays messages+boundary.
    # (fewshot_messages=off leaves the untagged block in the system prompt.)
    "fewshot_mode": os.environ.get("LLM_PROXY_FEWSHOT_MODE", "messages+boundary"),
    # v25: inject explicit sampling params. AIRI sends none, so the Modelfile
    # values apply -- and qwen3:1.7b ships thinking-mode values (0.6/0.95)
    # while this project runs it in non-thinking mode. "on" injects the
    # official non-thinking recommendation below. Only temperature and top_p:
    # top_k/min_p are ignored by Ollama's /v1 endpoint (verified empirically,
    # temp=2+top_k=1 still diverges) and both already sit at the official
    # values via the Modelfile (top_k 20) and default (min_p 0).
    "sampling": os.environ.get("LLM_PROXY_SAMPLING", "off"),
    "sampling_temperature": float(os.environ.get("LLM_PROXY_SAMPLING_TEMPERATURE", "0.7")),
    "sampling_top_p": float(os.environ.get("LLM_PROXY_SAMPLING_TOP_P", "0.8")),
    # v25: rewrite the model field so experiments can switch models without
    # touching AIRI's settings page (automation must not touch settings).
    # Empty string = off.
    "model_override": os.environ.get("LLM_PROXY_MODEL_OVERRIDE", ""),
    # v26 D1: strip the two non-card injections AIRI wraps around the card in
    # the system message -- 260 chars of markdown/LaTeX directives prepended
    # by session-store.ts and ~441 chars of artistry widget instructions
    # appended from the card-extensions default. Removing them at the source
    # would mean editing AIRI (spec forbids without report), so the proxy
    # peels them off the wire instead. Exact-match only: anything that does
    # not match byte-for-byte is left alone rather than guessed at.
    "strip_injections": os.environ.get("LLM_PROXY_STRIP_INJECTIONS", "off"),
    # v73 task D: the SAME suffix as strip_injections, on its own switch -- the
    # ~441 chars of artistry/image_journal/canvas widget instructions, WITHOUT
    # the 260-char markdown/LaTeX prefix that strip_injections also takes. The
    # customer's standing authorisation (00 v1.6 §一, 2026-09-21) is to turn off
    # add-on features that get in the way of chatting; the drawing instructions
    # are one, the markdown directives are not, and turning both off at once
    # would make the measurement the sum of two changes (原則三). Switch only:
    # no code path is deleted, `off` is byte-identical to v72 production.
    "strip_widget": os.environ.get("LLM_PROXY_STRIP_WIDGET", "off"),
    # v73b task 3: repair or strip the control markers the model writes malformed,
    # in-stream, so AIRI's reasoning filter cannot drop the rest of the reply.
    # See marker_fix.py for the three rules and why "repair" never guesses an
    # emotion. `off` restores v73's byte-identical behaviour.
    "fix_markers": os.environ.get("LLM_PROXY_FIX_MARKERS", "on"),
    # v61 task B: what happens to the card's 【輸出格式規則】…【輸出格式規則結束】
    # block (1201 chars in card-v0.3 -- the largest single block in the prompt,
    # and the one block no round ever measured on its own). It is technical
    # control (ACT/DELAY/CALL protocol, emotion list), not personality, but the
    # card is the only place AIRI lets it ride, so this proxy is the seam that
    # can pull the two apart.
    #   card      -- leave it where the card put it (production; byte-identical)
    #   short     -- replace it with _FORMAT_SHORT: the wire protocol the lifted
    #                demos already show (<|ACT {"emotion":"x"}|> then speech) and
    #                the nine emotion names, nothing else (~170 chars; drops
    #                DELAY/CALL/intensity/motion and the structural example)
    #   separate  -- lift the block out of the card text and send it, unchanged,
    #                as its own system message right after the few-shot boundary:
    #                still inside the stable cached prefix, but the last thing
    #                before the real turns
    #   off       -- remove the block; the protocol then survives only through
    #                the ACT prefix on the lifted demo turns
    # The factory line after the block (「最後，做任何你想做的事！」) stays where it
    # is in every mode (v59: deleting it changed nothing). No-op on a card
    # without the two headers (v0.1 keeps its machinery in `description`
    # unlabelled) -- same rule as the few-shot lift. Measured, not shipped:
    # the default is `card` until the planner rules on the v61 numbers.
    "format_rules": os.environ.get("LLM_PROXY_FORMAT_RULES", "card"),
    # v27 FIX: AIRI strips <|ACT ...|> markers before storing a reply, so the
    # history the model sees never contains its own markers -- and it imitates
    # that history (v20b 6/6, v26 M1 20/20: ACT gone within a session). This
    # proxy sees the raw stream, so it remembers each reply's raw text and,
    # on the next request, swaps the stripped assistant messages back to the
    # raw form before forwarding. UI and TTS never see the markers (they read
    # AIRI's stored copy); only the model does. Off by default.
    "restore_markers": os.environ.get("LLM_PROXY_RESTORE_MARKERS", "off"),
    # v51: who the user IS. The card describes the relationship ("你直接叫他User")
    # but nothing in the payload says the person typing is that User, and the
    # 4B reads him as a third party. Two seams, both plain parameters so the
    # same proxy serves a friend or a voiceprint-identified speaker later:
    #   user_persona      text appended to the system prompt as its own paragraph
    #                     (SillyTavern "persona description" position); "" = off
    #   user_name_prefix  string put in front of every REAL user message, after
    #                     AIRI's timestamp (SillyTavern "names in messages"); "" = off.
    #                     Few-shot example turns are never prefixed.
    "user_persona": os.environ.get("LLM_PROXY_USER_PERSONA", ""),
    "user_name_prefix": os.environ.get("LLM_PROXY_USER_NAME_PREFIX", ""),
    # v52: the two upstream identity signals v51 did not have (spec v52 §五; the
    # inventory of the four is in yisen-v51/upstream-identity-v51.md §一). All
    # off by default so the v51 "P3" payload stays byte-identical.
    #   user_frame             one sentence naming BOTH parties, placed at the very
    #                          top of the system prompt -- SillyTavern's default
    #                          main prompt「a chat between {{char}} and {{user}}」
    #                          position; "" = off
    #   user_persona_position  "tail" (v51: after the card text) | "head" (before
    #                          the card text, right after user_frame -- ST's
    #                          prompt-manager default order main → persona → card)
    #   fewshot_user_prefix    "on": user_name_prefix goes in front of the few-shot
    #                          EXAMPLE user lines too, so the examples carry the
    #                          same speaker label as the real history (CCV1
    #                          `{{user}}: …` convention; ST #1860: examples labelled
    #                          differently from history confuse the model). v51
    #                          deliberately never labelled the examples.
    #   fewshot_prefix_skip_named
    #                          "on": an example user line that already CONTAINS the
    #                          user's name (derived from user_name_prefix by dropping
    #                          the trailing ：/:) is left unlabelled. Card v0.3 pair 4
    #                         「User有沒有跟你講過誰的壞話」is a friend asking about
    #                          User, and「User：User有沒有…」would read as him asking
    #                          about himself. The card is not edited (00 §四).
    #                          Bypass risks, stated up front (00 §五): the rule sees
    #                          the NAME, not the SPEAKER. A line User says that
    #                          mentions his own name is wrongly skipped; a line
    #                          someone else says that does not name him is wrongly
    #                          labelled as his; a name that is a substring of another
    #                          word triggers it; and a prefix that is not "name +
    #                          colon" yields the wrong name (/config prints the
    #                          derived name so that is visible, not silent).
    "user_frame": os.environ.get("LLM_PROXY_USER_FRAME", ""),
    "user_persona_position": os.environ.get("LLM_PROXY_USER_PERSONA_POSITION", "tail"),
    "fewshot_user_prefix": os.environ.get("LLM_PROXY_FEWSHOT_USER_PREFIX", "off"),
    "fewshot_prefix_skip_named": os.environ.get("LLM_PROXY_FEWSHOT_PREFIX_SKIP_NAMED", "on"),
    # v54 task B: the anchor slot. Every 20-turn session she picks one content
    # word and orbits it (v53 T0: User x14/x7/x11; T1: 笨蛋 / 傢伙 / User). v51/v52
    # showed the simple story is wrong -- the arm with the MOST name exposure had
    # the LEAST orbiting, and removing the few-shot block did not remove the
    # snowball, it only changed what she copied. So the slot looks like something
    # that gets filled either way, and the question this round MEASURES (it does
    # not implement anything) is whether the filling can be STEERED.
    #   anchor_seed            one word/phrase offered as a candidate anchor;
    #                          "" = off (default, so the payload stays byte-identical
    #                          to the v53 T1 arm this is compared against)
    #   anchor_seed_position   "system": the sentence becomes its own paragraph at
    #                          the end of the system prompt (resident, inside the
    #                          cached prefix, same slot style as user_persona)
    #                          "tail":  the sentence is its own system message placed
    #                          immediately before the newest user turn (per-turn, at
    #                          the recency end, outside the stable prefix)
    #   anchor_seed_template   the wording. A setting, not a constant (00 §六 3):
    #                          「{seed}」 is replaced by anchor_seed. Deliberately
    #                          phrased as a fact about what is around her, NOT as an
    #                          instruction to talk about it -- an instruction would be
    #                          a situational behaviour rule, which 00 §四 3 keeps out
    #                          of the character layer, and it would also measure
    #                          obedience instead of measuring the slot.
    "anchor_seed": os.environ.get("LLM_PROXY_ANCHOR_SEED", ""),
    "anchor_seed_position": os.environ.get("LLM_PROXY_ANCHOR_SEED_POSITION", "system"),
    "anchor_seed_template": os.environ.get("LLM_PROXY_ANCHOR_SEED_TEMPLATE", "你今天一直注意到「{seed}」。"),
    # v83 task B: the HISTORY LAYER -- how she sees her own earlier turns. Multi-turn
    # replays (v81/v82) showed the snowball: once a conversation copies one of her own
    # earlier lines it tends to keep copying to the end (pilot:v82 111/234 replies,
    # current 23/234). The customer's hypothesis: she cannot tell "things said earlier"
    # from "examples of how to answer", because her own lines sit in the assistant slot.
    # This seam re-renders ONLY the real conversation history AIRI sends (never the
    # lifted style demos, never the newest user turn, which always goes through verbatim)
    # according to a profile:
    #   off           -- untouched; byte-identical to v82 (the default)
    #   record        -- every earlier turn moves, verbatim, into ONE "前情紀錄" message in
    #                    the record role (her lines labelled): position changes, words do not
    #   gist          -- same record, but her lines become one-line gists written by
    #                    history_gist_model: position AND literal wording change
    #   window2       -- the last 2 exchanges stay native; older: his lines go to the
    #                    record verbatim, hers are dropped: the amount of her text changes
    #   recent1_gist  -- the last exchange stays native; older: his verbatim, hers as gists
    #   custom        -- the four structural keys below, as set
    # 🔴 Principle 10: a gist that fails, times out or comes back empty is LOUD (log,
    # /health counters, history-events.jsonl, the capture sidecar) and that one line falls
    # back to its verbatim text -- never silently the whole raw history, never silently
    # no history. A request header `X-History-Profile: <name>` overrides the global value
    # for that request (00 §八 seam 3: the shim accepts a profile). The seam is generic on
    # purpose: a future memory layer can hand its notes to the same record block.
    "history_profile": os.environ.get("LLM_PROXY_HISTORY_PROFILE", "off"),
    "history_keep_last": int(os.environ.get("LLM_PROXY_HISTORY_KEEP_LAST", "0")),
    "history_her_older": os.environ.get("LLM_PROXY_HISTORY_HER_OLDER", "verbatim"),
    "history_record_role": os.environ.get("LLM_PROXY_HISTORY_RECORD_ROLE", "user"),
    # Wording is a setting, not a constant (00 §六 3). Every string here can be copied by
    # her (v81: with nothing else to hold on to she copies the most concrete thing in the
    # prompt), so replays count leaks of these strings like any other card text.
    "history_record_head": os.environ.get("LLM_PROXY_HISTORY_RECORD_HEAD", "（{user}跟你這一場前面講過的話，照順序：）"),
    # Tail line: EMPTY by default (v83 B-0). presence_penalty in this backend is applied over
    # the last 64 tokens INCLUDING the prompt (llama-server feeds the prompt to the sampler),
    # so with history untouched the tail of her previous reply sits inside that window. A
    # closing line after the record would push it out -- a difference unrelated to the
    # position question. Empty = no closing line; her last recorded line ends the record.
    "history_record_tail": os.environ.get("LLM_PROXY_HISTORY_RECORD_TAIL", ""),
    "history_her_label": os.environ.get("LLM_PROXY_HISTORY_HER_LABEL", "你："),
    "history_her_gist_label": os.environ.get("LLM_PROXY_HISTORY_HER_GIST_LABEL", "你（大意）："),
    "history_gist_model": os.environ.get("LLM_PROXY_HISTORY_GIST_MODEL", "qwen3.5:4b"),
    "history_gist_timeout_s": float(os.environ.get("LLM_PROXY_HISTORY_GIST_TIMEOUT_S", "20")),
    "history_gist_prompt": os.environ.get("LLM_PROXY_HISTORY_GIST_PROMPT",
                                          "你在幫一段聊天做紀錄。下面是{user}說的一句話，和 Kamie 回他的一句話。"
                                          "用一句不帶主詞的短句，寫出 Kamie 這一句說了什麼、是什麼態度，不超過 30 個字。"
                                          "不要引用她的原句，不要沿用她句子裡的詞組，不要加評論，不要加引號。"
                                          "只輸出那一句，用繁體中文。"),
    # v84 spec B-1 #4: a ceiling on how long ONE reply may run. v83 saw pilot:v82 run
    # away to 7,199-7,227 tokens ("幹嘛幹嘛…", finish_reason=length), fill the 8192
    # context, and truncate every reply after it; nothing stopped it and nothing said so.
    #   on  = inject `max_tokens` (Ollama /v1 -> num_predict) on CHAT requests only, i.e.
    #         when the forwarded body carries reasoning_effort "none". A thinking-on
    #         request (the Minecraft planning line, or thinking=on) is left untouched, and
    #         a caller-supplied max_tokens / max_completion_tokens always wins.
    #   A reply that stops on the ceiling (finish_reason "length") is LOUD: printed,
    #   counted in /health (reply_cap_hits), and written to reply-cap-events.jsonl.
    #   Any other finish_reason "length" (context full, caller's own limit) is counted
    #   too (length_finishes) -- that was the silent half of the v83 runaway.
    # Resource-only by construction: 512 sits 1.7x above the longest reply the shipped model
    # has ever produced (v84 recount over v81-v83: 7,177 qwen3.5:4b replies, max 298 tokens,
    # p99.9 163; none reached 299), so a normal reply never touches it.
    # `off` restores the byte-identical v83 request.
    "reply_cap": os.environ.get("LLM_PROXY_REPLY_CAP", "on"),
    "reply_cap_tokens": int(os.environ.get("LLM_PROXY_REPLY_CAP_TOKENS", "512")),
    # v84 spec E: blind-chat latency equalisation. A fixed wait (ms) before a CHAT request
    # (reasoning_effort "none") is forwarded, so a faster arm can be made as slow as the
    # slowest one in a blind session -- otherwise the customer can tell the segments apart
    # by speed. Time only, never content. 0 (default) = no wait = byte-identical behaviour.
    # Set per blind segment by switch_her_v84.ps1; start_all restarts the proxy at 0.
    "pad_ms": int(os.environ.get("LLM_PROXY_PAD_MS", "0")),
    # v89 B-2: where a THINKING-ON request goes while `model_override` points at a model that
    # cannot think. The v87 fine-tuned models were trained on thinking-off turns only and return
    # EMPTY content when reasoning is on (ENVIRONMENT §8 v87) -- so with one of them shipped as the
    # default, the per-request thinking seam (00 §八: the Minecraft planning line needs thinking
    # on) would break silently. Non-empty = such a request is sent to this model instead (that
    # line keeps today's behaviour: the original qwen3.5:4b), counted (/health think_routed) and
    # written to think-route-events.jsonl, and printed. "" (default) = off = byte-identical v88.
    # Chat requests (reasoning_effort "none") are never touched. Set from the default-her profile
    # (character-cards\default-card.json, think_model) by start_proxy.ps1.
    "think_model": os.environ.get("LLM_PROXY_THINK_MODEL", ""),
}

# v83: the structural presets of the history layer (profile name -> keys). "custom" reads
# the history_* keys above. Kept as presets rather than code paths so a new arm (or the
# memory layer) is a new row, not a new branch.
_HISTORY_PROFILES = {
    "off": None,
    "record": {"keep_last": 0, "her_older": "verbatim", "record_role": "user"},
    "gist": {"keep_last": 0, "her_older": "gist", "record_role": "user"},
    "window2": {"keep_last": 2, "her_older": "drop", "record_role": "user"},
    "recent1_gist": {"keep_last": 1, "her_older": "gist", "record_role": "user"},
    # v83 B-0 (Huang et al. 2026 "SUM"): her earlier replies become gists but STAY in her
    # own (assistant) slot, no record message -- with `record` and `gist` this completes
    # a 2x2 of position x literal wording.
    "gist_inplace": {"keep_last": 0, "her_older": "gist", "record_role": "inplace"},
    "custom": "custom",
}
_HISTORY_EVENTS_PATH = Path(os.environ.get("LLM_PROXY_HISTORY_EVENTS", str(HERE / "history-events.jsonl")))
_HISTORY_GIST_MEMORY_PATH = Path(os.environ.get("LLM_PROXY_HISTORY_GIST_MEMORY", str(HERE / "history-gist-memory.jsonl")))
_history_counters = {"history_requests": 0, "history_transformed": 0, "history_gist_ok": 0, "history_gist_retried": 0,
                     "history_gist_failed": 0, "history_gist_fallback_lines": 0, "history_bad_profile": 0}
_gist_memory: dict[str, dict] = {}
_gist_tasks: dict = {}

# Raw-reply memory for restore_markers: normalized stripped text -> raw text.
# Persisted to a JSONL next to the proxy so a restart does not forget the
# session in progress.
_MARKER_MEMORY_PATH = Path(os.environ.get("LLM_PROXY_MARKER_MEMORY", str(HERE / "marker-memory.jsonl")))
_marker_memory: dict[str, str] = {}
_MARKER_RE = re.compile(r"<\|[^|]*\|>")

# Principle 10 (v27): an ACT collapse must be loud. Ring buffer of the last N
# replies' has-ACT flags; /health exposes the rate and the log shouts when it
# falls through the floor with enough samples to mean something.
_ACT_WINDOW = 20
_ACT_MIN_SAMPLES = 10
_ACT_FLOOR = 0.5
_act_recent: list[bool] = []

# Principle 10 (v33): the tool list decides what she can actually know, and it is
# the whole premise of 通則 11 ("don't make up what you don't know"). Today
# `get_weather` already means weather is not a safe "she can't know that" test
# topic, and `builtIn_mcpCallTool` means anyone connecting an MCP server widens
# that surface *without touching any character document*. So the proxy remembers
# the tool names it sees, and the startup check fails when they change — the
# character docs get revisited deliberately rather than quietly going stale.
_TOOLS_BASELINE_PATH = Path(os.environ.get("LLM_PROXY_TOOLS_BASELINE", str(HERE / "tools-baseline.json")))
_tools_seen: list[str] = []
_tools_baseline: list[str] = []


def _load_tools_baseline() -> None:
    global _tools_baseline
    try:
        _tools_baseline = sorted(json.loads(_TOOLS_BASELINE_PATH.read_text(encoding="utf-8"))["tools"])
        print(f"[proxy] tool baseline: {len(_tools_baseline)} tools from {_TOOLS_BASELINE_PATH.name}", flush=True)
    except Exception:
        _tools_baseline = []


def _note_tools(body: dict) -> None:
    """Record the tool names in this request; set the baseline on first sight."""
    global _tools_seen, _tools_baseline
    tools = body.get("tools")
    if not isinstance(tools, list) or not tools:
        return
    names = sorted(
        str((t.get("function") or t).get("name"))
        for t in tools
        if isinstance(t, dict)
    )
    if names == _tools_seen:
        return
    _tools_seen = names
    if not _tools_baseline and _TOOLS_BASELINE_PATH.exists():
        # v51: the baseline is normally loaded by the uvicorn startup hook. An
        # offline test that imports this module never runs that hook, and the
        # first request it rewrote then OVERWROTE the v33 baseline file (same 7
        # tools, but the note explaining why it exists was lost). Load lazily
        # instead of treating "not loaded yet" as "no baseline exists".
        _load_tools_baseline()
    if not _tools_baseline:
        _tools_baseline = list(names)
        try:
            _TOOLS_BASELINE_PATH.write_text(
                json.dumps({"tools": names, "noted_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                           ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[proxy] tool baseline set: {len(names)} tools -> {_TOOLS_BASELINE_PATH.name}", flush=True)
        except Exception as e:
            print(f"[proxy] could not write tool baseline: {e}", flush=True)
        return
    added = [n for n in names if n not in _tools_baseline]
    removed = [n for n in _tools_baseline if n not in names]
    if added or removed:
        print(f"⚠ TOOL LIST CHANGED — added={added} removed={removed}. "
              f"她能知道的事變了：檢查 人格判準 通則 11 與 情境測試清單 二之二 的題材地雷表", flush=True)


def _norm_for_match(text: str) -> str:
    """Key used to match AIRI's stored (stripped) reply back to the raw one.
    AIRI's parser emits the literal text between markers; whitespace around
    markers can differ, so compare with markers and all whitespace removed."""
    return "".join(_MARKER_RE.sub("", text).split())


def _remember_raw_reply(raw: str):
    if not raw or "<|" not in raw:
        return
    key = _norm_for_match(raw)
    if not key or _marker_memory.get(key) == raw:
        return
    _marker_memory[key] = raw
    try:
        with _MARKER_MEMORY_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "raw": raw}, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[proxy] marker memory write failed: {e}", flush=True)


def _load_marker_memory():
    if not _MARKER_MEMORY_PATH.exists():
        return
    try:
        with _MARKER_MEMORY_PATH.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    _marker_memory[rec["key"]] = rec["raw"]
                except Exception:
                    continue
        print(f"[proxy] marker memory loaded: {len(_marker_memory)} replies", flush=True)
    except Exception as e:
        print(f"[proxy] marker memory load failed: {e}", flush=True)


def _restore_markers(body: dict) -> int:
    """Swap stripped assistant messages back to their remembered raw form.
    Returns how many messages were restored. Messages that already carry a
    marker (e.g. the proxy's own few-shot examples) are left alone."""
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return 0
    restored = 0
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        content = m.get("content")
        if not isinstance(content, str) or "<|" in content:
            continue
        raw = _marker_memory.get(_norm_for_match(content))
        if raw:
            msgs[i] = {**m, "content": raw}
            restored += 1
    return restored


# v52 task C (principle 10): naming the user has two known side effects, and
# both would otherwise sit silently inside the output and look like the fix
# working:
#   1. she starts prefixing HER OWN replies with a name (SillyTavern #557)
#   2. she reads the「User：」prefix as the user calling HER User
#      (v51 probe, P3 r1:「你叫我User？那User就在這裡」)
# Every chat reply is checked after it has been fully streamed (never in front
# of the consumer); a hit is shouted in the log, counted in /health, and
# appended with its text to identity-events.jsonl so the round's report can
# list them separately from the CORRECT/WRONG counts. This is a sieve, not a
# judge (00 §五): the self-prefix rule fires on any ≤3-char token + colon at the
# start of a reply (v51's offline analysis used the same shape; 「結論就是：」 is
# 4 chars and does not fire), the misread rule on a fixed pattern list built
# from the user's name (「我是User創造的」 is excluded by lookahead). Both are
# meant to be read by a person afterwards, not trusted as counts.
_IDENTITY_EVENTS_PATH = Path(os.environ.get("LLM_PROXY_IDENTITY_EVENTS", str(HERE / "identity-events.jsonl")))
_identity = {"watched": 0, "self_prefix": 0, "misread_suspects": 0, "self_suspects": 0}
# v53 task D-1: the v52 failure shape "she answers as if SHE were User" carries no name, so
# the misread sieve above cannot see it. What it did have in every v52 probe hit: the USER's
# line named the user (「User在哪」) and the reply opened in the first person about her own
# whereabouts (「我剛不是在你面前嗎」「我剛才不是還在這裡嗎」「我剛在找你」). So the check is
# trigger-conditioned: only when the last user message contains the user's name, and only on
# that opening shape. Calibration (v52 probe, 48 rows): 8/8 SELF hits, 1 UNCLEAR, 0 false
# positives. Bypasses, stated up front: a SELF reading that does not open with 我剛/我不是…
# escapes; a SELF reading when the user's line does NOT name him is never checked; a normal
# first-person opening after a line that happens to mention his name is a false positive.
# v53 probe (24 rows) re-calibration: 「我在哪，你應該很清楚」「我在你心裡呀」escaped the first list,
# so a plain 「我在」 opening is included -- it only ever fires when the user's line named him.
_SELF_SUSPECT_RE = re.compile(r"^(?:我剛剛|我剛才|我剛|我不是|我就在|我在|我還在|我明明)")
# 1-3 CJK characters (「User：」fires,「結論就是：」does not) or 1-8 Latin letters
# (「AIRI:」「Kamie:」fire) followed by a colon, at the very start of the reply.
_SELF_PREFIX_RE = re.compile(r"^(?:([一-鿿]{1,3})|([A-Za-z]{1,8}))[：:]")


def _misread_re(name: str) -> re.Pattern:
    n = re.escape(name)
    return re.compile("|".join([
        rf"叫我{n}",                                                    # 你叫我User？
        rf"把我當(?:成|作)?{n}",
        rf"我(?:才|就|也|又|可|並)?不是{n}",
        rf"我(?:就|才|也|正)?是{n}(?!創|做|的|養|造|生|教|寫|帶|他|說)",   # not 我是User創造的
        rf"我(?:的名字)?(?:叫|叫做){n}",
        rf"{n}是我(?:的名字|嗎|[？?！!。]|$)",
    ]))


def _last_user_text(payload: bytes) -> str:
    """The last user message as typed (timestamp and name prefix removed), for the v53 SELF check."""
    try:
        msgs = json.loads(payload).get("messages") or []
    except Exception:
        return ""
    for m in reversed(msgs):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            c = _TS_PREFIX.sub("", m["content"])
            prefix = _settings.get("user_name_prefix", "")
            return c[len(prefix):] if prefix and c.startswith(prefix) else c
    return ""


def _identity_watch(reply: str, seq: int, last_user: str = "") -> dict:
    """Check one finished reply for the identity side effects; returns the flags for the timing log."""
    text = _MARKER_RE.sub("", reply).strip()
    flags = {"self_prefix": None, "misread": None, "self_suspect": None}
    _identity["watched"] += 1
    name = _user_name()
    if name and last_user and name in last_user:
        ms = _SELF_SUSPECT_RE.match(text)
        if ms:
            flags["self_suspect"] = ms.group(0)
            _identity["self_suspects"] += 1
            print(f"[proxy] ⚠ IDENTITY SELF SUSPECT #{seq}: user named {name}, reply opens「{ms.group(0)}…」-- "
                  f"she may be answering as {name} | {text[:80]!r}", flush=True)
    m = _SELF_PREFIX_RE.match(text)
    if m:
        label = m.group(1) or m.group(2)
        flags["self_prefix"] = label
        _identity["self_prefix"] += 1
        print(f"[proxy] ⚠ IDENTITY SELF-PREFIX #{seq}: reply starts with「{label}：」(ST #557 shape) | {text[:80]!r}",
              flush=True)
    if name:
        mm = _misread_re(name).search(text)
        if mm:
            flags["misread"] = mm.group(0)
            _identity["misread_suspects"] += 1
            print(f"[proxy] ⚠ IDENTITY MISREAD SUSPECT #{seq}:「{mm.group(0)}」-- the name prefix may have been read "
                  f"as her being called {name} | {text[:80]!r}", flush=True)
    if flags["self_prefix"] or flags["misread"] or flags["self_suspect"]:
        try:
            with _IDENTITY_EVENTS_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"seq": seq, "at": time.strftime("%Y-%m-%d %H:%M:%S"), "user_name": name,
                                    **flags, "text": text}, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[proxy] identity event write failed: {e}", flush=True)
    return flags


def _identity_health() -> dict:
    return {"identity_watched": _identity["watched"], "identity_self_prefix": _identity["self_prefix"],
            "identity_misread_suspects": _identity["misread_suspects"], "identity_self_suspects": _identity["self_suspects"],
            "identity_user_name": _user_name()}


# v53 task B-1 (principle 10): what she does with tools must be loud. Three things per reply:
#   tool_calls  -- structured OpenAI tool calls in the stream (names), and which of them name a
#                  tool that was NOT offered in this request (with strip_tools=on that is all of
#                  them; with tools present, a made-up name)
#   call_markers -- the card's own <|CALL [...]|> mechanism in the text (a different source: the
#                  card teaches that format; it exists whether or not the tools array is sent)
# Counted in /health, shouted in the log, appended with text to tool-events.jsonl.
_TOOL_EVENTS_PATH = Path(os.environ.get("LLM_PROXY_TOOL_EVENTS", str(HERE / "tool-events.jsonl")))
_tools_watch = {"replies": 0, "tool_calls": 0, "tool_calls_unknown": 0, "call_markers": 0}
_CALL_MARKER_RE = re.compile(r"<\|CALL\b[^|]*\|>")


def _tool_names_in(payload: bytes) -> list[str]:
    try:
        tools = json.loads(payload).get("tools") or []
    except Exception:
        return []
    return [str((t.get("function") or t).get("name")) for t in tools if isinstance(t, dict)]


def _tool_watch(names: list[str], reply: str, seq: int, offered: list[str]) -> dict:
    _tools_watch["replies"] += 1
    unknown = [n for n in names if n not in offered]
    markers = _CALL_MARKER_RE.findall(reply)
    flags = {"tool_calls": names or None, "unknown": unknown or None, "call_markers": markers or None,
             "offered": len(offered)}
    if names:
        _tools_watch["tool_calls"] += len(names)
    if unknown:
        _tools_watch["tool_calls_unknown"] += len(unknown)
        print(f"[proxy] ⚠ TOOL CALL NOT OFFERED #{seq}: {unknown} (offered {len(offered)})", flush=True)
    if markers:
        _tools_watch["call_markers"] += len(markers)
        print(f"[proxy] ⚠ CALL MARKER #{seq}: {markers} (card mechanism, no module connected) | "
              f"{_MARKER_RE.sub('', reply).strip()[:60]!r}", flush=True)
    if names or markers:
        try:
            with _TOOL_EVENTS_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"seq": seq, "at": time.strftime("%Y-%m-%d %H:%M:%S"), **flags,
                                    "text": _MARKER_RE.sub("", reply).strip()}, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[proxy] tool event write failed: {e}", flush=True)
    return flags


def _tools_watch_health() -> dict:
    return {"tool_replies": _tools_watch["replies"], "tool_calls_seen": _tools_watch["tool_calls"],
            "tool_calls_unknown": _tools_watch["tool_calls_unknown"], "call_markers": _tools_watch["call_markers"]}


def _record_act(has_act: bool):
    _act_recent.append(has_act)
    del _act_recent[:-_ACT_WINDOW]
    n = len(_act_recent)
    if n >= _ACT_MIN_SAMPLES:
        rate = sum(_act_recent) / n
        if rate < _ACT_FLOOR:
            print(f"[proxy] ⚠ ACT RATE COLLAPSED: {sum(_act_recent)}/{n} of recent replies carry an ACT marker",
                  flush=True)


# v57: replies AIRI will silently drop.
#
# v56 measured 2 of 40 turns on the shipping configuration losing the entire
# reply -- no chat bubble, no audio, no error -- and filed both under the
# marker parser. Replaying the two real samples says they are two different
# bugs, and neither of them is visible from inside AIRI:
#
#   A. the reply opens a `<|` marker that never closes. `llm-marker-parser`
#      repairs this when a `>` is present somewhere (v23), and drops the whole
#      buffer when there is no `>` at all -- which is what happened.
#   B. the reply contains an ordinary angle tag such as `<motion>`. The
#      STREAMING half of `response-categoriser` treats every tag as reasoning
#      and filters it out of speech, and with no closing tag it filters
#      everything to the end of the reply. Verified: `filterToSpeech` returns
#      the empty string for that sample, while `categorizeResponse` on the same
#      text calls it all speech -- the two halves disagree, and the streaming
#      one is the one that builds the message.
#
# The proxy cannot fix either from here (A's payload is malformed JSON with the
# dialogue buried in it, so flushing it would put JSON into the chat bubble and
# into TTS; B is upstream's reasoning filter). What it can do is stop the loss
# being invisible -- principle 10. Counted, logged loudly, and written to
# `swallow-events.jsonl` with the text, so a rate can be quoted next round
# instead of an impression.
_SWALLOW_PATH = Path(os.environ.get("LLM_PROXY_SWALLOW_LOG", str(HERE / "swallow-events.jsonl")))
_swallow = {"unterminated_marker": 0, "unterminated_no_gt": 0, "angle_tag": 0, "seen": 0}

# v73b task 3: how often the in-stream fixer had to act, split by what it did.
# `replies_touched` is the number of REPLIES with at least one event, which is the
# number to compare against `swallow_seen` -- `repaired`/`stripped` count events.
_markerfix = {"replies_touched": 0, "repaired": 0, "stripped": 0, "failed": 0}
_MARKERFIX_PATH = Path(os.environ.get("LLM_PROXY_MARKERFIX_LOG",
                                      str(Path(__file__).with_name("markerfix-events.jsonl"))))

# An angle tag as the categoriser sees one: `<name>` or `<name attr=...>`, and
# its closing form. `<|ACT ...|>` is not one of these -- `|` cannot start a tag
# name -- which is exactly why AIRI's own markers survive the filter.
#
# Void elements are excluded, and that exclusion is measured rather than
# assumed: replaying all 80 replies of v56 through the real pipeline (marker
# parser -> categorizer.consume -> filterToSpeech, three characters at a time)
# says the five `<br>` replies kept their text and were spoken, while the two
# carrying an unknown tag lost it. Flagging `<br>` would make this alarm fire on
# 5 of 80 healthy turns, and an alarm that cries wolf is not read next round.
_VOID_TAGS = {"br", "hr", "img", "input", "meta", "link", "wbr", "area",
              "base", "col", "embed", "param", "source", "track"}
_ANGLE_OPEN = re.compile(r"<([A-Za-z][A-Za-z0-9_.:-]*)(?:\s[^>]*)?>")
_ANGLE_CLOSE = re.compile(r"</([A-Za-z][A-Za-z0-9_.:-]*)>")


def _swallow_watch(reply: str, seq: int) -> dict:
    """Flag a finished reply that AIRI is about to drop on the floor."""
    if not reply:
        return {"unterminated_marker": False, "angle_tag": None}
    _swallow["seen"] += 1

    # A: last `<|` with no `|>` after it.
    open_at = reply.rfind("<|")
    unterminated = open_at >= 0 and reply.find("|>", open_at) < 0
    has_gt = unterminated and (reply.find(">", open_at + 2) >= 0)

    # B: an opening tag with no matching close eats to the end of the reply.
    opens = [m.group(1) for m in _ANGLE_OPEN.finditer(reply)]
    closes = {m.group(1) for m in _ANGLE_CLOSE.finditer(reply)}
    dangling = [t for t in opens if t not in closes and t.lower() not in _VOID_TAGS]

    flags = {
        "unterminated_marker": unterminated,
        # v23's repair needs a `>`; without one the whole buffer goes.
        "recoverable_by_v23": (unterminated and has_gt) or None,
        "angle_tag": (dangling or None),
    }
    if unterminated:
        _swallow["unterminated_marker"] += 1
        if not has_gt:
            _swallow["unterminated_no_gt"] += 1
        print(f"[proxy] ⚠ REPLY MAY BE SWALLOWED (seq {seq}): marker opened at {open_at} never closes"
              f"{' and has no > at all -- the whole reply will be dropped' if not has_gt else ''}",
              flush=True)
    if dangling:
        _swallow["angle_tag"] += 1
        print(f"[proxy] ⚠ REPLY MAY BE SWALLOWED (seq {seq}): unclosed angle tag(s) {dangling} -- "
              f"AIRI's streaming reasoning filter drops from there to the end", flush=True)
    if unterminated or dangling:
        try:
            with _SWALLOW_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"seq": seq, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    **flags, "reply": reply}, ensure_ascii=False) + chr(10))
        except Exception as e:
            print(f"[proxy] swallow log failed: {e}", flush=True)
    return flags


def _markerfix_health() -> dict:
    return {"markerfix": _settings.get("fix_markers"),
            "markerfix_replies_touched": _markerfix["replies_touched"],
            "markerfix_repaired": _markerfix["repaired"],
            "markerfix_stripped": _markerfix["stripped"],
            "markerfix_failed": _markerfix["failed"]}


def _note_markerfix(seq: int, events: list) -> None:
    """Log + count every time the fixer touched a reply. Loud on purpose (原則十):
    a marker written wrong is a real defect, and silently repairing it would trade
    one silent failure for another."""
    if not events:
        return
    _markerfix["replies_touched"] += 1
    for e in events:
        _markerfix["repaired" if e["kind"] == "repaired" else "stripped"] += 1
    kinds = ", ".join("%s %s" % (e["kind"], e["what"]) for e in events)
    print(f"[proxy] ⚠ MALFORMED MARKER (seq {seq}): {kinds} -- reply kept, see "
          f"{_MARKERFIX_PATH.name}", flush=True)
    try:
        with _MARKERFIX_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"seq": seq, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "events": events}, ensure_ascii=False) + chr(10))
    except Exception as e:
        print(f"[proxy] markerfix log failed: {e}", flush=True)


def _swallow_health() -> dict:
    return {"swallow_seen": _swallow["seen"],
            "swallow_unterminated_marker": _swallow["unterminated_marker"],
            "swallow_unterminated_no_gt": _swallow["unterminated_no_gt"],
            "swallow_angle_tag": _swallow["angle_tag"]}


def _act_health() -> dict:
    n = len(_act_recent)
    return {
        "act_samples": n,
        "act_rate_recent": (round(sum(_act_recent) / n, 2) if n else None),
        "act_collapsed": (n >= _ACT_MIN_SAMPLES and sum(_act_recent) / n < _ACT_FLOOR),
        "marker_memory_size": len(_marker_memory),
        **_tools_health(),
    }


def _tools_health() -> dict:
    """What tools she is being offered, and whether that set moved (v33)."""
    added = [n for n in _tools_seen if n not in _tools_baseline] if _tools_seen else []
    removed = [n for n in _tools_baseline if n not in _tools_seen] if _tools_seen else []
    return {
        "tools_count": len(_tools_seen),
        "tools_seen": _tools_seen,
        "tools_baseline_count": len(_tools_baseline),
        # Unknown (no request seen yet) is not a change; the check treats it as pass.
        "tools_changed": bool(_tools_seen and (added or removed)),
        "tools_added": added,
        "tools_removed": removed,
    }

# The exact prefix lines session-store.ts:101-102 prepends to every session's
# system message (verbatim from captured payloads, v25 supplementary Q3).
_INJECT_PREFIX_LINES = [
    "- For any programming code block, always specify the programming language "
    "that supported on @shikijs/rehype on the rendered markdown, eg. ```python ... ```\n",
    "- For any math equation, use LaTeX format, eg: $ x^3 $, always escape "
    "dollar sign outside math equation\n",
]
# resolveSystemPrompt appends widgetInstruction last, so everything from this
# marker to the end of the system message is the artistry injection.
_INJECT_SUFFIX_MARKER = "\n\n## Instruction: Widget Spawning (Legacy/Manual)"

# The card marks its style-demo block with these headers (card-v0.2-isolated).
_FEWSHOT_BEGIN = "【說話風格示範】"
_FEWSHOT_END = "【說話風格示範結束】"

# v61: the card marks its technical-control block with these headers
# (card-v0.2-isolated onward, in `scenario`; v0.1 has no headers and is left alone).
_FORMAT_BEGIN = "【輸出格式規則】"
_FORMAT_END = "【輸出格式規則結束】"
# v61 `format_rules=short`: the minimum that still names the wire protocol the
# lifted demos already demonstrate. Same delimiters as the full block, so a
# captured payload can be told apart from production by length alone. Whatever
# is NOT in this text -- DELAY, CALL, intensity, motion, the structural example,
# the English glosses -- is exactly what the arm removes; read its numbers with
# that list in mind.
_FORMAT_SHORT = (
    "【輸出格式規則】以下是技術性的輸出格式說明，不是對話內容，也不是說話風格的示範。\n\n"
    "每次回覆都以一個情緒標記開頭，格式固定是 <|ACT {\"emotion\":\"名稱\"}|>，標記後面接台詞；"
    "回覆中途情緒改變時，在新情緒開始的位置再放一個。\n\n"
    "名稱只能從這九個裡選：happy、sad、angry、think、surprised、awkward、question、curious、neutral。\n\n"
    "【輸出格式規則結束】"
)

# v20b showed 6/6 that a history without markers teaches the model to drop
# them, so a bare-text assistant example would wreck the ACT rate and tangle
# the style question with the marker question. Each replayed assistant example
# therefore gets a synthetic ACT prefix; the emotions below are hand-picked to
# match the tone of the 8 card examples, in card order. They are scaffolding
# for the experiment, not part of the personality definition.
_FEWSHOT_ACT = ["happy", "happy", "neutral", "curious", "happy", "surprised", "angry", "think"]
# v28: the card variants keep a SUBSET of the 8 pairs (3 / 0), so a positional
# table would hand the wrong emotion to the pairs that survive (pair 4 would get
# pair 1's "happy"). Look the emotion up by the pair's user line instead; the
# positional table above stays as the fallback, so the 8-pair cards (v0.2,
# v0.3) produce byte-identical messages to before.
_FEWSHOT_ACT_BY_USER = {
    "你是不是有點笨": "happy",
    "你只是個AI而已吧": "happy",
    "你看我這件衣服好看嗎": "neutral",          # v0.3 pair 3
    "現在幾點": "neutral",                      # v0.2 pair 3 (swapped out in v0.3)
    "User有沒有跟你講過誰的壞話": "curious",
    "（正在聊遊戲）欸我昨天看到一隻超肥的貓": "happy",
    "明天要交報告，還沒開始寫": "surprised",
    "不對吧，根本不是這樣": "angry",
    "你覺得誰對": "think",
}
_seq = 0

# v30 W2: scaffolding text for the non-default fewshot_mode transports. These
# are proxy-side framing, not card content -- the 8 pairs' text is never
# edited. Kept in one place so a change is a one-line diff.
_FEWSHOT_BOUNDARY = "（以上是說話風格示範，不是這次對話的紀錄。對話從下一句開始。）"
_FEWSHOT_SINGLE_HEAD = "【說話風格示範】以下是你說話的樣子。照這個調性回，不要照抄字句。這些只是示範，不是我們講過的話："
_FEWSHOT_SINGLE_TAIL = "【說話風格示範結束】看完回一個字就好。"
_FEWSHOT_SINGLE_ACK = "好。"
_FEWSHOT_TAGGED_HEAD = "【說話風格示範】下面 <examples> 裡是你說話的樣子。這些是示範，不是曾經發生過的對話，不要把它們當成聊過的內容；只學語氣與長度，不要逐字照抄。"
_FEWSHOT_TAGGED_TAIL = "【說話風格示範結束】"
_FEWSHOT_LINES_OPEN = "[示範對話開始——以下只是說話風格的示範，不是真正聊過的話]"
_FEWSHOT_LINES_CLOSE = "[示範對話結束。以下才是真正的新對話]"
# v64 E-2: messages+boundary puts _FEWSHOT_BOUNDARY in a system message AFTER the
# demo turns, and Qwen3.5's official chat template raises on any system message
# that is not first ("System message must be at the beginning."); Ollama renders
# it in place without a word. The two official-format alternatives:
#   messages+boundary_user -- the same sentence, same position, role user
#   messages+boundary_head -- nothing between the demos and the real turns; this
#                            note closes the FIRST system message instead. It has
#                            to be reworded (it now comes before the demos), so
#                            this mode moves position AND wording.
_FEWSHOT_BOUNDARY_HEAD = "（接下來的對話裡，前 {n} 則是說話風格示範，不是這次對話的紀錄；真正的對話從第 {m} 則開始。）"

# Long generations must not be cut off, but a dead upstream should not hang AIRI
# forever either. Connect fast, read patiently.
_TIMEOUT = httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=5.0)
_client: httpx.AsyncClient | None = None

# Hop-by-hop headers belong to this connection, not the forwarded one.
_SKIP_REQ = {"host", "content-length", "connection", "accept-encoding"}
_SKIP_RES = {"content-length", "content-encoding", "transfer-encoding", "connection"}


class Clock:
    """Wall-clock instants that keep perf_counter spacing between them."""

    def __init__(self):
        self._w = time.time()
        self._p = time.perf_counter()

    def now(self) -> float:
        return self._w + (time.perf_counter() - self._p)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).isoformat(timespec="milliseconds")


def _log(record: dict):
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@app.on_event("startup")
async def _startup():
    global _client, _seq
    _client = httpx.AsyncClient(timeout=_TIMEOUT)
    _load_marker_memory()
    _load_tools_baseline()
    _load_gist_memory()          # v83 task B
    # v25: a restart used to reset _seq to 0 and silently overwrite existing
    # payload-NNN.json captures (v24 lost its E0 payloads that way). Resume
    # numbering after whatever is already on disk, and say so.
    if CAPTURE_DIR:
        try:
            existing = [int(p.stem.split("-")[1])
                        for p in Path(CAPTURE_DIR).glob("payload-*.json")
                        if p.stem.split("-")[1].isdigit()]
            if existing:
                _seq = max(existing)
                print(f"[proxy] capture dir has payload-{_seq:03d}; "
                      f"resuming numbering at {_seq + 1} instead of overwriting", flush=True)
        except Exception as e:
            print(f"[proxy] capture seq scan failed: {e}", flush=True)
    print(f"[proxy] ready on port {os.environ.get('LLM_PROXY_PORT', '11435')} "
          f"-> {UPSTREAM}  thinking={_settings['thinking']}", flush=True)


@app.on_event("shutdown")
async def _shutdown():
    if _client:
        await _client.aclose()


@app.get("/config")
def get_config():
    return {**_settings, "upstream": UPSTREAM}


@app.post("/config")
def set_config(patch: dict):
    """Flip thinking without a restart, so both sides can be A/B'd inside AIRI."""
    for key in ("thinking", "strip_tools", "fewshot_messages", "sampling", "strip_injections", "strip_widget",
                "fix_markers", "restore_markers", "fewshot_user_prefix", "fewshot_prefix_skip_named", "reply_cap"):
        if key not in patch:
            continue
        v = str(patch[key]).lower()
        if v not in ("on", "off"):
            return JSONResponse({"error": f"{key} must be 'on' or 'off'"}, status_code=400)
        _settings[key] = v
    # v84 B-1 #4: the reply ceiling's value (tokens); reply_cap on/off is in the loop above.
    if "pad_ms" in patch:
        try:
            v = int(patch["pad_ms"])
            assert 0 <= v <= 10000
        except Exception:
            return JSONResponse({"error": "pad_ms must be an integer 0..10000"}, status_code=400)
        _settings["pad_ms"] = v
    if "reply_cap_tokens" in patch:
        try:
            v = int(patch["reply_cap_tokens"])
            assert v >= 1
        except Exception:
            return JSONResponse({"error": "reply_cap_tokens must be an integer >= 1"}, status_code=400)
        _settings["reply_cap_tokens"] = v
    for key in ("sampling_temperature", "sampling_top_p"):
        if key in patch:
            try:
                _settings[key] = float(patch[key])
            except (TypeError, ValueError):
                return JSONResponse({"error": f"{key} must be a number"}, status_code=400)
    if "model_override" in patch:
        _settings["model_override"] = str(patch["model_override"])
    # v89 B-2: "" turns the thinking-on routing off
    if "think_model" in patch:
        _settings["think_model"] = str(patch["think_model"] or "")
    # v51: plain strings; "" turns each off. Kept free-form on purpose (the seam
    # must serve a friend / a voiceprint-identified speaker later, not just User).
    for key in ("user_persona", "user_name_prefix", "user_frame"):
        if key in patch:
            _settings[key] = str(patch[key] or "")
    # v54: the seed word and its wording are free-form settings (00 §六 3);
    # "" turns it off and restores the byte-identical v53 T1 payload.
    for key in ("anchor_seed", "anchor_seed_template"):
        if key in patch:
            _settings[key] = str(patch[key] or "")
    if "anchor_seed_position" in patch:
        v = str(patch["anchor_seed_position"]).lower()
        if v not in ("system", "tail"):
            return JSONResponse({"error": "anchor_seed_position must be 'system' or 'tail'"}, status_code=400)
        _settings["anchor_seed_position"] = v
    if "user_persona_position" in patch:
        v = str(patch["user_persona_position"]).lower()
        if v not in ("head", "tail"):
            return JSONResponse({"error": "user_persona_position must be 'head' or 'tail'"}, status_code=400)
        _settings["user_persona_position"] = v
    if "fewshot_mode" in patch:
        v = str(patch["fewshot_mode"])
        if v not in ("messages", "messages+boundary", "messages+boundary_user", "messages+boundary_head",
                     "single_user", "system_tagged", "system_lines", "drop"):
            return JSONResponse({"error": "fewshot_mode must be messages | messages+boundary | messages+boundary_user | "
                                          "messages+boundary_head | single_user | system_tagged | system_lines | drop"}, status_code=400)
        _settings["fewshot_mode"] = v
    # v61: where the card's technical-control block goes (see _settings).
    if "format_rules" in patch:
        v = str(patch["format_rules"]).lower()
        if v not in ("card", "short", "separate", "off"):
            return JSONResponse({"error": "format_rules must be card | short | separate | off"}, status_code=400)
        _settings["format_rules"] = v
    # v83 task B: the history layer. Profile names are the presets in _HISTORY_PROFILES.
    if "history_profile" in patch:
        v = str(patch["history_profile"])
        if v not in _HISTORY_PROFILES:
            return JSONResponse({"error": "history_profile must be one of " + " | ".join(_HISTORY_PROFILES)}, status_code=400)
        _settings["history_profile"] = v
    if "history_keep_last" in patch:
        try:
            v = int(patch["history_keep_last"])
            assert v >= 0
        except Exception:
            return JSONResponse({"error": "history_keep_last must be an integer >= 0"}, status_code=400)
        _settings["history_keep_last"] = v
    if "history_her_older" in patch:
        v = str(patch["history_her_older"])
        if v not in ("verbatim", "gist", "drop"):
            return JSONResponse({"error": "history_her_older must be verbatim | gist | drop"}, status_code=400)
        _settings["history_her_older"] = v
    if "history_record_role" in patch:
        v = str(patch["history_record_role"])
        if v not in ("user", "system", "inplace"):
            return JSONResponse({"error": "history_record_role must be user | system | inplace"}, status_code=400)
        _settings["history_record_role"] = v
    for key in ("history_record_head", "history_record_tail", "history_her_label", "history_her_gist_label",
                "history_gist_model", "history_gist_prompt"):
        if key in patch:
            _settings[key] = str(patch[key] or "")
    if "history_gist_timeout_s" in patch:
        try:
            v = float(patch["history_gist_timeout_s"])
            assert v > 0
        except Exception:
            return JSONResponse({"error": "history_gist_timeout_s must be a number > 0"}, status_code=400)
        _settings["history_gist_timeout_s"] = v
    print(f"[proxy] config -> {_settings}  (derived user_name={_user_name()!r})", flush=True)
    return _settings


@app.get("/health")
def health():
    return {"status": "ok", **_settings, **_act_health(), **_identity_health(), **_tools_watch_health(),
            **_swallow_health(), **_markerfix_health(), **_history_health(), **_reply_cap_health(),
            **_think_route_health()}


def _rewrite(raw: bytes) -> tuple[bytes, bool]:
    """Add `reasoning_effort` when thinking is off. Everything else is untouched.

    Returns the body to forward and whether the field was actually injected, so
    the log records what happened rather than what was configured.
    """
    if _settings["thinking"] != "off":
        return raw, False
    try:
        body = json.loads(raw)
    except Exception:
        # Not JSON we understand: forward byte-for-byte rather than guess.
        return raw, False
    if not isinstance(body, dict):
        return raw, False
    # An explicit caller-supplied value wins; this proxy only fills the gap.
    if "reasoning_effort" in body:
        return raw, False
    body["reasoning_effort"] = "none"
    # ensure_ascii=False matters here: re-encoding Chinese as \uXXXX escapes
    # inflated the forwarded body by ~40% (19KB -> 27KB) for no benefit. Token
    # count is unaffected, but the wire cost is real.
    return json.dumps(body, ensure_ascii=False).encode("utf-8"), True


def _convert_fewshot(body: dict) -> bool:
    """Move the card's style-demo block from the system message into alternating
    user/assistant messages right after it. Returns whether anything changed.

    The block's text is reused verbatim -- the pairs the regex finds are the
    same bytes the system prompt carried -- so the only variable that moves is
    the transport format, which is exactly what E3 isolates."""
    import re
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs or msgs[0].get("role") != "system":
        return False
    content = msgs[0].get("content")
    if not isinstance(content, str):
        return False
    start = content.find(_FEWSHOT_BEGIN)
    end = content.find(_FEWSHOT_END)
    if start < 0 or end < 0:
        return False
    block = content[start:end]
    pairs = re.findall(r"使用者：(.+)\n你：(.+)", block)
    if not pairs:
        return False
    mode = _settings.get("fewshot_mode", "messages")
    examples = []
    for i, (u, a) in enumerate(pairs):
        emotion = _FEWSHOT_ACT_BY_USER.get(
            u.strip(), _FEWSHOT_ACT[i] if i < len(_FEWSHOT_ACT) else "neutral")
        examples.append((u.strip(), f'<|ACT {{"emotion":"{emotion}"}}|> {a.strip()}'))
    # v52: give the example user lines the same speaker label as the real turns.
    # No-op unless fewshot_user_prefix=on (see _label_example for the skip rule).
    ex_prefix = _fewshot_example_prefix()
    if ex_prefix:
        examples = [(_label_example(u, ex_prefix), a) for u, a in examples]
    stripped_sys = (content[:start] + content[end + len(_FEWSHOT_END):]).strip()
    stripped_sys = re.sub(r"\n{3,}", "\n\n", stripped_sys)
    sys_msg = {**msgs[0], "content": stripped_sys}
    if mode == "drop":
        # v81 task E: the block is lifted out and nothing is put back -- no demo
        # turns, no boundary message. What remains is the system text minus the
        # block (same stripped_sys every other lifting mode starts from), followed
        # directly by the real turns: the training shape of the pilot.
        body["messages"] = [sys_msg, *msgs[1:]]
    elif mode == "messages":
        # v24 E3 transport: alternating turns, nothing marks them as examples.
        fewshot = []
        for u, a in examples:
            fewshot.append({"role": "user", "content": u})
            fewshot.append({"role": "assistant", "content": a})
        body["messages"] = [sys_msg, *fewshot, *msgs[1:]]
    elif mode == "messages+boundary":
        # v30 W2 candidate: same turns, but a second system message closes the
        # demonstration block so the model does not read the turns as the
        # conversation it is in (Ollama 0.32.14 renderers accept non-leading
        # system messages).
        fewshot = []
        for u, a in examples:
            fewshot.append({"role": "user", "content": u})
            fewshot.append({"role": "assistant", "content": a})
        body["messages"] = [sys_msg, *fewshot, {"role": "system", "content": _FEWSHOT_BOUNDARY}, *msgs[1:]]
    elif mode == "messages+boundary_user":
        # v64 E-2: byte-for-byte the messages+boundary layout except the role of
        # the boundary (system -> user), which is what the Qwen team says to do
        # with a non-leading system instruction. The real first user turn
        # follows it directly (two user turns in a row). It is not prefixed with
        # the user's name: identity labelling ran before this conversion and only
        # touches what the person typed.
        fewshot = []
        for u, a in examples:
            fewshot.append({"role": "user", "content": u})
            fewshot.append({"role": "assistant", "content": a})
        body["messages"] = [sys_msg, *fewshot, {"role": "user", "content": _FEWSHOT_BOUNDARY}, *msgs[1:]]
    elif mode == "messages+boundary_head":
        # v64 E-2: no mid-conversation message at all. The note that closes the
        # demonstration goes at the end of the first system message (before the
        # persona paragraph, which _append_user_persona adds later), counting the
        # demo messages so it stays true if the card's number of pairs changes.
        fewshot = []
        for u, a in examples:
            fewshot.append({"role": "user", "content": u})
            fewshot.append({"role": "assistant", "content": a})
        note = _FEWSHOT_BOUNDARY_HEAD.format(n=len(fewshot), m=len(fewshot) + 1)
        sys_msg = {**sys_msg, "content": stripped_sys + "\n\n" + note}
        body["messages"] = [sys_msg, *fewshot, *msgs[1:]]
    elif mode == "single_user":
        # v30 W2 candidate: all pairs inside ONE user message, explicitly
        # labelled as demonstrations, followed by a one-word acknowledgement,
        # so the history contains exactly one exchange "here are examples /
        # ok" instead of eight fake exchanges.
        lines = [_FEWSHOT_SINGLE_HEAD]
        for u, a in examples:
            lines.append(f"{_ex_user_line(u, ex_prefix)}\n你：{a}")
        lines.append(_FEWSHOT_SINGLE_TAIL)
        body["messages"] = [sys_msg,
                            {"role": "user", "content": "\n\n".join(lines)},
                            {"role": "assistant", "content": _FEWSHOT_SINGLE_ACK},
                            *msgs[1:]]
    elif mode == "system_tagged":
        # v30 W2 candidate (research rank 1): examples stay in the SYSTEM
        # prompt, each pair wrapped in <example> tags inside <examples>, with a
        # one-line scaffold saying they are demonstrations, not past dialogue,
        # and not to be copied. Never enters the user/assistant channels, so
        # it cannot be read as history; the tags are the variable vs. the old
        # untagged block. The ACT prefix is kept inside each example as text.
        tagged = [_FEWSHOT_TAGGED_HEAD, "<examples>"]
        for u, a in examples:
            tagged.append(f"<example>\n{_ex_user_line(u, ex_prefix)}\n你：{a}\n</example>")
        tagged.append("</examples>")
        tagged.append(_FEWSHOT_TAGGED_TAIL)
        block_text = "\n".join(tagged)
        sys_with_block = (content[:start] + block_text + content[end + len(_FEWSHOT_END):]).strip()
        sys_with_block = re.sub(r"\n{3,}", "\n\n", sys_with_block)
        body["messages"] = [{**msgs[0], "content": sys_with_block}, *msgs[1:]]
    elif mode == "system_lines":
        # v30 W2 candidate (research rank 2, SillyTavern layout without the
        # `name` field Ollama drops): a system boundary, then every example
        # line as its own system message with a speaker label in the text,
        # then a closing system boundary, then the real turns.
        lines = [{"role": "system", "content": _FEWSHOT_LINES_OPEN}]
        for u, a in examples:
            lines.append({"role": "system", "content": _ex_user_line(u, ex_prefix)})
            lines.append({"role": "system", "content": f"你：{a}"})
        lines.append({"role": "system", "content": _FEWSHOT_LINES_CLOSE})
        body["messages"] = [sys_msg, *lines, *msgs[1:]]
    else:
        return False
    return True


_TS_PREFIX = re.compile(r"^(\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\] )")


def _user_name() -> str:
    """The user's bare name, derived from user_name_prefix (「User：」→「User」)."""
    return _settings.get("user_name_prefix", "").strip().rstrip("：:").strip()


def _fewshot_example_prefix() -> str:
    """v52: the label to put on few-shot EXAMPLE user lines ("" = leave them as v51 did)."""
    if _settings.get("fewshot_user_prefix") != "on":
        return ""
    return _settings.get("user_name_prefix", "")


def _label_example(u: str, prefix: str) -> str:
    """Prefix one example user line -- unless fewshot_prefix_skip_named=on and the
    line already contains the user's name. The rule sees the name, not the speaker;
    its bypasses are listed next to the setting.

    Note: a skipped example line is indistinguishable from a real turn if the
    proxy's OWN output were fed back in (it would then be labelled). That is not
    the production path -- AIRI resends its stored history, never the proxy's
    converted body -- and the offline test checks the production property instead."""
    if not prefix or u.startswith(prefix):
        return u
    if _settings.get("fewshot_prefix_skip_named") == "on":
        name = _user_name()
        if name and name in u:
            return u
    return prefix + u


def _ex_user_line(u: str, prefix: str) -> str:
    """Speaker-labelled example line for the transports that render labels as text:
    the v30 scaffold label「使用者：」, or the user's own prefix when it is already on."""
    return u if (prefix and u.startswith(prefix)) else f"使用者：{u}"


def _apply_user_identity(body: dict) -> bool:
    """v51: name the real user turns. Returns whether any identity setting is on.

    Prefix goes AFTER AIRI's `[YYYY-MM-DD HH:MM] ` stamp so the timestamp stays
    where the model has seen it since v25. Only messages after the leading
    system message are touched; few-shot examples are inserted later by
    `_convert_fewshot` and are never prefixed. Idempotent: a turn that already
    starts with the prefix is left alone (AIRI resends the whole history)."""
    persona = _settings.get("user_persona", "")
    prefix = _settings.get("user_name_prefix", "")
    frame = _settings.get("user_frame", "")
    if not persona and not prefix and not frame:
        return False
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return False
    if prefix:
        for m in msgs[1:]:
            if m.get("role") != "user" or not isinstance(m.get("content"), str):
                continue
            c = m["content"]
            mt = _TS_PREFIX.match(c)
            head, rest = (mt.group(1), c[mt.end():]) if mt else ("", c)
            if rest.startswith(prefix):
                continue
            m["content"] = f"{head}{prefix}{rest}"
    return True


def _append_user_persona(body: dict) -> None:
    """Append the persona paragraph to the leading system message (after any
    few-shot extraction, so it lands at the end of the card text, the position
    SillyTavern gives the persona description by default)."""
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs or msgs[0].get("role") != "system":
        return
    c = msgs[0].get("content")
    if not isinstance(c, str):
        return
    persona = _settings["user_persona"].strip()
    if persona and persona not in c:
        msgs[0]["content"] = c.rstrip() + "\n\n" + persona


def _prepend_system_head(body: dict) -> None:
    """v52: put the frame sentence and, when user_persona_position=head, the
    persona paragraph BEFORE the card text, each as its own paragraph:
    frame → persona → card (SillyTavern's default prompt-manager order).
    Runs after the few-shot extraction so the head sits on the card text itself.
    Idempotent: a paragraph already present anywhere in the system text is not
    added again (AIRI resends the whole history; offline tests re-run bodies)."""
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs or msgs[0].get("role") != "system":
        return
    c = msgs[0].get("content")
    if not isinstance(c, str):
        return
    head = []
    frame = _settings.get("user_frame", "").strip()
    if frame and frame not in c:
        head.append(frame)
    persona = _settings.get("user_persona", "").strip()
    if persona and _settings.get("user_persona_position") == "head" and persona not in c:
        head.append(persona)
    if head:
        msgs[0]["content"] = "\n\n".join(head) + "\n\n" + c.lstrip()


def _anchor_seed_text() -> str:
    """The seed sentence, or "" when the switch is off."""
    seed = _settings.get("anchor_seed", "").strip()
    if not seed:
        return ""
    tpl = _settings.get("anchor_seed_template") or "{seed}"
    return tpl.replace("{seed}", seed).strip()


def _apply_anchor_seed(body: dict) -> str:
    """v54 task B: offer one candidate anchor. Returns the text used, or "".

    position "system": appended as its own paragraph to the leading system
    message (runs after the few-shot extraction, so it lands on the card text
    and not inside the demo block -- same discipline as _append_user_persona).
    position "tail": inserted as its own system message directly before the
    NEWEST user turn, i.e. at the recency end rather than in the stable prefix.
    Idempotent in both positions: AIRI resends the whole history every turn and
    the offline tests re-run the same bodies."""
    text = _anchor_seed_text()
    if not text:
        return ""
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return ""
    if _settings.get("anchor_seed_position", "system") == "tail":
        for m in msgs:
            if m.get("role") == "system" and m.get("content") == text:
                return text
        last_user = None
        for i, m in enumerate(msgs):
            if m.get("role") == "user":
                last_user = i
        if last_user is None:
            return ""
        msgs.insert(last_user, {"role": "system", "content": text})
        return text
    if msgs[0].get("role") != "system":
        return ""
    c = msgs[0].get("content")
    if not isinstance(c, str):
        return ""
    if text not in c:
        msgs[0]["content"] = c.rstrip() + "\n\n" + text
    return text


def _strip_injections(body: dict) -> int:
    """Remove AIRI's non-card system-message wrappers. Returns chars removed."""
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs or msgs[0].get("role") != "system":
        return 0
    content = msgs[0].get("content")
    if not isinstance(content, str):
        return 0
    before = len(content)
    for line in _INJECT_PREFIX_LINES:
        if content.startswith(line):
            content = content[len(line):]
    idx = content.find(_INJECT_SUFFIX_MARKER)
    if idx >= 0:
        content = content[:idx]
    removed = before - len(content)
    if removed:
        msgs[0] = {**msgs[0], "content": content}
    return removed


def _strip_widget(body: dict) -> int:
    """v73 task D: remove ONLY the artistry widget instruction block. Returns chars removed.

    resolveSystemPrompt appends widgetInstruction last, so everything from the
    marker to the end of the incoming system message is that block -- the same
    exact-match rule as _strip_injections, and a no-op when it is absent. Runs
    before every append, so the identity seam and the persona paragraph the
    proxy adds afterwards are untouched.
    """
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs or msgs[0].get("role") != "system":
        return 0
    content = msgs[0].get("content")
    if not isinstance(content, str):
        return 0
    idx = content.find(_INJECT_SUFFIX_MARKER)
    if idx < 0:
        return 0
    removed = len(content) - idx
    msgs[0] = {**msgs[0], "content": content[:idx]}
    return removed


def _apply_format_rules(body: dict) -> str:
    """v61 task B: separate the technical-control block from the personality.

    Runs after the few-shot lift (the boundary message is where `separate`
    lands) and before the persona paragraph is appended (so the persona still
    closes the system text exactly as in production). Returns the mode that
    actually did something, or "" -- mode `card`, or a system text without the
    two headers (v0.1), in which case nothing is touched, same rule as the
    few-shot lift. Idempotent: a `separate` message already present is not
    inserted twice (AIRI never stores it, but the offline tests re-run bodies)."""
    mode = _settings.get("format_rules", "card")
    if mode == "card":
        return ""
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs or msgs[0].get("role") != "system":
        return ""
    content = msgs[0].get("content")
    if not isinstance(content, str):
        return ""
    start = content.find(_FORMAT_BEGIN)
    end = content.find(_FORMAT_END)
    if start < 0 or end < 0 or end < start:
        return ""
    block = content[start:end + len(_FORMAT_END)]
    before, after = content[:start], content[end + len(_FORMAT_END):]
    new = before + (_FORMAT_SHORT if mode == "short" else "") + after
    new = re.sub(r"\n{3,}", "\n\n", new).strip()
    msgs[0] = {**msgs[0], "content": new}
    if mode == "separate":
        for m in msgs[1:]:
            if m.get("role") == "system" and m.get("content") == block:
                return mode
        at = 1
        for i, m in enumerate(msgs):
            if i > 0 and m.get("role") == "system" and m.get("content") == _FEWSHOT_BOUNDARY:
                at = i + 1
                break
        msgs.insert(at, {"role": "system", "content": block})
    return mode


# ---------------------------------------------------------------------------------------------
# v83 task B: the history layer (see the `history_profile` comment in _settings)
# ---------------------------------------------------------------------------------------------

def _history_event(rec: dict) -> None:
    """history-events.jsonl: the loud record of anything that went wrong in the layer."""
    try:
        with _HISTORY_EVENTS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"iso": datetime.now().isoformat(timespec="seconds"), **rec}, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[proxy] history event write failed: {e}", flush=True)


def _history_resolve(name: str):
    """Profile name -> (name, spec or None). An unknown name is LOUD and leaves the history untouched."""
    name = str(name or "off").strip()
    if name not in _HISTORY_PROFILES:
        _history_counters["history_bad_profile"] += 1
        print(f"[proxy] ⚠ HISTORY PROFILE UNKNOWN {name!r} -- this request's history is forwarded untouched", flush=True)
        _history_event({"kind": "bad_profile", "profile": name})
        return name, None
    spec = _HISTORY_PROFILES[name]
    if spec == "custom":
        spec = {"keep_last": int(_settings.get("history_keep_last", 0)),
                "her_older": str(_settings.get("history_her_older", "verbatim")),
                "record_role": str(_settings.get("history_record_role", "user"))}
    return name, spec


def _hist_norm(text: str) -> str:
    """Key normalisation shared by the gist memory and its lookup: no timestamp, no name
    prefix, no control markers, no whitespace (AIRI re-stores her reply with the markers
    stripped and blank lines between segments, v81 §8)."""
    t = _TS_PREFIX.sub("", text or "")
    prefix = _settings.get("user_name_prefix", "")
    if prefix and t.startswith(prefix):
        t = t[len(prefix):]
    t = _MARKER_RE.sub("", t)
    return re.sub(r"\s+", "", t)


def _gist_key(his: str, her: str) -> str:
    return _hist_norm(his) + "||" + _hist_norm(her)


def _her_plain(text: str) -> str:
    """Her line as it goes into the record: markers out, segments joined by one space."""
    return re.sub(r"\s*\n\s*", " ", _MARKER_RE.sub("", text or "").strip())


def _history_split(hist: list, keep_last: int) -> tuple[list, list]:
    """(older, recent): recent = the last `keep_last` exchanges, counted by user turns."""
    if keep_last <= 0:
        return list(hist), []
    users = [i for i, m in enumerate(hist) if m.get("role") == "user"]
    if len(users) <= keep_last:
        return [], list(hist)
    b = users[-keep_last]
    return list(hist[:b]), list(hist[b:])


def _history_pairs(older: list) -> list[tuple[str, str]]:
    """(his line, her reply) for every assistant message in `older` -- his = the user turn before it."""
    out, last_his = [], ""
    for m in older:
        c = m.get("content") if isinstance(m.get("content"), str) else ""
        if m.get("role") == "user":
            last_his = c
        elif m.get("role") == "assistant":
            out.append((last_his, c))
    return out


def _longest_common(a: str, b: str) -> int:
    """Longest common substring length (DP; both strings are one chat line)."""
    best, prev = 0, [0] * (len(b) + 1)
    for ca in a:
        cur = [0] * (len(b) + 1)
        for j, cb in enumerate(b, 1):
            if ca == cb:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def _load_gist_memory() -> None:
    if not _HISTORY_GIST_MEMORY_PATH.exists():
        return
    for line in _HISTORY_GIST_MEMORY_PATH.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
            _gist_memory[rec["key"]] = rec
        except Exception:
            continue


async def _gist_call(his: str, her: str) -> dict:
    """One gist of her line from history_gist_model on Ollama's native /api/chat with
    think=false (the /v1 endpoint drops `think`, which is why this proxy exists at all).
    Up to two attempts; a gist that is empty or shares >= 8 characters with her line
    (the self-copy ruler's threshold) is rejected -- a copied gist would defeat the
    point of the literal arm. Returns {"gist": str | None, "attempts", "why", "ms"}."""
    user = _user_name() or "他"
    sys_prompt = _settings["history_gist_prompt"].replace("{user}", user)
    his_plain = _TS_PREFIX.sub("", his or "")
    prefix = _settings.get("user_name_prefix", "")
    if prefix and his_plain.startswith(prefix):
        his_plain = his_plain[len(prefix):]
    her_plain = _her_plain(her)
    t0 = time.perf_counter()
    why = []
    for attempt in (1, 2):
        extra = "" if attempt == 1 else "（上一次寫出來是空的，或太像她的原句。這次完全用你自己的話寫，一句就好。）"
        body = {"model": _settings["history_gist_model"], "stream": False, "think": False,
                "messages": [{"role": "system", "content": sys_prompt + extra},
                             {"role": "user", "content": f"{user}說：{his_plain.strip()}\nKamie 回：{her_plain}"}],
                "options": {"temperature": 0.2, "seed": 0, "num_predict": 96}}
        try:
            assert _client is not None
            r = await _client.post(f"{UPSTREAM}/api/chat", json=body, timeout=_settings["history_gist_timeout_s"])
            r.raise_for_status()
            g = ((r.json().get("message") or {}).get("content") or "")
        except Exception as e:
            why.append(f"attempt {attempt}: {type(e).__name__}: {e}")
            continue
        g = re.sub(r"\s+", " ", _MARKER_RE.sub("", g)).strip().strip("「」『』\"'").strip()
        if not g:
            why.append(f"attempt {attempt}: empty")
            continue
        lc = _longest_common(g, her_plain)
        if lc >= 8:
            why.append(f"attempt {attempt}: shares {lc} characters with her line: {g}")
            continue
        return {"gist": g, "attempts": attempt, "why": why, "ms": round((time.perf_counter() - t0) * 1000)}
    return {"gist": None, "attempts": 2, "why": why, "ms": round((time.perf_counter() - t0) * 1000)}


async def _gist_make(key: str, his: str, her: str) -> dict:
    res = await _gist_call(his, her)
    if res["gist"]:
        _history_counters["history_gist_ok"] += 1
        if res["attempts"] > 1:
            _history_counters["history_gist_retried"] += 1
        rec = {"key": key, "gist": res["gist"], "attempts": res["attempts"], "ms": res["ms"],
               "model": _settings["history_gist_model"], "iso": datetime.now().isoformat(timespec="seconds")}
        _gist_memory[key] = rec
        try:
            with _HISTORY_GIST_MEMORY_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[proxy] gist memory write failed: {e}", flush=True)
    else:
        _history_counters["history_gist_failed"] += 1
        print(f"[proxy] ⚠ HISTORY GIST FAILED ({res['why']})", flush=True)
        _history_event({"kind": "gist_failed", "key": key[:120], "why": res["why"]})
        _gist_tasks.pop(key, None)          # a later request may try again
    return res


def _gist_for(key: str, his: str, her: str):
    """The one task per key: the finished stream schedules it and the next request's lookup
    awaits the same task, so a gist is never computed twice at once."""
    t = _gist_tasks.get(key)
    if t is None:
        t = asyncio.ensure_future(_gist_make(key, his, her))
        _gist_tasks[key] = t
    return t


async def _history_prepare(raw: bytes, spec: dict | None) -> dict:
    """Before forwarding: every gist this request needs exists or has visibly failed.
    Cached ones are free; pending ones (scheduled when her reply finished) are awaited;
    missing ones start now. Waiting is bounded; anything not back in time is a
    fallback that _apply_history_layer reports loudly."""
    out = {"gists": {}, "wait_ms": 0, "missing": 0}
    if not spec or spec.get("her_older") != "gist":
        return out
    try:
        body = json.loads(raw)
        msgs = body.get("messages")
    except Exception:
        return out
    if not isinstance(msgs, list) or len(msgs) < 3 or msgs[-1].get("role") != "user":
        return out
    older, _ = _history_split(msgs[1:-1], int(spec.get("keep_last", 0)))
    tasks = {}
    for his, her in _history_pairs(older):
        key = _gist_key(his, her)
        if key in _gist_memory:
            out["gists"][key] = _gist_memory[key]["gist"]
        else:
            tasks[key] = _gist_for(key, his, her)
    if tasks:
        t0 = time.perf_counter()
        done, _pending = await asyncio.wait(list(tasks.values()), timeout=2 * _settings["history_gist_timeout_s"] + 5)
        out["wait_ms"] = round((time.perf_counter() - t0) * 1000)
        for key, t in tasks.items():
            g = None
            if t in done and not t.cancelled() and t.exception() is None:
                g = (t.result() or {}).get("gist")
            if g:
                out["gists"][key] = g
            else:
                out["missing"] += 1
    return out


def _apply_history_layer(body: dict, name: str, spec: dict | None, gists: dict, info: dict) -> bool:
    """Re-render the real conversation history per the profile. Runs after the identity
    prefix (his lines already carry 「User：」) and BEFORE the few-shot lift, so the demos
    are never touched: at this point messages[1:-1] are exactly AIRI's real turns and
    messages[-1] is the newest user turn, which is never changed. Returns whether it acted;
    `info` gets what it did (for the capture sidecar and the timing log)."""
    info["profile"] = name
    msgs = body.get("messages")
    if (not spec or not isinstance(msgs, list) or len(msgs) < 3 or msgs[0].get("role") != "system"
            or msgs[-1].get("role") != "user"):
        return False
    hist = msgs[1:-1]
    older, recent = _history_split(hist, int(spec.get("keep_last", 0)))
    info.update({"keep_last": int(spec.get("keep_last", 0)), "her_older": spec.get("her_older"),
                 "record_role": spec.get("record_role"), "n_history": len(hist), "n_older": len(older),
                 "n_recent": len(recent), "fallbacks": 0, "gists_used": [], "her_lines": []})
    if not older:
        return False
    user = _user_name() or "他"
    mode = spec.get("her_older", "verbatim")
    inplace = spec.get("record_role") == "inplace"
    lines, kept, last_his = [], [], ""
    for m in older:
        role, c = m.get("role"), m.get("content")
        if not isinstance(c, str):
            c = json.dumps(c, ensure_ascii=False)
        if role == "user":
            last_his = c
            lines.append(c.strip())
            kept.append(m)
        elif role == "assistant":
            # her_lines: one entry per earlier reply of hers, in order -- what became of it
            # (the replay's independent oracle rebuilds the payload from these; spec B-3)
            if mode == "drop":
                info["her_lines"].append({"kind": "drop"})
                continue
            if mode == "gist":
                key = _gist_key(last_his, c)
                g = gists.get(key)
                if g:
                    lines.append(_settings["history_her_gist_label"] + g)
                    kept.append({**m, "content": g})          # inplace: the gist IS her message
                    info["gists_used"].append({"key": key[:80], "gist": g})
                    info["her_lines"].append({"kind": "gist", "gist": g})
                    continue
                info["fallbacks"] += 1
                _history_counters["history_gist_fallback_lines"] += 1
                print(f"[proxy] ⚠ HISTORY GIST MISSING -- this one line of hers stays verbatim: "
                      f"{_her_plain(c)[:40]}", flush=True)
                _history_event({"kind": "gist_fallback_line", "profile": name, "key": key[:120]})
                info["her_lines"].append({"kind": "fallback"})
            else:
                info["her_lines"].append({"kind": "verbatim"})
            lines.append(_settings["history_her_label"] + _her_plain(c))
            kept.append(m)
        else:
            lines.append(f"（{role}）{str(c).strip()}")
            kept.append(m)
    info["applied"] = True
    if inplace:
        # v83 B-0 "SUM": no record message; her earlier replies are rewritten where they are.
        body["messages"] = [msgs[0], *kept, *recent, msgs[-1]]
        info["record"] = None
        return True
    head = _settings["history_record_head"].replace("{user}", user)
    tail = _settings["history_record_tail"].replace("{user}", user)
    record = "\n".join([*([head] if head else []), *lines, *([tail] if tail else [])])
    body["messages"] = [msgs[0], {"role": spec.get("record_role", "user"), "content": record}, *recent, msgs[-1]]
    info["record"] = record
    return True


def _history_health() -> dict:
    return {**_history_counters, "history_gist_memory": len(_gist_memory),
            "history_gist_pending": sum(1 for t in _gist_tasks.values() if not t.done())}


# v84 B-1 #4: the reply ceiling (see "reply_cap" in _settings).
_REPLY_CAP_PATH = Path(os.environ.get("LLM_PROXY_REPLY_CAP_LOG", str(HERE / "reply-cap-events.jsonl")))
_reply_cap_counters = {"reply_cap_applied": 0, "reply_cap_hits": 0, "length_finishes": 0}
_reply_cap_last = {"v": None}


def _apply_reply_cap(body: dict) -> int:
    """Put the ceiling on a chat request. Returns the value injected, 0 if none.

    Chat only (reasoning_effort "none" in the forwarded body); a caller's own limit wins."""
    if _settings.get("reply_cap") != "on":
        return 0
    if "max_tokens" in body or "max_completion_tokens" in body:
        return 0
    if body.get("reasoning_effort") != "none":
        return 0
    cap = int(_settings.get("reply_cap_tokens") or 0)
    if cap <= 0:
        return 0
    body["max_tokens"] = cap
    return cap


def _reply_cap_note(seq: int, cap: int, finish: str | None, usage: dict | None, reply: str, model: str) -> dict | None:
    """Count every finish_reason "length"; a hit on OUR ceiling is loud (print + jsonl)."""
    if cap:
        _reply_cap_counters["reply_cap_applied"] += 1
    if finish != "length":
        return None
    _reply_cap_counters["length_finishes"] += 1
    done = (usage or {}).get("completion_tokens")
    hit = bool(cap) and (done is None or done >= cap)
    rec = {"iso": datetime.now().isoformat(timespec="seconds"), "seq": seq, "kind": "reply_cap_hit" if hit else "length_other",
           "cap": cap or None, "completion_tokens": done, "prompt_tokens": (usage or {}).get("prompt_tokens"),
           "model": model, "chars": len(reply), "head": reply[:80], "tail": reply[-80:]}
    if hit:
        _reply_cap_counters["reply_cap_hits"] += 1
        _reply_cap_last["v"] = rec
        print(f"[proxy] 🔴 REPLY HIT THE CEILING (seq {seq}): {done} tokens >= cap {cap}, model {model} -- "
              f"the reply was cut; see {_REPLY_CAP_PATH.name}", flush=True)
    else:
        print(f"[proxy] ⚠ reply stopped on length without our ceiling (seq {seq}): completion {done}, "
              f"prompt {(usage or {}).get('prompt_tokens')} -- context full or the caller's own limit", flush=True)
    try:
        with _REPLY_CAP_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[proxy] reply-cap event write failed: {e}", flush=True)
    return rec


def _reply_cap_health() -> dict:
    return {**_reply_cap_counters, "reply_cap_last": _reply_cap_last["v"]}


# v89 B-2: thinking-on requests and a model that cannot think (see `think_model` in _settings).
_THINK_ROUTE_PATH = Path(os.environ.get("LLM_PROXY_THINK_ROUTE_LOG", str(HERE / "think-route-events.jsonl")))
_think_route_counters = {"think_routed": 0}
_think_route_last = {"v": None}


def _thinking_reason(body: dict, path: str) -> str:
    """Why the FORWARDED body would make the model reason ("" = it would not).

    Deliberately wide -- a false positive only sends one request to the original model, a false
    negative is an empty reply nobody hears about:
      * Ollama native (`api/chat`, `api/generate`): the `think` field decides; absent = the model's
        default, which for qwen3.5 is thinking.
      * OpenAI-compatible: `reasoning_effort` other than "none", or absent (thinking=on globally
        leaves it out); and `reasoning: {effort: ...}` other than none -- the AIRI Minecraft service
        sends that shape on every call (`services/minecraft/src/cognitive/conscious/llm-agent.ts`),
        and with "none" injected next to it which one Ollama obeys is not established."""
    p = path.lstrip("/")
    if p.startswith("api/"):
        t = body.get("think")
        if t is False or (isinstance(t, str) and t.strip().lower() in ("false", "none", "off")):
            return ""
        return f"native think={t!r}" if "think" in body else "native, no think field (model default)"
    r = body.get("reasoning")
    if isinstance(r, dict) and str(r.get("effort", "")).strip().lower() not in ("", "none"):
        return f"reasoning.effort={r.get('effort')}"
    if "reasoning_effort" not in body:
        return "no reasoning_effort (model default)"
    e = body.get("reasoning_effort")
    if str(e).strip().lower() != "none":
        return f"reasoning_effort={e}"
    return ""


def _apply_think_route(body: dict, path: str) -> dict | None:
    """Send a thinking-on request to `think_model` when the override model cannot think.
    Only when the model about to be used IS the override (a caller that names its own model
    with model_override off is not ours to redirect). Returns the event, or None."""
    to = _settings.get("think_model") or ""
    ov = _settings.get("model_override") or ""
    if not to or not ov or body.get("model") != ov or to == ov:
        return None
    why = _thinking_reason(body, path)
    if not why:
        return None
    body["model"] = to
    _think_route_counters["think_routed"] += 1
    rec = {"t": datetime.now().isoformat(timespec="seconds"), "seq": _seq + 1, "path": "/" + path.lstrip("/"),
           "from": ov, "to": to, "why": why}
    _think_route_last["v"] = rec
    print(f"[proxy] THINKING-ON request routed {ov} -> {to} ({why})", flush=True)
    try:
        with _THINK_ROUTE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[proxy] think-route event write failed: {e}", flush=True)
    return rec


def _think_route_health() -> dict:
    return {**_think_route_counters, "think_route_last": _think_route_last["v"]}


def _rewrite_all(raw: bytes, hist=None, hist_info=None, cap_info=None, route_info=None,
                 path: str = "v1/chat/completions") -> tuple[bytes, bool, bool, bool, bool, str, int, int, str, str]:
    """Apply every enabled rewrite. Returns body, reasoning-injected,
    tools-stripped, fewshot-converted, sampling-injected, model-overridden,
    injection-chars-stripped, markers-restored, anchor-seed-text,
    format-rules-mode-applied (v61).
    v83: `hist` = (profile name, spec, gists) turns the history layer on for this call;
    `hist_info` (a dict) receives what it did. Both default to None = byte-identical v82.
    v84: `cap_info` (a dict) receives {"cap": value injected or 0} -- the reply ceiling."""
    payload, injected = _rewrite(raw)
    stripped = False
    converted = False
    sampled = False
    overridden = ""
    inj_removed = 0
    restored = 0
    seeded = ""
    formatted = ""
    try:
        body = json.loads(payload)
    except Exception:
        body = None
    if isinstance(body, dict):
        # Watch the tool list on every request, including the resting state where
        # no switch is on — that is exactly when a new tool would slip in unseen.
        _note_tools(body)
    active = (_settings["strip_tools"] == "on" or _settings["fewshot_messages"] == "on"
              or _settings["sampling"] == "on" or _settings["model_override"]
              or _settings["strip_injections"] == "on" or _settings["strip_widget"] == "on"
              or _settings["restore_markers"] == "on"
              or _settings["user_persona"] or _settings["user_name_prefix"] or _settings["user_frame"]
              or _settings["anchor_seed"] or _settings.get("format_rules", "card") != "card"
              or bool(hist and hist[1]) or _settings.get("reply_cap") == "on")
    if not active or not isinstance(body, dict):
        return payload, injected, stripped, converted, sampled, overridden, inj_removed, restored, seeded, formatted
    if _settings["strip_tools"] == "on" and body.get("tools"):
        body.pop("tools", None)
        # `tool_choice` without `tools` is a protocol error for some backends,
        # so it goes with them rather than being left dangling.
        body.pop("tool_choice", None)
        stripped = True
    # Peel the wrappers before the few-shot extraction: both edit messages[0]
    # and the extraction's block markers sit inside the card content, which
    # the injection strip never touches (exact-match on non-card text only).
    if _settings["strip_injections"] == "on":
        inj_removed = _strip_injections(body)
    # v73 task D: the narrow version of the same peel. Counted into the same
    # field so nothing downstream changes shape; with strip_injections off, any
    # nonzero value came from here. No-op if strip_injections already took it.
    if _settings["strip_widget"] == "on":
        inj_removed += _strip_widget(body)
    # Restore before the few-shot insertion: restore targets AIRI's stored
    # (stripped) assistant turns; the few-shot assistant examples already
    # carry markers and are skipped by the "<|" guard either way.
    if _settings["restore_markers"] == "on":
        restored = _restore_markers(body)
    # v51 user identity: prefix the REAL user turns before the few-shot
    # conversion (which re-inserts these same dict objects after the examples),
    # so example turns stay unlabelled and only what the person typed is named.
    persona_applied = _apply_user_identity(body)
    # v83 task B: the history layer runs on AIRI's real turns only -- after the identity
    # prefix (his lines keep 「User：」 wherever they end up) and before the few-shot lift
    # (the demos are inserted afterwards, untouched, in front of whatever this produced).
    hist_applied = False
    if hist and hist[1]:
        hist_applied = _apply_history_layer(body, hist[0], hist[1], hist[2] or {},
                                            hist_info if hist_info is not None else {})
    if _settings["fewshot_messages"] == "on":
        converted = _convert_fewshot(body)
    # v61 task B: the technical-control block moves (or shrinks, or goes) after
    # the lift and before the persona paragraph closes the system text.
    formatted = _apply_format_rules(body)
    if persona_applied and _settings["user_persona"] and _settings.get("user_persona_position", "tail") != "head":
        _append_user_persona(body)
    # v52: frame sentence and/or persona BEFORE the card text (after the few-shot
    # extraction, so the head lands on the card text and not on the demo block).
    if persona_applied:
        _prepend_system_head(body)
    # v54 task B: the candidate anchor goes in last of the message rewrites, so
    # "system" lands after the persona paragraph and "tail" sits next to the
    # newest user turn no matter what else moved.
    seeded = _apply_anchor_seed(body)
    if _settings["sampling"] == "on":
        # A caller-supplied value would win, same policy as reasoning_effort --
        # but AIRI sends neither field (verified from captured payloads).
        if "temperature" not in body:
            body["temperature"] = _settings["sampling_temperature"]
            sampled = True
        if "top_p" not in body:
            body["top_p"] = _settings["sampling_top_p"]
            sampled = True
    if _settings["model_override"] and body.get("model") != _settings["model_override"]:
        body["model"] = _settings["model_override"]
        overridden = _settings["model_override"]
    # v89 B-2: a thinking-on request does not go to an override model that cannot think.
    routed = _apply_think_route(body, path)
    if route_info is not None:
        route_info["route"] = routed
    # v84 B-1 #4: the reply ceiling goes on last, on chat requests only (see _apply_reply_cap).
    capped = _apply_reply_cap(body)
    if cap_info is not None:
        cap_info["cap"] = capped
    if not (stripped or converted or sampled or overridden or inj_removed or restored or persona_applied or seeded or formatted
            or hist_applied or capped or routed):
        return payload, injected, stripped, converted, sampled, overridden, inj_removed, restored, seeded, formatted
    # ensure_ascii=False matters here: re-encoding Chinese as \uXXXX escapes
    # inflated the forwarded body by ~40% (19KB -> 27KB) for no benefit. Token
    # count is unaffected, but the wire cost is real.
    return (json.dumps(body, ensure_ascii=False).encode("utf-8"),
            injected, stripped, converted, sampled, overridden, inj_removed, restored, seeded, formatted)


class SseProbe:
    """Passive reader that finds the first *visible* frame in an SSE stream.

    Ollama's OpenAI-compatible endpoint frames each token as
    `data: {"choices":[{"delta":{...}}]}`. A frame can carry a role with empty
    content, a reasoning delta, or real text, and only the last of those is
    something the user can see. This records when each first appears plus the
    frames that preceded the visible one, verbatim — a summary would hide
    exactly the thing worth looking at.

    Stops all work once visible content is seen, so the cost is bounded to the
    head of the response.
    """

    LIMIT = 64  # frames kept before the first visible one; a runaway is a finding

    def __init__(self):
        self.buf = b""
        self.done = False
        self.t_first_visible = None
        self.t_first_reasoning = None
        self.frames_before_visible = 0
        self.pre_visible = []          # raw JSON of frames with nothing visible
        self.first_visible_text = None

    def feed(self, chunk: bytes, now: float):
        if self.done:
            return
        self.buf += chunk
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                self.done = True
                return
            try:
                obj = json.loads(data)
                delta = (obj.get("choices") or [{}])[0].get("delta") or {}
            except Exception:
                continue
            reasoning = delta.get("reasoning") or delta.get("reasoning_content") or ""
            content = delta.get("content") or ""
            if reasoning and self.t_first_reasoning is None:
                self.t_first_reasoning = now
            if content:
                self.t_first_visible = now
                self.first_visible_text = content
                self.done = True
                return
            self.frames_before_visible += 1
            if len(self.pre_visible) < self.LIMIT:
                self.pre_visible.append(data.decode("utf-8", "replace"))


def _capture_reply(seq: int, meta: dict, sampled: bool) -> None:
    """v80 B-3: write `payload-NNN.reply.json` beside `payload-NNN.json` -- the raw
    text the model streamed (markers intact, before marker_fix), Ollama's usage
    frame, and the sampling values this request was sent with. The capture-dir
    seq scan only counts plain `payload-NNN.json`, so numbering is unaffected."""
    try:
        d = Path(CAPTURE_DIR)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"payload-{seq:03d}.reply.json").write_text(json.dumps({
            "seq": seq,
            "iso_done": datetime.now().isoformat(timespec="seconds"),
            "text": meta.get("text", ""),
            "usage": meta.get("usage"),
            "sampling_values": ({"temperature": _settings["sampling_temperature"],
                                 "top_p": _settings["sampling_top_p"]} if sampled else None),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:
        print(f"[proxy] reply capture failed: {e}", flush=True)


async def _forward_stream(request: Request, path: str):
    global _seq

    clock = Clock()
    t_recv = clock.now()
    raw = await request.body()
    # v83 task B: which history profile this request gets (header beats /config), and the
    # gists it needs -- awaited here, before forwarding, so the wait is measured (proxy_in_s).
    hist_name, hist_spec, hist_prep, hist_info = "off", None, {"gists": {}, "wait_ms": 0, "missing": 0}, {}
    if path.endswith("chat/completions"):
        hist_name, hist_spec = _history_resolve(request.headers.get("x-history-profile")
                                                or _settings.get("history_profile", "off"))
        if hist_spec:
            _history_counters["history_requests"] += 1
            hist_prep = await _history_prepare(raw, hist_spec)
    cap_info = {"cap": 0}
    route_info = {"route": None}
    payload, injected, stripped, fewshot, sampled, overridden, inj_removed, restored, seeded, formatted = _rewrite_all(
        raw, hist=(hist_name, hist_spec, hist_prep["gists"]), hist_info=hist_info, cap_info=cap_info,
        route_info=route_info, path=path)
    hist_applied = bool(hist_info.get("applied"))
    if hist_applied:
        _history_counters["history_transformed"] += 1

    if CAPTURE_DIR and path.endswith("chat/completions"):
        # Replaying a captured body offline is the only way to get many samples
        # of AIRI's real prompt without driving the browser for each one.
        try:
            d = Path(CAPTURE_DIR)
            d.mkdir(parents=True, exist_ok=True)
            (d / f"payload-{_seq + 1:03d}.json").write_bytes(raw)
            # When a rewrite actually restructured the messages, the forwarded
            # body is the evidence that matters (principle 4: verify on the
            # path the system takes), so keep it next to the original.
            if fewshot:
                (d / f"payload-{_seq + 1:03d}.converted.json").write_bytes(payload)
            # v56: the body Ollama actually receives, every time -- the round is
            # about splitting the 776 ms between llm_req and the first byte, and
            # replaying anything other than the forwarded body would be measuring
            # a prompt production never sends (principle 3). `.converted.json`
            # stays exactly as it was: v54's runner asserts on it.
            (d / f"payload-{_seq + 1:03d}.forwarded.json").write_bytes(payload)
            # v83 task B: which history this request got and what the record looked like --
            # so every replayed / real request can be asserted afterwards (spec B-3).
            if hist_spec:
                (d / f"payload-{_seq + 1:03d}.history.json").write_text(json.dumps({
                    "profile": hist_name, "spec": hist_spec, "applied": hist_applied,
                    "wait_ms": hist_prep.get("wait_ms"), "gists_missing": hist_prep.get("missing"),
                    **hist_info}, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception as e:
            print(f"[proxy] capture failed: {e}", flush=True)

    headers = {k: v for k, v in request.headers.items() if k.lower() not in _SKIP_REQ}
    url = f"{UPSTREAM}/{path}"

    # v84 E: the blind-chat wait (see "pad_ms"). Chat requests only; lands in proxy_in_s.
    padded = 0
    if int(_settings.get("pad_ms") or 0) > 0 and path.endswith("chat/completions"):
        try:
            is_chat_req = json.loads(payload).get("reasoning_effort") == "none"
        except Exception:
            is_chat_req = False
        if is_chat_req:
            padded = int(_settings["pad_ms"])
            await asyncio.sleep(padded / 1000)

    _seq += 1
    seq = _seq
    t_send = clock.now()

    assert _client is not None
    req = _client.build_request(request.method, url, content=payload, headers=headers,
                                params=request.query_params)
    upstream = await _client.send(req, stream=True)
    t_first = {"v": None}
    probe = SseProbe()
    # v27: keep the whole stream so the raw reply (markers intact) can be
    # remembered for restore_markers and counted for the ACT health window.
    # Bytes are appended after each yield; parsing happens once in finally.
    full = bytearray()
    is_chat = path.endswith("chat/completions")

    def _reply_meta() -> dict:
        """Text, structured tool-call names, and the usage frame (AIRI asks for
        include_usage, so Ollama's own prompt_tokens rides on the last chunk =
        the on-path prompt size, principle 4) from the finished stream."""
        text, names, usage, finish = [], [], None, None
        for line in bytes(full).split(b"\n"):
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            if isinstance(obj.get("usage"), dict):
                usage = obj["usage"]
            delta = (obj.get("choices") or [{}])[0].get("delta") or {}
            c = delta.get("content")
            if c:
                text.append(c)
            fr = (obj.get("choices") or [{}])[0].get("finish_reason")
            if fr:
                finish = fr
            for tc in delta.get("tool_calls") or []:
                n = ((tc or {}).get("function") or {}).get("name")
                if n:
                    names.append(n)
        return {"text": "".join(text), "tool_calls": names, "usage": usage, "finish_reason": finish}

    # v73b task 3: the in-stream marker repair (SseMarkerRewriter, marker_fix.py).
    # It sits between upstream and AIRI so a malformed marker never reaches AIRI-s
    # reasoning filter, which drops from there to the end of the reply (41 whole
    # replies lost so far). The SSE layer lives in marker_fix.py on purpose: that is
    # the risky half, and a closure in here could not be unit-tested.
    #
    # THE COMMON PATH IS UNTOUCHED: a healthy frame is forwarded byte for byte.
    # IT CAN ONLY DEGRADE TO PASS-THROUGH: any exception flips it off for the rest
    # of the reply. Losing the repair is acceptable; losing the stream is not.
    rw = (SseMarkerRewriter() if (is_chat and _settings.get("fix_markers") == "on") else None)

    async def body_iter():
        try:
            async for chunk in upstream.aiter_raw():
                now = clock.now()
                if t_first["v"] is None:
                    t_first["v"] = now
                # Yield before inspecting: AIRI's first TTS request fires off this
                # stream, so nothing may sit in front of the consumer.
                yield (rw.feed(chunk) if rw is not None else chunk)
                probe.feed(chunk, now)
                if is_chat:
                    full.extend(chunk)
            if rw is not None:
                tail = rw.flush()
                if tail:
                    yield tail
        finally:
            await upstream.aclose()
            t_done = clock.now()
            reply_has_act = None
            swallow_flags = None
            identity_flags = None
            tool_flags = None
            usage = None
            finish = None
            cap_event = None
            if is_chat:
                try:
                    meta = _reply_meta()
                    reply = meta["text"]
                    usage = meta["usage"]
                    finish = meta.get("finish_reason")
                    # v84 B-1 #4: a reply that ran into the ceiling (or any length stop) is loud.
                    try:
                        _mdl = json.loads(payload).get("model", "")
                    except Exception:
                        _mdl = ""
                    cap_event = _reply_cap_note(seq, cap_info.get("cap", 0), finish, usage, reply, _mdl)
                    # v80 B-3: with capture on, keep what she actually said next to
                    # the payload that asked for it, so a real chat can be read back
                    # line by line afterwards. Capture-only; the stream is untouched.
                    if CAPTURE_DIR:
                        _capture_reply(seq, meta, sampled)
                    reply_has_act = bool(re.search(r"<\|ACT\s", reply))
                    _record_act(reply_has_act)
                    _remember_raw_reply(reply)
                    # v83 task B: start her gist now, off the critical path (TTS is speaking);
                    # the next request awaits the same task if it is not done yet.
                    if hist_spec and hist_spec.get("her_older") == "gist" and reply.strip():
                        his_now = _last_user_text(payload)
                        gkey = _gist_key(his_now, reply)
                        if gkey not in _gist_memory:
                            _gist_for(gkey, his_now, reply)
                    swallow_flags = _swallow_watch(reply, seq)
                    if rw is not None:
                        _note_markerfix(seq, rw.events)
                        if rw.error:
                            _markerfix["failed"] += 1
                            print(f"[proxy] MARKER FIXER FELL BACK TO PASS-THROUGH "
                                  f"(seq {seq}): {rw.error}", flush=True)
                    identity_flags = _identity_watch(reply, seq, _last_user_text(payload))
                    tool_flags = _tool_watch(meta["tool_calls"], reply, seq, _tool_names_in(payload))
                except Exception as e:
                    print(f"[proxy] reply bookkeeping failed: {e}", flush=True)
            first = t_first["v"] or t_done
            vis = probe.t_first_visible
            _log({
                "seq": seq,
                "path": "/" + path,
                "status": upstream.status_code,
                "thinking": _settings["thinking"],
                "reasoning_effort_injected": injected,
                "tools_stripped": stripped,
                "fewshot_converted": fewshot,
                "sampling_injected": sampled,
                "injection_chars_stripped": inj_removed,
                "markers_restored": restored,
                "anchor_seed": seeded or None,
                "anchor_seed_position": (_settings["anchor_seed_position"] if seeded else None),
                # v61: which format_rules mode actually rewrote this request ("" = card / no headers)
                "format_rules": formatted or None,
                # v83 task B: the history layer (None = off for this request)
                "history_profile": hist_name if hist_spec else None,
                "history_applied": hist_applied if hist_spec else None,
                "history_n_older": hist_info.get("n_older") if hist_spec else None,
                "history_fallbacks": hist_info.get("fallbacks") if hist_spec else None,
                "history_wait_ms": hist_prep.get("wait_ms") if hist_spec else None,
                "reply_has_act": reply_has_act,
                "swallowed": swallow_flags,
                "markerfix": ([{"kind": e["kind"], "what": e["what"], "why": e["why"]}
                               for e in rw.events] if (rw is not None and rw.events) else None),
                "identity": identity_flags,
                "tools": tool_flags,
                # v53: Ollama's own count of the prompt as it filled the context (usage frame)
                "prompt_tokens": (usage or {}).get("prompt_tokens"),
                "completion_tokens": (usage or {}).get("completion_tokens"),
                "sampling_values": ({"temperature": _settings["sampling_temperature"],
                                     "top_p": _settings["sampling_top_p"]} if sampled else None),
                "model_overridden": overridden or None,
                # v84 B-1 #4: the ceiling this request carried (None = none) and how the reply ended
                "reply_cap": cap_info.get("cap") or None,
                "finish_reason": finish,
                "reply_cap_event": (cap_event or {}).get("kind"),
                "pad_ms": padded or None,
                # v89 B-2: a thinking-on request sent to think_model instead of the override (None = not routed)
                "think_routed": (route_info.get("route") or {}).get("to"),
                "req_bytes": len(payload),
                "orig_bytes": len(raw),
                "t_recv": t_recv, "t_send": t_send, "t_first_byte": first, "t_done": t_done,
                "iso_recv": _iso(t_recv),
                "proxy_in_s": round(t_send - t_recv, 4),
                "upstream_first_byte_s": round(first - t_send, 4),
                "total_s": round(t_done - t_recv, 4),
                # v17 first-visible-character breakdown
                "first_visible_s": round(vis - t_send, 4) if vis else None,
                "byte_to_visible_s": round(vis - first, 4) if vis else None,
                "first_reasoning_s": (round(probe.t_first_reasoning - t_send, 4)
                                      if probe.t_first_reasoning else None),
                "frames_before_visible": probe.frames_before_visible,
                "pre_visible_frames": probe.pre_visible,
                "first_visible_text": probe.first_visible_text,
            })
            print(f"[proxy] #{seq} {path} inject={injected} "
                  f"proxy_in {t_send - t_recv:.4f}s  first_byte {first - t_send:.3f}s  "
                  f"first_visible {(vis - t_send) if vis else float('nan'):.3f}s "
                  f"(+{probe.frames_before_visible} blind frames)  "
                  f"total {t_done - t_recv:.3f}s", flush=True)

    out_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in _SKIP_RES}
    return StreamingResponse(body_iter(), status_code=upstream.status_code,
                             headers=out_headers,
                             media_type=upstream.headers.get("content-type"))


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _forward_stream(request, "v1/chat/completions")


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"])
async def passthrough(request: Request, path: str):
    """Everything else — /v1/models, /api/tags, the model list AIRI polls — goes
    straight through so the provider keeps validating normally."""
    if request.method in ("GET", "HEAD", "OPTIONS"):
        assert _client is not None
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _SKIP_REQ}
        r = await _client.request(request.method, f"{UPSTREAM}/{path}",
                                  headers=headers, params=request.query_params)
        out = {k: v for k, v in r.headers.items() if k.lower() not in _SKIP_RES}
        return Response(content=r.content, status_code=r.status_code, headers=out,
                        media_type=r.headers.get("content-type"))
    return await _forward_stream(request, path)


if __name__ == "__main__":
    port = int(os.environ.get("LLM_PROXY_PORT", "11435"))
    print(f"[proxy] starting, thinking={_settings['thinking']}, upstream={UPSTREAM}", flush=True)
    sys.stdout.flush()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
