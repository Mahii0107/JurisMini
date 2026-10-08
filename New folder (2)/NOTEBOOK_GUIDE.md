# Legal NER on Indian Judgments: Notebook Guide

This guide explains `Indian_Legal_NER_SLM_Comparison.ipynb` end to end:
- the dataset and its 14 entity types
- the three small language models that were compared
- how each model is trained and scored
- the follow-up work: word-boundary snapping, relaxed metrics, bidirectional attention
- how to use the final model for inference, in the notebook, in the Streamlit app, or from your own Python code

**Final result.** TinyLlama-1.1B with bidirectional attention and word snapping reaches **exact-match F1 0.802** on the official 4,501-document test set. The first version of the notebook got 0.644.

---

## Contents

1. [The task](#1-the-task)
2. [The dataset and its entities](#2-the-dataset-and-its-entities)
3. [The models](#3-the-models)
4. [Concepts used throughout](#4-concepts-used-throughout)
5. [Walkthrough of the notebook](#5-walkthrough-of-the-notebook)
6. [Results](#6-results)
7. [Using the final model for inference](#7-using-the-final-model-for-inference)
8. [Files and folders](#8-files-and-folders)
9. [Re-running and reproducing](#9-re-running-and-reproducing)

---

## 1. The task

**Named entity recognition (NER)** means finding the spans of text that name something and labelling each one with a type. Here is an example:

> In the **High Court of Judicature at Bombay** `COURT`, **Criminal Appeal No. 452 of 2019** `CASE_NUMBER`, the appellant **Ramesh Kumar** `PETITIONER` challenged the order dated **12th March 2018** `DATE` …

The notebook treats NER as **token classification**. The model reads the text and gives every token a tag. The tags are then joined back into character spans.

---

## 2. The dataset and its entities

**InLegalNER** comes from OpenNyAI (Kalamkar et al., 2022, *"Named Entity Recognition in Indian court judgments"*). It is hosted on Hugging Face as `opennyaiorg/InLegalNER`, and it is gated: you have to accept its terms on the dataset page before you can download it.

The documents come from Indian court judgments and are of two kinds:
- **Sentences** from the body of a judgment.
- **Preambles**: the header of a judgment, which lists the court, the parties, the lawyers, the judges and the case number. Preambles are much longer than sentences.

The data is in Label Studio format. Each record has the text in `data.text` and the entities in `annotations[].result[].value`, as `start` and `end` character offsets plus a `labels` field.

### Splits

| Split | Documents | Entities | How the notebook uses it |
|---|---|---|---|
| `train` | 10,995 | 29,964 | Split 90/10. 9,896 documents are used for training and 1,099 for validation, which picks the best epoch. |
| `dev` | 1,074 | 3,261 | Added to training, since the test split has labels and `dev` is not needed for scoring. That gives 10,970 training documents. |
| `test` | 4,501 | 13,365 | The **official test set**. Every score in this guide comes from it. |

If the Hub `test` split ever came without labels, the code would fall back to using `dev` as the test set.

### The 14 entity types

The counts below are annotation counts in `train` and `test`.

| Entity | Meaning | Example | Train | Test |
|---|---|---|---|---|
| `COURT` | A court of law | *High Court of Judicature at Bombay*, *Sessions Judge, Pune* | 2,367 | 1,221 |
| `JUDGE` | A judge hearing the case | *D.Y. Chandrachud*, *Suresh Kumar Kait* | 2,325 | 590 |
| `PETITIONER` | The party who filed the case (petitioner, appellant, applicant, complainant) | *Ramesh Kumar* | 3,068 | 862 |
| `RESPONDENT` | The party answering the case (respondent, opposite party, accused) | *State of Maharashtra*, *Princess Manolya Jah* | 3,862 | 1,113 |
| `LAWYER` | An advocate or counsel appearing for a party | *Kapil Sibal*, *D.Prakash Reddy* | 3,505 | 1,798 |
| `WITNESS` | A person examined as a witness | *PW-1 Ram Singh* (the name part) | 881 | 413 |
| `OTHER_PERSON` | Any other person who is not one of the roles above | *Tarlochan Singh* | 2,653 | 1,109 |
| `ORG` | An organisation that is not a court | *Reserve Bank of India*, *CBI* | 1,441 | 914 |
| `GPE` | A geopolitical entity: a country, state, city or village | *Kerala*, *Hyderabad* | 1,398 | 721 |
| `STATUTE` | An act, code or law | *Negotiable Instruments Act, 1881*, *IPC* | 1,804 | 975 |
| `PROVISION` | A section, article, rule or clause of a statute | *Section 138*, *Article 21* | 2,384 | 1,220 |
| `PRECEDENT` | A cited earlier judgment, with its party names and/or citation | *K. Bhaskaran v. Sankaran Vaidhyan Balan, (1999) 7 SCC 510* | 1,351 | 650 |
| `CASE_NUMBER` | The number of a case | *Criminal Appeal No. 452 of 2019* | 1,040 | 667 |
| `DATE` | A date | *12th March 2018*, *27.02.2018* | 1,885 | 1,112 |

These labels are hard in two ways:
- **Roles matter more than names.** The same name can be a PETITIONER, a RESPONDENT, a LAWYER, a WITNESS or an OTHER_PERSON, depending on where it appears and what surrounds it.
- **Long spans with internal punctuation.** PRECEDENT, CASE_NUMBER and ORG spans are long and full of punctuation, so getting the exact boundaries right is difficult.

### BIO tags

The model predicts one tag per token, using the **BIO scheme**. `B-X` marks the beginning of an entity of type X, `I-X` marks a token inside that entity, and `O` marks a token outside any entity. With 14 types that makes 29 tags (`O` plus a `B-` and an `I-` tag for each type). For example:

```
Ramesh   Kumar   challenged
B-PETITIONER  I-PETITIONER  O
```

---

## 3. The models

All three are **decoder-only small language models (SLMs)**, the same family as GPT-style chat models, only smaller. They were pretrained to predict the next token. Here each one is turned into a token classifier.

| | TinyLlama-1.1B | Qwen2.5-1.5B | Phi-3-mini |
|---|---|---|---|
| Hugging Face checkpoint | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `Qwen/Qwen2.5-1.5B` | `microsoft/Phi-3-mini-4k-instruct` |
| Parameters | 1.1 B | 1.5 B | 3.8 B |
| Architecture | Llama 2 | Qwen2 | Phi-3 |
| Layers / hidden size | 22 / 2,048 | 28 / 1,536 | 32 / 3,072 |
| Attention heads (query / key-value) | 32 / 4 (grouped-query) | 12 / 2 (grouped-query) | 32 / 32 (multi-head) |
| Vocabulary | 32,000 (SentencePiece BPE) | 151,936 (byte-level BPE) | 32,064 (SentencePiece BPE) |
| Context length | 2,048 | 32k (config allows up to 131k) | 4,096 |
| Variant used | Chat fine-tune | Base model | Instruct fine-tune |
| LoRA target modules | q, k, v, o, gate, up, down | q, k, v, o, gate, up, down | qkv (fused), o, gate_up (fused), down |
| Trainable parameters | 12.7 M (1.2 %) | 18.5 M (1.2 %) | 25.3 M (0.7 %) |

- **TinyLlama-1.1B.** An open model with the Llama 2 architecture and tokenizer, pretrained on about 3 trillion tokens. It is small and fast, and it turned out to be the best of the three.
- **Qwen2.5-1.5B.** From Alibaba's Qwen team, pretrained on a much larger multilingual corpus. Its large byte-level vocabulary merges a lot of punctuation and whitespace into single tokens, such as `,Umesh`. Those merged tokens often cross entity boundaries, which caps how well it can do on character-exact spans. Its tokenizer ceiling is 0.88, against 0.98 for the other two (the ceiling is explained in [section 4](#tokenizer-ceiling)).
- **Phi-3-mini (3.8B).** From Microsoft, trained heavily on synthetic and filtered "textbook-quality" data. It is the largest of the three and the slowest to train (about 62 min against about 25 min), and on this task it does not beat TinyLlama.

Gemma-2-2B can be swapped in through the config cell. It needs bf16, and it runs with eager attention.

---

## 4. Concepts used throughout

### Token classification head

A language model normally ends with a head that predicts the next word from its vocabulary. For NER that head is dropped (it shows up as `lm_head.weight | UNEXPECTED` in the load report). In its place is a small new linear layer called `score`, which maps each token's final hidden state to the 29 BIO tags. `AutoModelForTokenClassification` builds this layer automatically.

### QLoRA: 4-bit base model plus LoRA adapters

Fine-tuning every weight of a billion-parameter model will not fit in 8 GB of GPU memory, so the notebook uses **QLoRA**, which combines two ideas:
- **4-bit quantisation.** The frozen base weights are stored as 4-bit NF4 numbers, with double quantisation, through `bitsandbytes`. The computation itself still runs in bf16.
- **LoRA (Low-Rank Adaptation).** Next to each targeted weight matrix W, two small trainable matrices A and B are added (rank r = 16, alpha = 32, dropout 0.05). The layer's output becomes `W·x + (alpha/r)·B·A·x`. Only A and B are trained, along with the `score` head, which stays in full precision (`llm_int8_skip_modules=["score"]`).

This trains about 1% of the parameters. The result is a small **adapter** of about 50 MB, which is saved separately and loaded on top of the original base model at inference time.

`prepare_model_for_kbit_training` also turns on **gradient checkpointing**. Instead of keeping every layer's activations in memory, it recomputes them during the backward pass, trading some compute for a lot of memory.

### Causal vs bidirectional attention

Inside every transformer layer, **self-attention** lets each token look at other tokens. Which tokens it may look at is set by the **attention mask**.

- **Causal (left-to-right) attention** is what decoder-only LLMs are built with. Each token can see only itself and the tokens **before** it. That is essential for generating text one word at a time, but for tagging it means that when the model labels a word, it has not seen anything to the right of it.

  ```
  "Sri D.Prakash Reddy, Advocate for the petitioner"
                 ↑ when tagging "Prakash", the model cannot see "Advocate for the petitioner"
  ```

  This is why the causal models confuse role labels. Whether a name is a LAWYER, a PETITIONER or a WITNESS is often given away by words that come **after** it.

- **Bidirectional attention** is what encoder models such as BERT use. Every token can see **every** real token in the window, both to the left and to the right. A decoder can be switched to this mode, and with fine-tuning it learns to use the extra context.

**Padding** is a separate issue. Batches are padded to the same length with dummy tokens, and in both modes real tokens must never attend to those padding tokens. The bidirectional switch keeps the padding mask in place.

**How the switch is done (transformers 5.x).** Section 7.3 of the notebook contains the source reading behind this. Two changes are needed, and `set_attention_mode(model, bidirectional)` makes both:
1. `config.is_causal = False`. With this set, `masking_utils.create_causal_mask` hands off to `create_bidirectional_mask`, which builds a padding-only mask.
2. `self_attn.is_causal = False` on every attention module. When a batch has no padding, the mask is skipped as an optimisation, and PyTorch's sdpa kernel then falls back to the module's own `is_causal` flag. Without this second switch, unpadded batches would quietly stay causal.

Both are plain attributes, so they carry through 4-bit loading, gradient checkpointing, PEFT/LoRA and sdpa. The model has to be **trained** in bidirectional mode to benefit from it, and the same patch has to be applied again **every time the model is loaded** for inference.

The notebook checks all of this with assert-based tests on the exact training setup:

| Test | Causal | Bidirectional |
|---|---|---|
| Change the **last** token: does the **first** token's hidden state change? | No (difference exactly 0) | Yes (difference ≈ 2.4) |
| Same test through the training graph (gradient from the last token to the first) | 0 | > 0 |
| Change what is inside the padded positions: do real tokens change? | No (exactly 0) | No (exactly 0) |

### Sliding windows

Preambles can be longer than `MAX_LEN = 256` tokens, so documents are cut into **overlapping windows** of 256 tokens with a 64-token overlap (`STRIDE`). At prediction time, tokens that appear in more than one window have their logits (the raw class scores) **averaged** before the tag is chosen. This means every token gets a prediction that has context on both sides of the window edge.

### From tags back to character spans

`tags_to_spans` walks along the predicted tags and joins each `B-X I-X I-X` run into one character span `(start, end, X)`, using each token's character offsets. `clean_spans` then trims trailing spaces and `, ; :` that a token may have pulled across the end of an entity.

### Word-boundary snapping

**The problem.** Subword tokenizers do not always split on word boundaries. `" Sessions"` can become `" S" + "essions"`, and when the model tags only some of the pieces the predicted span starts or ends in the middle of a word. The first notebook run's demos showed four such cases:

| Model output | Should be |
|---|---|
| `essions Judge, Pune` | `Sessions Judge, Pune` |
| `Bhask` + `aran v. …` | `Bhaskaran v. …` |
| `ukkaram Jah Bahadur` | `Mukkaram Jah Bahadur` |
| `Suresh Kumar K` | `Suresh Kumar Kait` |

A span like that scores zero under exact matching, even when the model clearly found the entity.

**The fix.** `snap_spans` is a post-processing step that needs no retraining. It is switched on with `SNAP_TO_WORDS = True`.
1. **Extend cut words.** If a span edge falls inside a word, the edge moves outwards until the word is whole.
2. **Merge pieces.** Spans with the same label that overlap or touch after step 1 are joined into one.
3. **Resolve label conflicts.** If spans with different labels still overlap, the longer one is kept.

**What counts as a word matters.** The first attempt defined a word as any run of non-whitespace characters. On this corpus that rule does damage, because court text glues honorifics and punctuation onto names, and the gold spans leave them out:

```
Sri.[S.Sidhardhan]     Mr.[Gautam Tiwari]     [Joshi],Assistant     (brackets = gold span)
```

The notebook compares three definitions:

| Word rule | Gold spans changed by snapping | Tokenizer ceiling with snapping | TinyLlama exact F1 with snapping |
|---|---|---|---|
| Any non-whitespace run | 1,120 of 13,365 | 0.907 (worse than 0.980 without snapping) | 0.669 |
| Letters and digits together | 90 | 0.983 | 0.704 |
| **Letters with letters, digits with digits** (used) | **39** | **0.987** | **0.704** |

The rule in use only joins a letter to a letter or a digit to a digit. It repairs every cut word in the table above, but it does not cross `.` `,` `/` or a letter-digit boundary. A letter-digit boundary is where a footnote number is glued to a case name, as in `[Mehta v. Union of India]35`.

### Metrics

All scores are computed on **character spans**, so they do not depend on any one tokenizer and the models are compared on equal terms.

| Metric | A prediction counts as correct when… |
|---|---|
| **Exact (micro) P / R / F1** | the start, the end **and** the label all match a gold entity. This is the headline metric. |
| **Macro F1** | (same matching rule) F1 is computed for each of the 14 labels and then averaged, so rare labels count as much as common ones. |
| **Overlap F1** | the prediction overlaps a gold span that has the **same label**. Each gold span can be matched only once. This measures "found the right thing, even if the edges are a little off". |
| **Boundary F1** | the start and end match exactly, whatever the label. This measures segmentation alone. |

<a id="tokenizer-ceiling"></a>**Tokenizer ceiling.** This is the F1 a model would get if every token tag were perfect. It is below 1.0 when a single token straddles an entity boundary, as Qwen's merged punctuation tokens often do. It is an upper bound on what that tokenizer allows, and it is also used to check that snapping does not move correct spans.

---

## 5. Walkthrough of the notebook

| Section | What it does |
|---|---|
| **Intro** | Overview of the dataset, models, splits, metric and runtime. |
| **Setup (first cells)** | Installs packages and sets the configuration: `SEED=42`, `MAX_LEN=256`, `STRIDE=64`, `EPOCHS=2`, `LR=2e-4`, `BATCH=8`, `GRAD_ACC=2` (effective batch 16), LoRA r = 16 / alpha = 32 / dropout 0.05, 4-bit and bf16 on GPU. |
| **1. Hugging Face login** | Logs in to download the gated dataset. The token is cached after the first time. |
| **2. Load and parse** | Loads InLegalNER from the Hub, falling back to OpenNyAI's zip release if that fails, then parses the Label Studio records into `{text, entities: [(start, end, label)]}` and builds the splits. |
| **3. Tokenization** | `encode_docs` tokenizes into overlapping windows and gives each token its BIO label from the character offsets. Special tokens and whitespace-only tokens get `-100`, which means "ignore in the loss". |
| **4. Decoding and metrics** | `tags_to_spans`, `span_prf` (exact P/R/F1, macro F1 and per-label scores), `predict_docs` (prediction with window logits averaged), `clean_spans` and `tokenizer_ceiling`. |
| **5. Model factory** | `load_tokenizer`, `load_base` (4-bit base model with the token-classification head), `build_model` (adds the LoRA adapters) and `make_args` (Trainer settings). |
| **6. Train and evaluate** | `run_experiment` trains each model, keeps the epoch with the best validation F1, saves the adapter to `runs/<model>/best/`, scores the test set and writes `runs/<model>/results.json`. A model that already has results is skipped. |
| **7. Snapping, relaxed metrics, bidirectional attention and final model** | The follow-up work and the final model (detailed below). The original run's comparison, first final-model selection (TinyLlama, F1 0.644, saved to `final_legal_ner_model/`) and demos were removed once this section replaced them. |

### How each model is trained (section 6, and section 7.4 for the ablation)

1. **Tokenize.** Each document is split into windows of 256 tokens with a 64-token overlap. This gives about 13,000 training windows for TinyLlama and Phi-3 and about 11,000 for Qwen.
2. **Build the model.** The base model is loaded in 4-bit, a fresh 29-way `score` head is attached, the model is prepared for k-bit training with gradient checkpointing, and LoRA adapters are added to the attention and MLP projections.
3. **Train** with the Hugging Face `Trainer`:
   - Optimizer: `paged_adamw_8bit`, learning rate 2e-4, cosine schedule with 5% warm-up, weight decay 0.01.
   - Batches: batch size 8 with 2 gradient-accumulation steps, `group_by_length` so that padding is kept low.
   - Loss: cross-entropy over the BIO tags. Ignored tokens (`-100`) don't count.
   - Length: 2 epochs, which is 1,622 optimizer steps for TinyLlama.
4. **Choose the epoch.** After each epoch, entity-level F1 is computed on the validation windows, and the best checkpoint is reloaded at the end (`load_best_model_at_end`, `metric_for_best_model="f1"`).
5. **Score on test.** The model predicts the whole documents with window logits averaged, the tags are decoded to character spans, and the spans are scored against the gold annotations.

| Model | Train time (RTX 5060 Laptop, 8 GB) | Peak GPU memory | Test docs/s |
|---|---|---|---|
| TinyLlama-1.1B | 25 min | 1.8 GB | 41 |
| Qwen2.5-1.5B | 26 min | 3.0 GB | 43 |
| Phi-3-mini-3.8B | 62 min | 5.2 GB | 13 |

### Section 7 in detail

| Part | What it does |
|---|---|
| **7.1 Snapping** | Defines `snap_spans` and its three word rules, and wraps `predict_docs` so that it accepts `snap=` (the default is `SNAP_TO_WORDS`). Includes unit tests built from the failure cases in the demos. |
| **7.2 Relaxed metrics** | Adds `overlap_prf` and `boundary_prf`. It reloads each original adapter, predicts the test set **once** (the raw predictions are cached in `runs_v2/preds_<model>.json`), scores them with snapping off and on, and reports the snapped tokenizer ceiling. A diagnostic table compares the word rules. |
| **7.3 Bidirectional attention** | `set_attention_mode`, `build_model_v2`, `load_base_v2` and `load_for_inference`, together with the assert-based tests on the real 4-bit + PEFT + gradient-checkpointing + sdpa setup. |
| **7.4 Ablation** | `run_experiment_v2(name, ckpt, out_dir, bidirectional, max_steps)` retrains TinyLlama with exactly the original configuration, once causal (`runs_v2/TinyLlama-causal`) and once bidirectional (`runs_v2/TinyLlama-bidir`). It can resume from the last checkpoint (`get_last_checkpoint`), and `max_steps=20` runs a throughput smoke test. The section ends with the comparison table, per-label F1 and a bar chart. |
| **7.5 Final selection** | Picks the best combination of model × attention × snapping by exact F1, with ties broken by macro F1. If it beats the previous final model, it is saved to `final_legal_ner_model_v2/` with a `selection.json`. Then `extract_entities` is redefined to use it, and the demos are rerun. |
| **7.6 Summary** | The key numbers and conclusions. |

---

## 6. Results

All results are on the official test set (4,501 documents, 13,365 entities).

| Configuration | Exact P | Exact R | **Exact F1** | Macro F1 | Overlap F1 | Boundary F1 |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B, causal | .574 | .605 | .589 | .588 | .784 | .607 |
| Qwen2.5-1.5B, causal + snap | .632 | .637 | .635 | .632 | .797 | .660 |
| Phi-3-mini, causal | .556 | .720 | .627 | .614 | .820 | .647 |
| Phi-3-mini, causal + snap | .632 | .771 | .695 | .680 | .841 | .725 |
| TinyLlama, causal (original final model) | .574 | .729 | .643 | .628 | .829 | .663 |
| TinyLlama, causal + snap | .646 | .775 | .704 | .687 | .847 | .735 |
| TinyLlama, bidirectional | .738 | .819 | .777 | .767 | .895 | .794 |
| **TinyLlama, bidirectional + snap (final)** | **.773** | **.833** | **.802** | **.792** | **.903** | **.824** |

The first TinyLlama run re-predicted here scores 0.6428, against 0.6436 in its `results.json`. The tiny difference comes from bf16 rounding with different batch compositions. The causal re-run in `runs_v2/` reproduced the original closely, at 0.644 without snapping and 0.707 with it.

**What the numbers show:**
- **Snapping is free accuracy.** It adds 4.5 to 6.8 exact-F1 points to every model without any retraining. The gain comes mostly from boundaries: boundary F1 rises by 5 to 7 points, while overlap F1 rises by only 1 to 2.
- **Bidirectional attention is the largest single improvement.** It adds 13 exact-F1 points without snapping and 9.5 with it, at the same training cost (about 24 min). The biggest per-label gains are on labels that depend on context to the right or on long spans:

  | Label | Causal (snap on) | Bidirectional (snap on) |
  |---|---|---|
  | PRECEDENT | 0.32 | 0.58 |
  | ORG | 0.41 | 0.67 |
  | OTHER_PERSON | 0.66 | 0.85 |
  | CASE_NUMBER | 0.48 | 0.65 |
  | WITNESS | 0.72 | 0.88 |

- **One weakness remains.** RESPONDENT drops from 0.70 to 0.64 and LAWYER from 0.92 to 0.90. The bidirectional model sometimes swaps PETITIONER and RESPONDENT, two labels that appear in very similar contexts.
- **Bigger is not better here.** Phi-3 (3.8B) trains for 2.5× as long as TinyLlama and is still worse. Qwen is held back by its tokenizer.

---

## 7. Using the final model for inference

The final model is saved in `final_legal_ner_model_v2/`:
- `adapter_model.safetensors` and `adapter_config.json`: the LoRA adapter plus the `score` head.
- The tokenizer files.
- `selection.json`, which records:

```json
{ "selected": "TinyLlama-bidir", "base_checkpoint": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "attention": "bidir", "bidirectional": true, "snap_to_words": true,
  "labels": ["O", "B-CASE_NUMBER", "I-CASE_NUMBER", ...], "max_len": 256, "stride": 64, ... }
```

To get the reported results, inference must follow four rules:
1. Load the **base model**, then the **adapter** on top of it.
2. Apply the **bidirectional patch** after loading. Without it, the model runs causally and its accuracy collapses.
3. Use the same **windowing** (256 tokens, stride 64) and **average the logits**.
4. Apply **snapping** to the predicted spans.

### Option A: inside the notebook

Run the setup and definition cells (the config cell and sections 2, 3, 4 and 5, plus 7.1 and 7.3), then the cells of section 7.5. After that:

```python
extract_entities("Hon'ble Mr. Justice D.Y. Chandrachud heard Mr. Kapil Sibal, learned senior counsel for the State of Kerala.")
# [{'text': 'D.Y. Chandrachud', 'label': 'JUDGE', 'start': 20, 'end': 36}, ...]
```

`load_final()` reads `selection.json`, applies the bidirectional patch if it is recorded there, and sets the snapping default.

### Option B: the Streamlit app

```bash
streamlit run app.py        # then open http://localhost:8501
```

- **Extract entities tab.** Paste text, pick a built-in example or upload a `.txt` judgment. Entities are highlighted inline and listed in a table, and they can be downloaded as JSON or CSV.
- **Sidebar.**
  - The model picker defaults to **★ Final**; the other trained runs are listed below it.
  - A **"Snap spans to whole words"** toggle.
  - The selected model's attention mode and its test scores.
- **Model comparison tab.** Charts and tables for every configuration and for per-label F1.

The app needs only the folders on disk, not the notebook. On first use it loads the model, which takes about 10 seconds. The base model is downloaded the very first time and comes from the cache after that.

### Option C: your own Python code

The script below is self-contained and reproduces the notebook's inference exactly. It needs `transformers>=5`, `peft`, `bitsandbytes`, `accelerate` and `torch`.

```python
import json, torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForTokenClassification, BitsAndBytesConfig
from peft import PeftModel

FINAL = "final_legal_ner_model_v2"
sel = json.load(open(f"{FINAL}/selection.json"))
labels = sel["labels"]; id2label = dict(enumerate(labels))

# 1) tokenizer + 4-bit base model + adapter
tok = AutoTokenizer.from_pretrained(FINAL)
tok.pad_token = tok.pad_token or tok.unk_token; tok.padding_side = "right"
model = AutoModelForTokenClassification.from_pretrained(
    sel["base_checkpoint"], num_labels=len(labels), id2label=id2label,
    label2id={l: i for i, l in id2label.items()}, ignore_mismatched_sizes=True, dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["score"]),
    device_map={"": 0})
model.config.pad_token_id = tok.pad_token_id
model = PeftModel.from_pretrained(model, FINAL).eval()

# 2) bidirectional patch (must be re-applied on every load)
def set_attention_mode(model, bidirectional):
    for m in model.modules():
        if hasattr(getattr(m, "config", None), "_attn_implementation"):
            m.config.is_causal = not bidirectional
        if type(m).__name__.endswith("Attention") and hasattr(m, "is_causal"):
            m.is_causal = not bidirectional
set_attention_mode(model, sel["bidirectional"])

# 3) windowed prediction with logit averaging -> character spans
@torch.no_grad()
def predict(text):
    enc = tok([text], max_length=sel["max_len"], stride=sel["stride"], truncation=True,
              return_overflowing_tokens=True, return_offsets_mapping=True)
    store = defaultdict(lambda: None)
    batch = tok.pad({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]},
                    return_tensors="pt").to(model.device)
    logits = model(**batch).logits.float().cpu()
    for i, lg in enumerate(logits):
        for t, ((s, e), sid) in enumerate(zip(enc["offset_mapping"][i], enc.sequence_ids(i))):
            if sid is None or e <= s: continue
            while s < e and text[s].isspace(): s += 1
            if s < e: store[(s, e)] = lg[t] if store[(s, e)] is None else store[(s, e)] + lg[t]
    spans, cur = [], None
    for (s, e) in sorted(store):
        tag = id2label[int(store[(s, e)].argmax())]
        if tag == "O":
            if cur: spans.append(tuple(cur)); cur = None
            continue
        p, lab = tag.split("-", 1)
        if p == "B" or cur is None or cur[2] != lab:
            if cur: spans.append(tuple(cur))
            cur = [s, e, lab]
        else:
            cur[1] = e
    if cur: spans.append(tuple(cur))
    cleaned = []                                  # clean_spans: drop trailing whitespace , ; :
    for s, e, l in spans:
        while e > s and text[e - 1] in " \t\n,;:": e -= 1
        if e > s: cleaned.append((s, e, l))
    return snap_spans(text, cleaned) if sel["snap_to_words"] else cleaned

# 4) word-boundary snapping ("class" rule)
def snap_spans(text, spans):
    joined = lambda a, b: (a.isalpha() and b.isalpha()) or (a.isdigit() and b.isdigit())
    out = []
    for s, e, l in spans:
        while s > 0 and joined(text[s - 1], text[s]): s -= 1
        while e < len(text) and joined(text[e - 1], text[e]): e += 1
        out.append((s, e, l))
    merged = []
    for l in sorted({x[2] for x in out}):         # merge overlapping/touching same-label spans
        cur = None
        for s, e, _ in sorted(x for x in out if x[2] == l):
            if cur and s <= cur[1]: cur[1] = max(cur[1], e)
            else:
                if cur: merged.append(tuple(cur))
                cur = [s, e, l]
        if cur: merged.append(tuple(cur))
    kept = []                                     # different labels overlapping -> keep the longer
    for s, e, l in sorted(merged, key=lambda x: (-(x[1] - x[0]), x[0])):
        if all(e <= ks or s >= ke for ks, ke, _ in kept): kept.append((s, e, l))
    return sorted(kept)

text = "The respondent relied on Section 138 of the Negotiable Instruments Act, 1881."
print([(text[s:e], l) for s, e, l in predict(text)])
# [('Section 138', 'PROVISION'), ('Negotiable Instruments Act, 1881', 'STATUTE')]
```

**Notes:**
- **Long documents.** The script sends all windows of a document as one batch. For very long judgments, process the windows in chunks of about 16, as `predict_docs` in the notebook does.
- **Without a GPU.** Drop `quantization_config` and `device_map` and use `dtype=torch.float32`. It works, but slowly.
- **Hardware needed.** Inference with the 4-bit TinyLlama model needs about 1 to 2 GB of GPU memory.

---

## 8. Files and folders

| Path | Contents |
|---|---|
| `Indian_Legal_NER_SLM_Comparison.ipynb` | The notebook, with all outputs saved. |
| `app.py` | The Streamlit UI. |
| `runs/<model>/best/` | The original causal adapters: TinyLlama-1.1B, Qwen2.5-1.5B and Phi-3-mini-3.8B. |
| `runs/<model>/results.json` | The original metrics, per-label F1, timings and validation log. |
| `runs/<model>/checkpoint-*` | The last Trainer checkpoint, including optimizer state. Used only for resuming. |
| `runs_v2/preds_<model>.json` | Cached raw test predictions of the original models. |
| `runs_v2/TinyLlama-causal/`, `runs_v2/TinyLlama-bidir/` | The ablation runs: `best/` adapter, `results.json` (scores with snapping off and on, per-label F1, ceilings, timings) and `test_preds.json`. |
| `final_legal_ner_model/` | The previous final model (TinyLlama causal, F1 0.644). Kept unchanged. |
| `final_legal_ner_model_v2/` | **The current final model** (TinyLlama bidirectional + snapping, F1 0.802) and its `selection.json`. |
| `~/.cache/huggingface/` | The downloaded base models and the dataset. |

---

## 9. Re-running and reproducing

- **Nothing has to be re-run after a restart.** Results, adapters and notebook outputs are all saved on disk. To use the model, just run `streamlit run app.py` or follow [section 7](#7-using-the-final-model-for-inference).
- **Re-running the notebook is cheap.**
  - Training cells skip any model that already has a `results.json`.
  - Section 7.2 reads the cached predictions, so it needs no GPU pass.
  - Only small model loads remain: the attention tests, the final-model save and the demos, each well under a minute.
- **Retraining from scratch.** Delete or rename the relevant `runs/…` or `runs_v2/…` folder. A run that was interrupted resumes from its last checkpoint.
- **Quick experiments.** Set `TRAIN_SUBSAMPLE` (for example 3000) in the config cell, or call `run_experiment_v2(..., max_steps=20)` for a throughput check. Bidirectional TinyLlama runs at 0.92 it/s on an RTX 5060 Laptop GPU, which is about 24 to 30 min per full run.
- **Hardware used.** An RTX 5060 Laptop GPU (8 GB) in bf16, with PyTorch 2.14 (cu130), transformers 5.18, peft 0.21 and bitsandbytes 0.50.
