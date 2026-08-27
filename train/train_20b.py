# -*- coding: utf-8 -*-
# ==================================================
#  NEXTGEN FLAGSHIP — fine-tune gpt-oss:20b itself
#  Student = gpt-oss-20b (open-weights) via QLoRA
#  Data    = the merged distill dataset (880 rows)
#  Output  = Q4_K_M GGUF uploaded to Kaggle dataset
#  Fallback: Qwen3-14B if 20B cannot fit/export
# ==================================================
import json, os, subprocess, time, shutil, sys

OUT_JSONL  = "/kaggle/working/train.jsonl"
DS_MODEL   = "kingking1111/nextgen-model-20b"
DS_FULL    = "kingking1111/nextgen-distill-full"
PARTS      = ["kingking1111/nextgen-distill-data",
              "kingking1111/nextgen-distill-part2",
              "kingking1111/nextgen-distill-part3",
              "kingking1111/nextgen-distill-part4",
              "kingking1111/nextgen-distill-part5",
              "kingking1111/nextgen-distill-part6"]
WAIT_MIN   = 180

def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)

def sh(cmd, silent=True, timeout=3600):
    if not silent: print(">", cmd[:140], flush=True)
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0 and not silent:
            print(r.stderr[-400:], flush=True)
        return r
    except Exception as e:
        print("cmd failed:", e, flush=True); return None

# --------------------------------------------------
# 1) Collect all chunks (wait for Colab parts if needed)
# --------------------------------------------------
part_root = "/kaggle/working/parts"
os.makedirs(part_root, exist_ok=True)
got = {}
deadline = time.time() + WAIT_MIN * 60
while time.time() < deadline and len(got) < len(PARTS):
    for ds in PARTS:
        if ds in got:
            continue
        d = os.path.join(part_root, ds.split("/")[-1])
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        r = sh("kaggle datasets download -d %s -p %s --unzip --quiet" % (ds, d), timeout=900)
        if r is not None and r.returncode == 0 and os.path.exists(os.path.join(d, "train.jsonl")):
            got[ds] = d
            log("Found:", ds)
    if len(got) < len(PARTS):
        log("Parts:", len(got), "/", len(PARTS), "- retrying in 90s...")
        time.sleep(90)

rows = []
seen = set()
for ds, d in got.items():
    for ln in open(os.path.join(d, "train.jsonl"), encoding="utf-8"):
        try:
            x = json.loads(ln)
        except Exception:
            continue
        key = (x.get("teacher"), x.get("prompt", "").strip().lower())
        if key in seen:
            continue
        seen.add(key)
        rows.append(x)
log("Merged dataset rows:", len(rows))
if not rows:
    log("NO DATA - aborting"); sys.exit(1)

with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# upload the full merged dataset for future runs
os.makedirs("/kaggle/working/ds_full", exist_ok=True)
shutil.copy(OUT_JSONL, "/kaggle/working/ds_full/train.jsonl")
meta = {"id": DS_FULL, "title": "NextGen Distill Full", "isPrivate": False,
        "licenses": [{"name": "other"}], "updateFrequency": "never"}
with open("/kaggle/working/ds_full/dataset-metadata.json", "w") as f:
    json.dump(meta, f)
r = sh("kaggle datasets create -p /kaggle/working/ds_full -r zip --quiet")
if r is None or r.returncode != 0:
    r = sh("kaggle datasets version -p /kaggle/working/ds_full -m update --quiet")
log("Full dataset upload:", "OK" if (r is not None and r.returncode == 0) else "FAILED")

# --------------------------------------------------
# 2) Install unsloth
# --------------------------------------------------
log("Installing unsloth...")
sh("pip install -q unsloth", silent=False, timeout=1800)
sh("pip install -q trl datasets", silent=False, timeout=1800)

import torch
from datasets import Dataset
from transformers import TrainingArguments

def build_train_ds():
    dr = []
    for ln in open(OUT_JSONL, encoding="utf-8"):
        try:
            dr.append(json.loads(ln))
        except Exception:
            pass
    ds = Dataset.from_list([{"prompt": x["prompt"], "answer": x["answer"]} for x in dr])
    def fmt(ex):
        return {"text": "<s>[INST] " + ex["prompt"] + " [/INST] " + ex["answer"] + "</s>"}
    return ds.map(fmt)

from unsloth import FastLanguageModel, is_bfloat16_supported

def try_train(model_id, ds, name):
    log("Loading", name, "...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_id, max_seq_length=1024, load_in_4bit=True,
        )
    except Exception as e:
        log(name, "load failed:", str(e)[:200]); return False
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    from trl import SFTTrainer
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        max_seq_length=1024, dataset_text_field="text",
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_ratio=0.05,
            num_train_epochs=2,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            output_dir="/kaggle/working/out",
            report_to=[],
        ),
    )
    log("Training", name, "...")
    trainer.train()
    log("Training done.")
    # export GGUF
    shutil.rmtree("/kaggle/working/gguf", ignore_errors=True)
    try:
        model.save_pretrained_gguf("/kaggle/working/gguf", tokenizer, quantization_method="q4_k_m")
    except Exception as e:
        log("GGUF export failed:", str(e)[:200]); return False
    files = os.listdir("/kaggle/working/gguf")
    log("GGUF files:", files)
    src = os.path.join("/kaggle/working/gguf", "nextgen-trained.gguf")
    if not os.path.exists(src):
        for f in files:
            if f.endswith(".gguf"):
                os.rename(os.path.join("/kaggle/working/gguf", f), src)
                break
    if not os.path.exists(src):
        log("NO GGUF PRODUCED"); return False
    import gc
    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    os.makedirs("/kaggle/working/ds_model", exist_ok=True)
    shutil.copy(src, "/kaggle/working/ds_model/nextgen-trained.gguf")
    meta = {"id": DS_MODEL, "title": "NextGen Model 20B", "isPrivate": False,
            "licenses": [{"name": "other"}], "updateFrequency": "never"}
    with open("/kaggle/working/ds_model/dataset-metadata.json", "w") as f:
        json.dump(meta, f)
    r = sh("kaggle datasets create -p /kaggle/working/ds_model -r zip --quiet")
    if r is None or r.returncode != 0:
        r = sh("kaggle datasets version -p /kaggle/working/ds_model -m update --quiet")
    log("Model upload:", "OK" if (r is not None and r.returncode == 0) else "FAILED")
    return True

ds = build_train_ds()
log("Train rows:", len(ds))

# try 20B first (MoE, needs ~13GB 4-bit); fall back to Qwen3-14B
ok = False
for mid, name in [("unsloth/gpt-oss-20b-bnb-4bit", "gpt-oss-20b"),
                  ("unsloth/Qwen3-14B-bnb-4bit", "Qwen3-14B")]:
    try:
        if try_train(mid, ds, name):
            ok = True
            break
    except Exception as e:
        log(name, "train failed:", str(e)[:200])

log("FLAGSHIP RESULT:", "SUCCESS" if ok else "FAILED")
