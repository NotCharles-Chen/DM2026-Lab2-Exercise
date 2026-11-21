"""5-fold fine-tuning of one transformer on the DM2026 Lab 2 emotion data.

Usage: python train.py MODEL_NAME TAG [lr] [epochs] [max_len] [batch]
Writes /kaggle/working/out/{TAG}_oof.npy, {TAG}_test.npy, {TAG}_log.json
"""
import json, os, sys, time, math, random
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup

MODEL, TAG = sys.argv[1], sys.argv[2]
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 4
MAX_LEN = int(sys.argv[5]) if len(sys.argv) > 5 else 128
BS = int(sys.argv[6]) if len(sys.argv) > 6 else 16
SEED, N_FOLDS = 42, 5
QUICK = bool(os.environ.get("QUICK"))   # smoke test: 1 fold, 1 epoch, 200 rows
if QUICK: EPOCHS = 1
LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
import glob
DATA = os.environ.get("DATA_DIR") or os.path.dirname(
    sorted(glob.glob("/kaggle/input/**/final_posts.json", recursive=True))[0])   # competition mount point
OUT = "/kaggle/working/out"; os.makedirs(OUT, exist_ok=True)

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ---------- data ----------
posts = pd.DataFrame(json.load(open(f"{DATA}/final_posts.json")))
ids = pd.read_csv(f"{DATA}/data_identification.csv")
lab = pd.read_csv(f"{DATA}/emotion.csv")
df = posts.merge(ids, on="post_id").merge(lab, on="post_id", how="left")
train = df[df.set == "train"].reset_index(drop=True)
test = df[df.set == "test"].reset_index(drop=True)
y = train.label.map({l: i for i, l in enumerate(LABELS)}).to_numpy()

def clean(t):
    # light normalisation only: transformers use casing/punctuation as signal
    t = t.replace("@<username>", "@user").replace("[NAME]", "Alex")
    return " ".join(t.split())

tok = AutoTokenizer.from_pretrained(MODEL)
def encode(texts):
    return tok([clean(t) for t in texts], truncation=True, max_length=MAX_LEN)

enc_train, enc_test = encode(train.text.tolist()), encode(test.text.tolist())

def make_loader(enc, idx, labels=None, shuffle=False):
    items = [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
              **({"labels": int(labels[i])} if labels is not None else {})} for i in idx]
    if not shuffle:  # sort by length for fast inference, remember order
        order = np.argsort([len(x["input_ids"]) for x in items])
        items = [items[i] for i in order]
    else:
        order = None
    def collate(b):
        out = tok.pad([{k: v for k, v in x.items() if k != "labels"} for x in b], return_tensors="pt")
        if "labels" in b[0]:
            out["labels"] = torch.tensor([x["labels"] for x in b])
        return out
    return DataLoader(items, batch_size=BS if shuffle else 64, shuffle=shuffle, collate_fn=collate), order

@torch.no_grad()
def predict(model, loader, order):
    model.eval(); outs = []
    for b in loader:
        b = {k: v.cuda() for k, v in b.items() if k != "labels"}
        with torch.autocast("cuda", dtype=torch.float16):
            outs.append(model(**b).logits.float().softmax(-1).cpu().numpy())
    p = np.concatenate(outs)
    res = np.empty_like(p); res[order] = p   # undo length sorting
    return res

def load_model():
    m = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(LABELS), ignore_mismatched_sizes=True,
        id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})
    # If the checkpoint was already an emotion classifier with these label names,
    # copy its output rows so fine-tuning starts from the pretrained head.
    try:
        src = AutoModelForSequenceClassification.from_pretrained(MODEL)
        names = {v.lower(): int(k) for k, v in src.config.id2label.items()}
        if all(l in names for l in LABELS):
            sw = dict(src.named_parameters()); dw = dict(m.named_parameters())
            for n, p in dw.items():
                if p.shape[0] == len(LABELS) and n in sw and sw[n].shape[0] == len(names):
                    p.data.copy_(sw[n].data[[names[l] for l in LABELS]])
            print("copied pretrained emotion head rows")
        del src
    except Exception as e:
        print("head copy skipped:", e)
    return m.float().cuda()   # some checkpoints (e.g. DeBERTa-v3) load in fp16; AMP needs fp32 master weights

oof = np.zeros((len(train), len(LABELS))); test_pred = np.zeros((len(test), len(LABELS)))
log = {"model": MODEL, "lr": LR, "epochs": EPOCHS, "max_len": MAX_LEN, "folds": []}
skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
test_loader, test_order = make_loader(enc_test, range(len(test)))
t_start = time.time()
for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
    if QUICK and fold > 0: break
    if QUICK: tr_idx = tr_idx[:200]
    seed_all(SEED + fold + 100 * int(os.environ.get("SEED_OFFSET", "0")))   # folds stay fixed; init/shuffle seed varies
    model = load_model()
    tr_loader, _ = make_loader(enc_train, tr_idx, y, shuffle=True)
    va_loader, va_order = make_loader(enc_train, va_idx, y)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * len(tr_loader)
    sch = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler()
    best = (-1, None, None)
    for ep in range(EPOCHS):
        model.train()
        for b in tr_loader:
            b = {k: v.cuda() for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.cross_entropy(
                    model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits.float(),
                    b["labels"], label_smoothing=0.05)
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
        pv = predict(model, va_loader, va_order)
        f1 = f1_score(y[va_idx], pv.argmax(1), average="macro")
        print(f"[{TAG}] fold {fold} epoch {ep + 1} val macroF1 {f1:.4f} ({time.time() - t_start:.0f}s)", flush=True)
        if f1 > best[0]:
            best = (f1, pv, predict(model, test_loader, test_order))
    oof[va_idx] = best[1]; test_pred += best[2] / N_FOLDS
    log["folds"].append(best[0])
    del model, opt; torch.cuda.empty_cache()

log["oof_macro_f1"] = float(f1_score(y, oof.argmax(1), average="macro"))
log["seconds"] = round(time.time() - t_start)
print(f"[{TAG}] OOF macro F1 {log['oof_macro_f1']:.4f}", flush=True)
np.save(f"{OUT}/{TAG}_oof.npy", oof); np.save(f"{OUT}/{TAG}_test.npy", test_pred)
json.dump(log, open(f"{OUT}/{TAG}_log.json", "w"), indent=1)
