from datasets import load_dataset
import torch
import random
from tqdm import tqdm
import torch.nn as nn


def get_wikitext2_train(tokenizer, size, seed, seqlen):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    random.seed(seed)
    out = []
    for _ in range(size):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        out.append((inp[0], tar[0]))
    return out


def get_wikitext2_test(tokenizer, seqlen):
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt").input_ids[0]
    return enc


def get_c4_train(tokenizer, size, seed, seqlen):
    traindata = load_dataset(
        "allenai/c4",
        "allenai--c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    random.seed(seed)
    out = []
    for _ in range(size):
        while True:
            i = random.randint(0, len(traindata) - 1)
            enc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if enc.input_ids.shape[1] >= seqlen + 1:
                break
        a = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        b = a + seqlen
        inp = enc.input_ids[:, a:b]
        tar = inp.clone()
        tar[:, :-1] = -100
        out.append((inp[0], tar[0]))
    return out


def get_c4_test(tokenizer, seqlen, segments=256):
    valdata = load_dataset(
        "allenai/c4",
        "allenai--c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )
    random.seed(0)
    parts = []
    for _ in range(segments):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
            if tmp.input_ids.shape[1] >= seqlen:
                break
        a = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        b = a + seqlen
        parts.append(tmp.input_ids[:, a:b])
    return torch.hstack(parts)[0]


def get_redpajama_train(tokenizer, size, seed, seqlen):
    data = load_dataset(
        "togethercomputer/RedPajama-Data-1T-Sample", split="train"
    ).shuffle(seed=seed)
    random.seed(seed)
    out = []
    val_ratio = 0.9
    ntrain = int(len(data) * val_ratio)
    for _ in range(size):
        while True:
            i = random.randint(0, ntrain - 1)
            enc = tokenizer(data[i]["text"], return_tensors="pt")
            if enc.input_ids.shape[1] >= seqlen + 1:
                break
        a = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        b = a + seqlen
        inp = enc.input_ids[:, a:b]
        tar = inp.clone()
        tar[:, :-1] = -100
        out.append((inp[0], tar[0]))
    return out


def get_redpajama_test(tokenizer, seqlen, segments=256):
    data = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train")
    random.seed(0)
    val_start = int(len(data) * 0.9)
    parts = []
    for _ in range(segments):
        while True:
            i = random.randint(val_start, len(data) - 1)
            tmp = tokenizer(data[i]["text"], return_tensors="pt")
            if tmp.input_ids.shape[1] >= seqlen:
                break
        a = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        b = a + seqlen
        parts.append(tmp.input_ids[:, a:b])
    return torch.hstack(parts)[0]


def get_train(name, tokenizer, size=128, seed=0, seqlen=2048):
    if "wikitext2" in name:
        return get_wikitext2_train(tokenizer, size, seed, seqlen)
    elif "c4" in name:
        return get_c4_train(tokenizer, size, seed, seqlen)
    elif "redpajama" in name:
        return get_redpajama_train(tokenizer, size, seed, seqlen)
    else:
        raise NotImplementedError


def get_test(name, tokenizer, seqlen=2048):
    if "wikitext2" in name:
        return get_wikitext2_test(tokenizer, seqlen)
    elif "c4" in name:
        return get_c4_test(tokenizer, seqlen)
    elif "redpajama" in name:
        return get_redpajama_test(tokenizer, seqlen)
    else:
        raise NotImplementedError


@torch.no_grad()
def test_ppl(model, tokenizer, datasets=["wikitext2"], ppl_seqlen=2048):
    import torch.nn as nn

    results = {}
    for dataset in datasets:
        testenc = get_test(dataset, tokenizer, ppl_seqlen)
        seqlen = ppl_seqlen
        nsamples = testenc.numel() // seqlen

        use_cache = model.config.use_cache
        model.config.use_cache = False
        model.eval()

        nlls = []
        skipped = 0

        if hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Linear):
            classifier = model.lm_head
        elif hasattr(model.model, "lm_head"):
            classifier = None
        elif hasattr(model, "output"):
            classifier = model.output
        else:
            raise NotImplementedError

        for i in tqdm(range(nsamples)):
            batch = (
                testenc[(i * seqlen) : ((i + 1) * seqlen)]
                .to(model.device)
                .reshape(1, -1)
            )

            outputs = model.model(batch)
            if classifier is not None:
                hidden_states = outputs[0]
                logits = classifier(hidden_states.to(classifier.weight.dtype))
            else:
                logits = outputs[0]

            if not torch.isfinite(logits).all():
                skipped += 1
                continue

            shift_logits = logits[:, :-1, :]
            shift_labels = testenc[(i * seqlen) : ((i + 1) * seqlen)][1:].to(
                shift_logits.device
            )

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            if not torch.isfinite(loss):
                skipped += 1
                continue

            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)

        if len(nlls) == 0:
            raise RuntimeError(
                f"All chunks were non-finite for dataset {dataset}. "
                f"Skipped {skipped}/{nsamples} chunks."
            )

        ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seqlen))
        print(f"{dataset}: {ppl} (skipped {skipped}/{nsamples} chunks)")
        results[dataset] = ppl.item()

    model.config.use_cache = use_cache
    return results
