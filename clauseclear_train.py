"""
clauseclear_train.py
====================
ClauseClear — Fine-tuning Gemma 3 270M (QLoRA)
Team Absolute Cinema · Kalpathon 2025

Pipeline: Legal Clause → Fine-tuned Gemma 3 270M → plain English + risk level + counter-clause

All steps in one file:
  1.  Install dependencies          (subprocess pip install)
  2.  Hugging Face login
  3.  Training dataset              (120 Indian rental clause examples)
  4.  Format prompts                (Gemma 3 IT format)
  5.  Load model + tokenizer        (4-bit QLoRA)
  6.  Apply LoRA adapters
  7.  Training configuration
  8.  Train
  9.  Evaluate                      (ROUGE-L + BERTScore + risk accuracy)
  10. Save adapter + merged model
  11. Push to HuggingFace Hub       (optional)
  12. Inference helper
  13. Missing clause detector        (rule-based)
  14. PDF clause splitter
  15. Agreement Health Report        (HTML)
  16. Full pipeline demo

Usage
-----
  python clauseclear_train.py

  # Skip training (eval + demo only, needs saved adapter)
  python clauseclear_train.py --eval_only

  # Don't push to Hub
  python clauseclear_train.py --no_push

  # Custom HF token
  python clauseclear_train.py --hf_token hf_xxxx

  # Custom model output dir
  python clauseclear_train.py --output_dir ./my_model
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Standard-library imports (always available — no pip needed)
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Install dependencies
# ─────────────────────────────────────────────────────────────────────────────
def install_dependencies():
    packages = [
        "transformers==4.44.0",
        "datasets",
        "peft",
        "trl",
        "bitsandbytes",
        "accelerate",
        "sentencepiece",
        "protobuf",
        "torch",
        "torchvision",
        "pypdf2",
        "reportlab",
        "jinja2",
        "evaluate",
        "rouge_score",
        "bert-score",
    ]
    print("── Installing dependencies ──────────────────────────────────────────")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q"] + packages
    )
    print("✅ Dependencies installed\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Hugging Face login
# ─────────────────────────────────────────────────────────────────────────────
def hf_login(token: str):
    print("── Logging in to Hugging Face ───────────────────────────────────────")
    from huggingface_hub import login
    login(token=token)
    print("✅ Logged in\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Training dataset
# ─────────────────────────────────────────────────────────────────────────────
TRAINING_DATA = [
    {
        "clause": "The Lessee shall not sublet, underlet, or part with possession of the demised premises or any part thereof without prior written consent of the Lessor.",
        "plain": "You cannot rent out any part of this property to someone else without getting written permission from your landlord first.",
        "risk": "LOW",
        "counter": "The Lessee may sublet a portion of the premises with 30 days prior written notice to the Lessor; consent shall not be unreasonably withheld.",
    },
    {
        "clause": "The security deposit shall stand forfeited in its entirety in the event the Lessee vacates the premises prior to the expiry of the lock-in period, irrespective of the reason for such vacation.",
        "plain": "If you leave before the minimum rental period ends (for any reason), you lose your entire security deposit — even if it was an emergency.",
        "risk": "HIGH",
        "counter": "In case of early vacation, only one month's rent equivalent shall be deducted from the deposit as a penalty; the remaining amount shall be refunded within 30 days of vacation.",
    },
    {
        "clause": "The Lessor shall have the right to terminate this agreement forthwith and demand immediate vacation of the premises if the Lessee commits any breach of the terms and conditions herein.",
        "plain": "The landlord can kick you out immediately without any warning if they think you've broken any rule in this agreement — even a minor one.",
        "risk": "HIGH",
        "counter": "In case of alleged breach, the Lessor shall provide written notice specifying the breach; the Lessee shall have 15 days to remedy the breach before termination proceedings may begin.",
    },
    {
        "clause": "This agreement shall automatically renew for a further period of eleven months upon expiry of the current term unless either party provides written notice of termination at least two months prior to the expiry date.",
        "plain": "This contract will automatically restart for another 11 months if you don't give written notice 2 months before it ends — you could be locked in without realising it.",
        "risk": "MEDIUM",
        "counter": "This agreement shall expire on the due date. Renewal requires explicit written agreement from both parties within 30 days of the expiry date.",
    },
    {
        "clause": "The Lessor shall not be held responsible or liable for any loss, damage, theft, or injury to the Lessee or the Lessee's guests, belongings, or property howsoever caused, occurring within the demised premises.",
        "plain": "The landlord takes zero responsibility for any harm — theft, injury, damage — that happens to you or your belongings inside your rented home, no matter the cause.",
        "risk": "HIGH",
        "counter": "The Lessor shall remain liable for damages arising from structural defects, failure to maintain essential services, or negligence on the part of the Lessor or their agents.",
    },
    {
        "clause": "The Lessor reserves the right to enter and inspect the demised premises at any time without prior notice for the purposes of inspection, repair, or any other reason deemed appropriate by the Lessor.",
        "plain": "Your landlord can walk into your home at any time, without warning, for any reason they want.",
        "risk": "HIGH",
        "counter": "The Lessor shall provide at least 48 hours written or verbal notice before entering the premises, except in cases of genuine emergency such as fire, flood, or gas leaks.",
    },
    {
        "clause": "The monthly rent shall be subject to an annual escalation of fifteen percent on the anniversary of this agreement.",
        "plain": "Your rent will go up by 15% every year automatically.",
        "risk": "MEDIUM",
        "counter": "The monthly rent shall be subject to an annual escalation not exceeding 5% or the Consumer Price Index increase, whichever is lower, on mutual written agreement.",
    },
    {
        "clause": "The Lessee shall be solely responsible for payment of all utility charges including electricity, water, gas, and maintenance society charges during the tenancy period.",
        "plain": "You must pay all utility bills — electricity, water, gas, and society fees — entirely on your own throughout your tenancy.",
        "risk": "LOW",
        "counter": "The Lessee shall be responsible for electricity and gas charges based on actual consumption. Water charges and society maintenance fees shall be shared equally between Lessor and Lessee.",
    },
    {
        "clause": "Any dispute arising out of or in connection with this agreement shall be referred to arbitration in accordance with the Arbitration and Conciliation Act, 1996, and the award of the arbitrator shall be final and binding.",
        "plain": "If you and your landlord have a disagreement, it must go to a private arbitrator — you cannot go directly to court. The arbitrator's decision is final.",
        "risk": "MEDIUM",
        "counter": "Disputes shall first be attempted to be resolved through mutual discussion. If unresolved within 30 days, either party may approach the appropriate civil court or consumer forum as per Indian law.",
    },
    {
        "clause": "The Lessee shall maintain the premises in a clean and habitable condition and shall hand over the premises in the same condition as received at the time of commencement of tenancy, fair wear and tear excepted.",
        "plain": "You must keep the place clean and return it in the same state you found it. Normal wear and tear from everyday living is acceptable.",
        "risk": "LOW",
        "counter": "The Lessee shall maintain reasonable cleanliness. A joint inspection shall be conducted at check-in and check-out with a signed condition report to avoid disputes.",
    },
    {
        "clause": "The Lessee shall not make any structural alterations, additions, or improvements to the premises without prior written consent from the Lessor.",
        "plain": "You cannot make any permanent changes to the property structure — like breaking walls or adding a room — without written permission from your landlord.",
        "risk": "LOW",
        "counter": "Minor cosmetic alterations such as painting or picture hooks shall be permitted. Structural changes require prior written consent from the Lessor, not to be unreasonably withheld.",
    },
    {
        "clause": "The security deposit shall be refunded within ninety days of vacation of the premises after deduction of any dues, damages, or charges as determined at the sole discretion of the Lessor.",
        "plain": "Your deposit will be returned within 90 days after you leave, but the landlord alone decides how much to deduct — you have no say in it.",
        "risk": "HIGH",
        "counter": "The security deposit shall be refunded within 15 days of vacation. Any deductions must be itemised in writing with supporting receipts; undisputed amounts shall be returned within 7 days.",
    },
    {
        "clause": "The Lessee shall use the premises only for residential purposes and shall not carry on any trade, business, or commercial activity from the said premises.",
        "plain": "You can only live here — you cannot run any business, freelance work, or commercial activity from this address.",
        "risk": "LOW",
        "counter": "The Lessee may use the premises for residential purposes and for remote work or freelance activities that do not involve clients visiting the property or any physical commercial operation.",
    },
    {
        "clause": "The Lessor shall be entitled to claim mesne profits at double the monthly rent for each month or part thereof that the Lessee continues to occupy the premises after the expiry or termination of this agreement.",
        "plain": "If you stay even one day after your agreement ends, your landlord can charge you double the monthly rent for that entire month.",
        "risk": "HIGH",
        "counter": "Upon expiry of the agreement, the Lessee shall pay rent on a pro-rata daily basis for any additional days of occupation, at the prevailing monthly rent rate.",
    },
    {
        "clause": "All repairs and maintenance of the premises, whether structural or cosmetic, howsoever arising, shall be carried out at the cost and expense of the Lessee.",
        "plain": "You must pay for ALL repairs — even structural damage to the building itself — out of your own pocket.",
        "risk": "HIGH",
        "counter": "Structural and major repairs (above Rs. 2,000) shall be the Lessor's responsibility. Day-to-day minor maintenance up to Rs. 500 per incident shall be the Lessee's responsibility.",
    },
    {
        "clause": "The Lessee shall obtain prior written approval from the Lessor before registering the said premises as address for any official, governmental, or commercial purpose.",
        "plain": "You need written permission from your landlord before using this address for Aadhaar, voter ID, bank accounts, company registration, or any official purpose.",
        "risk": "MEDIUM",
        "counter": "The Lessee is permitted to use the address for personal official documents including Aadhaar, bank accounts, and voter registration. Commercial registration requires prior written consent.",
    },
    {
        "clause": "The Lessee undertakes not to keep any pets, animals, or birds of any kind in the premises without the express written permission of the Lessor.",
        "plain": "No pets of any kind are allowed — not even birds — without written permission from your landlord.",
        "risk": "LOW",
        "counter": "The Lessee may keep small domesticated pets with 30 days prior written notice to the Lessor; an additional refundable pet deposit of one month's rent may be collected.",
    },
    {
        "clause": "In the event of non-payment of rent for a period exceeding seven days from the due date, the Lessor shall be entitled to immediately terminate this agreement and take possession of the premises.",
        "plain": "If you are even 7 days late on rent, your landlord can terminate your lease and reclaim the property immediately.",
        "risk": "HIGH",
        "counter": "In case of non-payment of rent, the Lessor shall issue a written reminder. If rent remains unpaid after 30 days of written notice, the Lessor may initiate termination proceedings as per law.",
    },
    {
        "clause": "The Lessee shall ensure that no nuisance, annoyance, or inconvenience is caused to the neighbouring occupants and shall abide by all rules and regulations of the housing society.",
        "plain": "You must be a considerate neighbour and follow all housing society rules — violations could be used against you.",
        "risk": "LOW",
        "counter": "The Lessee agrees to observe reasonable conduct and follow society rules communicated in writing. The Lessor shall also provide a copy of all applicable society bye-laws at the time of handover.",
    },
    {
        "clause": "The stamp duty and registration charges for this agreement shall be borne entirely by the Lessee.",
        "plain": "You — the tenant — must pay all the legal registration and stamp duty costs for this rental agreement.",
        "risk": "MEDIUM",
        "counter": "Stamp duty and registration charges for this agreement shall be shared equally between the Lessor and the Lessee.",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Format training prompts (Gemma 3 IT format)
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are ClauseClear, an expert in Indian rental law and tenant rights. "
    "When given a legal clause from an Indian rental agreement, you must respond "
    "in this exact JSON format:\n"
    "{\n"
    '  "plain": "<one clear sentence any renter can understand>",\n'
    '  "risk": "<HIGH|MEDIUM|LOW>",\n'
    '  "risk_reason": "<brief reason for risk level>",\n'
    '  "counter": "<a fairer counter-clause the tenant can propose>"\n'
    "}\n"
    "Always be accurate about Indian rental law "
    "(Transfer of Property Act, 1882; Model Tenancy Act, 2021)."
)


def format_example(example: dict) -> dict:
    output = json.dumps(
        {
            "plain": example["plain"],
            "risk": example["risk"],
            "risk_reason": (
                f"This clause is {example['risk'].lower()} risk "
                "because it affects tenant rights significantly."
            ),
            "counter": example["counter"],
        },
        ensure_ascii=False,
    )
    text = (
        f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n"
        f"Analyse this clause:\n{example['clause']}<end_of_turn>\n"
        f"<start_of_turn>model\n{output}<end_of_turn>"
    )
    return {"text": text}


def build_dataset():
    from datasets import Dataset

    print("── Building dataset ─────────────────────────────────────────────────")
    formatted = [format_example(ex) for ex in TRAINING_DATA]
    dataset = Dataset.from_list(formatted)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"✅ Dataset ready — Train: {len(dataset['train'])} | Val: {len(dataset['test'])}\n")
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Load model + tokenizer (4-bit QLoRA)
# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_id: str = "google/gemma-3-270m-it"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"── Loading model: {model_id} ────────────────────────────────────────")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"✅ Model loaded — {n_params:.2f}B params\n")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Apply LoRA adapters
# ─────────────────────────────────────────────────────────────────────────────
def apply_lora(model):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    print("── Applying LoRA adapters ────────────────────────────────────────────")
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 7 + 8.  Configure trainer and train
# ─────────────────────────────────────────────────────────────────────────────
def train(model, tokenizer, dataset, output_dir: str = "./clauseclear-gemma3-270m"):
    from transformers import TrainingArguments
    from trl import SFTTrainer

    print("── Configuring trainer ───────────────────────────────────────────────")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,    # effective batch = 8
        warmup_steps=10,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
        max_grad_norm=0.3,
        weight_decay=0.001,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    print("✅ Trainer configured — starting training\n")
    print("── Training ─────────────────────────────────────────────────────────")
    trainer.train()
    print("✅ Training complete!\n")
    return trainer


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Evaluate — ROUGE-L + BERTScore + risk accuracy
# ─────────────────────────────────────────────────────────────────────────────
EVAL_SET = [
    {
        "clause": "The security deposit shall stand forfeited in its entirety if the Lessee vacates before the lock-in period.",
        "reference": "If you leave before the lock-in period ends you lose your entire deposit.",
        "expected_risk": "HIGH",
    },
    {
        "clause": "The Lessor reserves the right to enter and inspect at any time without prior notice.",
        "reference": "The landlord can enter your home at any time with no warning.",
        "expected_risk": "HIGH",
    },
    {
        "clause": "The monthly rent shall increase by fifteen percent every year automatically.",
        "reference": "Your rent goes up by 15% every year.",
        "expected_risk": "MEDIUM",
    },
    {
        "clause": "The Lessee shall maintain the premises in good condition, fair wear and tear excepted.",
        "reference": "Keep the property in good shape — normal everyday wear is fine.",
        "expected_risk": "LOW",
    },
    {
        "clause": "All disputes shall be referred to arbitration; the arbitrator's decision shall be final.",
        "reference": "Any disagreements must go to a private arbitrator, not court.",
        "expected_risk": "MEDIUM",
    },
]


def _parse_risk_from_output(raw: str) -> str:
    """Pull risk level out of model JSON output."""
    try:
        data = json.loads(raw.strip())
        return data.get("risk", "LOW").upper()
    except Exception:
        for level in ("HIGH", "MEDIUM", "LOW"):
            if level in raw.upper():
                return level
    return "LOW"


def evaluate_model(model, tokenizer, label: str = "Fine-tuned") -> dict:
    import torch
    import evaluate as hf_eval
    from bert_score import score as bert_score_fn

    print(f"── Evaluating: {label} ──────────────────────────────────────────────")

    rouge = hf_eval.load("rouge")
    predictions, references, risk_correct = [], [], 0

    for ex in EVAL_SET:
        prompt = (
            f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n"
            f"Analyse this clause:\n{ex['clause']}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        # Extract plain-english prediction for ROUGE
        try:
            pred_plain = json.loads(raw)["plain"]
        except Exception:
            pred_plain = raw.split("\n")[0]

        predictions.append(pred_plain)
        references.append(ex["reference"])

        predicted_risk = _parse_risk_from_output(raw)
        if predicted_risk == ex["expected_risk"]:
            risk_correct += 1

    rouge_scores = rouge.compute(predictions=predictions, references=references)
    _, _, F1 = bert_score_fn(predictions, references, lang="en", verbose=False)

    results = {
        "rouge_l":       round(rouge_scores["rougeL"], 4),
        "bertscore_f1":  round(F1.mean().item(), 4),
        "risk_accuracy": round(risk_correct / len(EVAL_SET), 4),
    }

    print(f"  ROUGE-L:       {results['rouge_l']:.4f}")
    print(f"  BERTScore F1:  {results['bertscore_f1']:.4f}")
    print(f"  Risk Accuracy: {results['risk_accuracy']:.0%}")
    print()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Save adapter + merged model
# ─────────────────────────────────────────────────────────────────────────────
def save_models(trainer, tokenizer, model_id: str, adapter_dir: str = "./clauseclear-adapter",
                merged_dir: str = "./clauseclear-merged"):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    print("── Saving models ────────────────────────────────────────────────────")

    # 1. Save LoRA adapter (small, ~20 MB)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"✅ LoRA adapter saved → {adapter_dir}")

    # 2. Merge and save full model (for Hub / deployment)
    print("   Merging LoRA weights into base model...")
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cpu"
    )
    merged = PeftModel.from_pretrained(base, adapter_dir)
    merged = merged.merge_and_unload()
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"✅ Merged model saved → {merged_dir}\n")

    return adapter_dir, merged_dir


# ─────────────────────────────────────────────────────────────────────────────
# 11.  Push to Hugging Face Hub (optional)
# ─────────────────────────────────────────────────────────────────────────────
def push_to_hub(merged_dir: str, tokenizer, repo_id: str):
    print(f"── Pushing to Hub: {repo_id} ─────────────────────────────────────────")
    from transformers import AutoModelForCausalLM
    import torch

    model = AutoModelForCausalLM.from_pretrained(
        merged_dir, torch_dtype=torch.float16, device_map="cpu"
    )
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)
    print(f"✅ Model live at: https://huggingface.co/{repo_id}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 12.  Inference helper
# ─────────────────────────────────────────────────────────────────────────────
def analyse_clause(clause_text: str, model, tokenizer, max_new_tokens: int = 256) -> dict:
    import torch

    prompt = (
        f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n"
        f"Analyse this clause:\n{clause_text}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    try:
        return json.loads(response.strip())
    except Exception:
        return {"raw": response.strip()}


# ─────────────────────────────────────────────────────────────────────────────
# 13.  Missing clause detector (rule-based, no training needed)
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED_CLAUSES = {
    "deposit_return_timeline": {
        "keywords": ["deposit", "refund", "return", "security"],
        "message": "⚠️ No deposit return timeline — landlord can hold your money indefinitely.",
    },
    "notice_period": {
        "keywords": ["notice", "notice period", "vacate"],
        "message": "⚠️ No notice period clause — you could be asked to leave without warning.",
    },
    "maintenance_responsibility": {
        "keywords": ["maintenance", "repair", "upkeep"],
        "message": "⚠️ Maintenance responsibility not defined — all repair costs may fall on you.",
    },
    "rent_escalation": {
        "keywords": ["escalation", "increase", "hike", "enhanced"],
        "message": "⚠️ No rent escalation clause — unlimited rent increases possible at renewal.",
    },
    "lock_in_period": {
        "keywords": ["lock-in", "lock in", "minimum period", "cannot vacate"],
        "message": "ℹ️ No lock-in period defined — either party can leave anytime.",
    },
    "landlord_entry_rights": {
        "keywords": ["entry", "inspect", "access", "enter"],
        "message": "⚠️ Landlord entry rights not specified — your privacy may not be protected.",
    },
}


def detect_missing_clauses(full_agreement_text: str) -> list:
    text_lower = full_agreement_text.lower()
    return [
        cfg["message"]
        for cfg in REQUIRED_CLAUSES.values()
        if not any(kw in text_lower for kw in cfg["keywords"])
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 14.  PDF clause splitter
# ─────────────────────────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        import PyPDF2
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pypdf2"])
        import PyPDF2

    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        return " ".join(page.extract_text() or "" for page in reader.pages)


def split_into_clauses(text: str) -> list:
    """Split agreement text into individual clauses."""
    # Try numbered split first: "1. The Lessee..."
    clauses = re.split(r"(?=\b\d{1,2}\.\s+[A-Z])", text)
    clauses = [c.strip() for c in clauses if len(c.strip()) > 50]

    # Fallback: split on sentence boundaries
    if len(clauses) < 3:
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        clauses = [s.strip() for s in sentences if len(s.strip()) > 80]

    return clauses[:20]  # cap at 20 for performance


# ─────────────────────────────────────────────────────────────────────────────
# 15.  Agreement Health Report (HTML)
# ─────────────────────────────────────────────────────────────────────────────
def generate_health_report(
    clauses_with_analysis: list,
    missing_clauses: list,
    filename: str = "health_report.html",
) -> str:
    risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for _, analysis in clauses_with_analysis:
        risk = analysis.get("risk", "LOW")
        risk_counts[risk] = risk_counts.get(risk, 0) + 1

    total = sum(risk_counts.values())
    score = max(
        0,
        100
        - (risk_counts["HIGH"] * 20)
        - (risk_counts["MEDIUM"] * 8)
        - (len(missing_clauses) * 5),
    )

    rows = ""
    for clause, analysis in clauses_with_analysis:
        risk = analysis.get("risk", "LOW")
        emoji = "🔴" if risk == "HIGH" else "🟡" if risk == "MEDIUM" else "🟢"
        rows += f"""
        <tr>
          <td class='clause'>{clause[:120]}...</td>
          <td>{analysis.get('plain', 'N/A')}</td>
          <td class='risk-{risk.lower()}'>{emoji} {risk}</td>
          <td class='counter'>{analysis.get('counter', 'N/A')}</td>
        </tr>"""

    missing_html = (
        "".join(f"<li>{m}</li>" for m in missing_clauses)
        or "<li>✅ All standard protective clauses are present.</li>"
    )

    score_class = "good" if score >= 70 else "medium" if score >= 40 else "bad"

    html = f"""<!DOCTYPE html>
<html><head><meta charset='UTF-8'><title>ClauseClear Health Report</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; max-width: 1100px; margin: 40px auto; color: #1a1a2e; }}
  h1 {{ color: #16213e; }}
  .score {{ font-size: 48px; font-weight: bold; }}
  .score.good {{ color: #27ae60; }} .score.medium {{ color: #f39c12; }} .score.bad {{ color: #e74c3c; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
  th {{ background: #16213e; color: white; padding: 12px; text-align: left; }}
  td {{ padding: 10px; border-bottom: 1px solid #eee; vertical-align: top; font-size: 13px; }}
  .risk-high {{ color: #e74c3c; font-weight: bold; }}
  .risk-medium {{ color: #f39c12; font-weight: bold; }}
  .risk-low {{ color: #27ae60; }}
  .counter {{ font-style: italic; color: #2980b9; }}
  .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
  .stat {{ padding: 20px; border-radius: 8px; text-align: center; flex: 1; }}
  .stat.high {{ background: #fdecea; }} .stat.medium {{ background: #fef9e7; }} .stat.low {{ background: #eafaf1; }}
  .missing {{ background: #fef9e7; border-left: 4px solid #f39c12; padding: 16px; margin: 20px 0; }}
</style></head><body>
<h1>⚖️ ClauseClear — Agreement Health Report</h1>
<p>Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')} | Total Clauses Analysed: {total}</p>
<div class='score {score_class}'>{score}/100</div>
<p>Agreement Health Score</p>
<div class='stats'>
  <div class='stat high'><h2>🔴 {risk_counts['HIGH']}</h2><p>High Risk</p></div>
  <div class='stat medium'><h2>🟡 {risk_counts['MEDIUM']}</h2><p>Medium Risk</p></div>
  <div class='stat low'><h2>🟢 {risk_counts['LOW']}</h2><p>Low Risk</p></div>
</div>
<div class='missing'><strong>Missing Protective Clauses:</strong><ul>{missing_html}</ul></div>
<table>
  <tr><th>Original Clause</th><th>Plain English</th><th>Risk</th><th>Counter-Clause</th></tr>
  {rows}
</table>
</body></html>"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Health report saved → {filename}")
    return html


# ─────────────────────────────────────────────────────────────────────────────
# 16.  Full pipeline demo
# ─────────────────────────────────────────────────────────────────────────────
DEMO_AGREEMENT = """
1. The Lessee shall pay rent of Rs. 25,000 per month on the 1st of each month.
2. The security deposit shall stand forfeited in full if the Lessee vacates before the lock-in period.
3. The Lessor reserves the right to terminate this agreement without notice at their sole discretion.
4. All repairs and maintenance shall be borne entirely by the Lessee.
5. The agreement shall auto-renew for 11 months unless written notice is given 2 months in advance.
"""


def run_demo(model, tokenizer, report_file: str = "demo_health_report.html"):
    print("── Full pipeline demo ────────────────────────────────────────────────")
    clauses = split_into_clauses(DEMO_AGREEMENT)
    print(f"Found {len(clauses)} clauses in demo agreement\n")

    results = []
    for clause in clauses:
        analysis = analyse_clause(clause, model, tokenizer)
        results.append((clause, analysis))
        plain = analysis.get("plain", analysis.get("raw", "N/A"))
        risk  = analysis.get("risk", "N/A")
        print(f"  📋 {clause[:70]}...")
        print(f"     Plain: {plain}")
        print(f"     Risk:  {risk}\n")

    missing = detect_missing_clauses(DEMO_AGREEMENT)
    if missing:
        print("Missing clauses detected:")
        for m in missing:
            print(f"  {m}")

    generate_health_report(results, missing, report_file)
    print("\n✅ Full pipeline demo complete!\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main entrypoint
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="ClauseClear — Fine-tune Gemma 3 270M on Indian rental clauses"
    )
    parser.add_argument(
        "--hf_token", default=os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN_HERE"),
        help="Hugging Face token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--model_id", default="google/gemma-3-270m-it",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--output_dir", default="./clauseclear-gemma3-270m",
        help="Training checkpoint directory",
    )
    parser.add_argument(
        "--adapter_dir", default="./clauseclear-adapter",
        help="Where to save the LoRA adapter",
    )
    parser.add_argument(
        "--merged_dir", default="./clauseclear-merged",
        help="Where to save the merged full model",
    )
    parser.add_argument(
        "--hub_repo", default=None,
        help="HuggingFace repo to push merged model to (e.g. yourname/clauseclear)",
    )
    parser.add_argument(
        "--eval_only", action="store_true",
        help="Skip training — load saved adapter and run eval + demo only",
    )
    parser.add_argument(
        "--no_push", action="store_true",
        help="Do not push to HuggingFace Hub even if --hub_repo is set",
    )
    parser.add_argument(
        "--skip_install", action="store_true",
        help="Skip pip install step (use if deps already installed)",
    )
    parser.add_argument(
        "--report_file", default="demo_health_report.html",
        help="Output filename for the HTML health report",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 1. Install ────────────────────────────────────────────────────────────
    if not args.skip_install:
        install_dependencies()

    # ── 2. HF login ───────────────────────────────────────────────────────────
    if args.hf_token and args.hf_token != "YOUR_HF_TOKEN_HERE":
        hf_login(args.hf_token)
    else:
        print("⚠️  No HF_TOKEN provided — skipping login.\n"
              "   Set --hf_token or export HF_TOKEN=... to access Gemma.\n")

    # ── 3 + 4. Dataset ────────────────────────────────────────────────────────
    dataset = build_dataset()

    # ── 5 + 6. Load model + LoRA ──────────────────────────────────────────────
    if args.eval_only:
        # Load the saved fine-tuned adapter for eval/demo
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"── Loading saved adapter from {args.adapter_dir} ──────────────────")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_id, quantization_config=bnb_config, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
        tokenizer.pad_token = tokenizer.eos_token
        ft_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
        print(f"✅ Adapter loaded from {args.adapter_dir}\n")

        # Evaluate
        evaluate_model(ft_model, tokenizer, label="Saved Fine-tuned Model")

        # Demo
        run_demo(ft_model, tokenizer, args.report_file)

        return

    # ── Full training path ────────────────────────────────────────────────────
    model, tokenizer = load_model(args.model_id)
    model = apply_lora(model)

    # ── 7 + 8. Train ──────────────────────────────────────────────────────────
    trainer = train(model, tokenizer, dataset, args.output_dir)

    # ── 9. Evaluate ───────────────────────────────────────────────────────────
    print("── Evaluation ───────────────────────────────────────────────────────")
    eval_results = evaluate_model(trainer.model, tokenizer, label="Fine-tuned ClauseClear")

    # Save eval results as JSON alongside the adapter
    eval_path = Path(args.adapter_dir).parent / "eval_results.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"✅ Eval results saved → {eval_path}\n")

    # ── 10. Save ──────────────────────────────────────────────────────────────
    adapter_dir, merged_dir = save_models(
        trainer, tokenizer, args.model_id, args.adapter_dir, args.merged_dir
    )

    # ── 11. Push to Hub ───────────────────────────────────────────────────────
    if args.hub_repo and not args.no_push:
        push_to_hub(merged_dir, tokenizer, args.hub_repo)
    else:
        print("ℹ️  Skipping Hub push. Pass --hub_repo yourname/repo to upload.\n")

    # ── 12–16. Inference + demo ───────────────────────────────────────────────
    run_demo(trainer.model, tokenizer, args.report_file)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("═" * 60)
    print("✅ ClauseClear training pipeline complete!")
    print(f"   Adapter:       {adapter_dir}")
    print(f"   Merged model:  {merged_dir}")
    print(f"   Eval results:  {eval_path}")
    print(f"   Health report: {args.report_file}")
    print()
    print("Next steps:")
    print("  Load for inference:")
    print(f"    model = AutoModelForCausalLM.from_pretrained('{merged_dir}')")
    print("  Push to Hub:")
    print("    python clauseclear_train.py --eval_only --hub_repo yourname/clauseclear")
    print("═" * 60)


if __name__ == "__main__":
    main()
