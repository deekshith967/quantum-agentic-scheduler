"""TEMPLATE, NOT RUN IN THIS PROTOTYPE. QLoRA SFT and DPO for the extraction SLM (Section 11.4).

Inputs are the JSONL files exported by Layer 14:
  data/runtime/finetune_sft.jsonl  {"input": {...raw text...}, "output": {...validated JSON...}}
  data/runtime/finetune_dpo.jsonl  {"prompt": {...}, "chosen": {...}, "rejected": {...}}

Requirements (not in requirements.txt): a CUDA GPU, torch, transformers, peft, trl, bitsandbytes,
datasets, and a downloaded base model, e.g. QAS_BASE_MODEL=Qwen/Qwen2.5-3B-Instruct.
A few hundred validated examples are needed before training is worthwhile.
"""
import json
import os
import sys
from pathlib import Path

RUNTIME = Path(os.environ.get("QAS_RUNTIME", Path(__file__).resolve().parent.parent / "data" / "runtime"))


def load(name):
    p = RUNTIME / name
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def main():
    sft, dpo = load("finetune_sft.jsonl"), load("finetune_dpo.jsonl")
    print(f"SFT examples: {len(sft)}  DPO pairs: {len(dpo)}")
    if len(sft) < 200:
        print("Not enough validated examples for training (need ~200+). Keep collecting approved episodes.")
        return
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer
    base = os.environ.get("QAS_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=BitsAndBytesConfig(load_in_4bit=True), device_map="auto")
    ds = Dataset.from_list([{"text": f"Extract planning entities as JSON.\n{json.dumps(r['input'])}\n### JSON\n{json.dumps(r['output'])}"} for r in sft])
    trainer = SFTTrainer(model=model, train_dataset=ds, processing_class=tok,
                         peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM"),
                         args=SFTConfig(output_dir="adapters/sft", num_train_epochs=2, per_device_train_batch_size=4))
    trainer.train()
    print("Adapter saved to adapters/sft. Run the frozen benchmark before promoting it (Section 11.5).")


if __name__ == "__main__":
    sys.exit(main())
