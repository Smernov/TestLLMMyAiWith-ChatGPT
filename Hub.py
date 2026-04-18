#!/usr/bin/env python3
"""
Local LLM Hub with simple console UI + LoRA fine-tuning.

Features:
- Load model from HF or local path (supports load_in_8bit if GPU + bitsandbytes available)
- Console menu (chat, change settings, show history, save/load state, reload model)
- Fine-tune using LoRA (PEFT) on a local JSONL dataset with items: {"prompt": "...", "response": "..."}
- Save LoRA adapters and use them by pointing model path to adapter directory

Requirements:
pip install torch transformers accelerate bitsandbytes peft datasets sentencepiece rich
"""

import os
import sys
import json
import tempfile
from typing import List, Dict
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import track
import torch

# Transformers / PEFT / Datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftConfig

console = Console()
STATE_FILE = "hub_state.json"


class Hub:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.temp = 0.7
        self.max_tokens = 256
        self.history: List[Dict] = []
        self.load_state()
        self.load_model()

    def load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    s = json.load(f)
                self.model_name = s.get("model_name", self.model_name)
                self.temp = s.get("temp", self.temp)
                self.max_tokens = s.get("max_tokens", self.max_tokens)
                self.history = s.get("history", [])
            except Exception:
                pass

    def save_state(self):
        s = {
            "model_name": self.model_name,
            "temp": self.temp,
            "max_tokens": self.max_tokens,
            "history": self.history[-200:],
        }
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)

    def load_model(self):
        console.print(f"[bold]Loading model:[/bold] {self.model_name} on {self.device}")
        try:
            # If model_name points to a PEFT adapter (folder with adapter_config.json), handle that
            peft_config = None
            try:
                peft_config = PeftConfig.from_pretrained(self.model_name)
            except Exception:
                peft_config = None

            base_model = self.model_name
            if peft_config is not None:
                base_model = peft_config.base_model_name_or_path
                console.print(f"[yellow]Detected PEFT adapter. Base model: {base_model}[/yellow]")

            self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

            # Prepare load kwargs
            if self.device == "cuda":
                # try 8-bit first
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        base_model, load_in_8bit=True, device_map="auto"
                    )
                    self.model = prepare_model_for_kbit_training(self.model)
                except Exception:
                    self.model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(base_model)

            # If PEFT adapter path provided, load adapters
            if peft_config is not None:
                from peft import PeftModel

                self.model = PeftModel.from_pretrained(self.model, self.model_name)

            # pipeline
            device_id = 0 if self.device == "cuda" else -1
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device_id,
            )
        except Exception as e:
            console.print(f"[red]Failed loading model:[/red] {e}")
            sys.exit(1)

    def show_menu(self):
        table = Table(title="Local LLM Hub", show_header=False)
        table.add_row("1", "Chat with model")
        table.add_row("2", f"Set model (current: {self.model_name})")
        table.add_row("3", f"Set temperature (current: {self.temp})")
        table.add_row("4", f"Set max tokens (current: {self.max_tokens})")
        table.add_row("5", "Show history")
        table.add_row("6", "Save state")
        table.add_row("7", "Reload model")
        table.add_row("8", "Fine-tune (LoRA) on local JSONL dataset")
        table.add_row("9", "Exit")
        console.print(table)

    def chat(self):
        console.print("[bold green]Enter messages. Empty line to return.[/bold green]")
        while True:
            prompt = Prompt.ask("[cyan]You[/cyan]", default="")
            if prompt.strip() == "":
                break
            self.history.append({"role": "user", "text": prompt})
            # Build simple context
            ctx_text = ""
            for h in self.history[-6:]:
                role = "User" if h["role"] == "user" else "Assistant"
                ctx_text += f"{role}: {h['text']}\n"
            ctx_text += "Assistant:"

            try:
                out = self.pipe(
                    ctx_text,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temp,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                text = out[0]["generated_text"][len(ctx_text):].strip()
            except Exception as e:
                console.print(f"[red]Generation error:[/red] {e}")
                text = "[generation failed]"
            console.print(f"[magenta]Assistant:[/magenta] {text}")
            self.history.append({"role": "assistant", "text": text})

    def set_model(self):
        m = Prompt.ask("Model name (HF repo or local path)", default=self.model_name)
        if m != self.model_name:
            self.model_name = m
            self.save_state()
            console.print("[yellow]Reloading model...[/yellow]")
            self.load_model()

    def set_temp(self):
        t = Prompt.ask("Temperature (0.0-2.0)", default=str(self.temp))
        try:
            v = float(t)
            self.temp = max(0.0, min(2.0, v))
        except:
            console.print("[red]Invalid value[/red]")

    def set_max_tokens(self):
        t = Prompt.ask("Max new tokens (16-2048)", default=str(self.max_tokens))
        try:
            v = int(t)
            self.max_tokens = max(16, min(2048, v))
        except:
            console.print("[red]Invalid value[/red]")

    def show_history(self):
        for i, h in enumerate(self.history[-200:], start=1):
            role = "[cyan]You[/cyan]" if h["role"] == "user" else "[magenta]Assistant[/magenta]"
            console.print(f"{i}. {role}: {h['text']}")

    def finetune_lora(self):
        console.print("[bold]LoRA fine-tune workflow[/bold]")
        dataset_path = Prompt.ask("Path to JSONL dataset (each line: {\"prompt\":\"..\",\"response\":\"..\"})")
        if not os.path.exists(dataset_path):
            console.print("[red]Dataset file not found[/red]")
            return
        output_dir = Prompt.ask("Output adapter dir", default="lora_out")
        epochs = int(Prompt.ask("Epochs", default="3"))
        batch_size = int(Prompt.ask("Per-device batch size", default="1"))
        learning_rate = float(Prompt.ask("Learning rate", default="1e-4"))

        # Load dataset
        console.print("[yellow]Loading dataset...[/yellow]")
        ds = load_dataset("json", data_files=dataset_path, split="train")

        # Minimal validation of fields
        def check_example(ex):
            if "prompt" in ex and "response" in ex:
                return True
            return False

        if not all(check_example(x) for x in ds[: min(100, len(ds))]):
            console.print("[red]Dataset items must contain 'prompt' and 'response' fields[/red]")
            return

        # tokenizer + model load (base)
        console.print("[yellow]Preparing model and tokenizer for training...[/yellow]")
        base_model = self.model_name
        # If current model is PEFT adapter, use its base
        try:
            pconf = PeftConfig.from_pretrained(self.model_name)
            base_model = pconf.base_model_name_or_path
        except Exception:
            base_model = self.model_name

        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        model = None
        if self.device == "cuda":
            try:
                model = AutoModelForCausalLM.from_pretrained(base_model, load_in_8bit=True, device_map="auto")
                model = prepare_model_for_kbit_training(model)
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(base_model)

        # Setup LoRA
        console.print("[yellow]Configuring LoRA...[/yellow]")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"] ,  # common for many models; may need adjust
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        # Prepare dataset: concatenate prompt + response separated, tokenise
        max_length = 1024

        def build_example(ex):
            prompt = ex["prompt"].strip()
            response = ex["response"].strip()
            full = prompt + tokenizer.eos_token + response + tokenizer.eos_token
            tok = tokenizer(full, truncation=True, max_length=max_length, padding="max_length")
            # labels: mask prompt tokens to -100 so loss computed only on response (simple approach)
            prompt_tokens = tokenizer(prompt + tokenizer.eos_token, truncation=True, max_length=max_length)["input_ids"]
            labels = tok["input_ids"].copy()
            # mask prompt tokens
            for i in range(len(prompt_tokens)):
                if i < len(labels):
                    labels[i] = -100
            return {"input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"], "labels": labels}

        console.print("[yellow]Tokenizing dataset...[/yellow]")
        ds_tok = ds.map(build_example, remove_columns=ds.column_names, batched=False)

        # Data collator
        data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")

        # Training args
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            fp16=self.device == "cuda",
            logging_steps=10,
            save_total_limit=2,
            remove_unused_columns=False,
            gradient_accumulation_steps=1,
            save_strategy="epoch",
        )

        # Trainer
        console.print("[yellow]Starting training...[/yellow]")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds_tok,
            data_collator=data_collator,
        )

        try:
            trainer.train()
            console.print("[green]Training complete. Saving adapters...[/green]")
            model.save_pretrained(output_dir)
            console.print(f"[green]Adapters saved to {output_dir}[/green]")
            # After saving, user can load by setting model_name to output_dir
        except Exception as e:
            console.print(f"[red]Training failed:[/red] {e}")

    def run(self):
        while True:
            self.show_menu()
            choice = Prompt.ask("Choice", choices=[str(i) for i in range(1, 10)], default="1")
            if choice == "1":
                self.chat()
            elif choice == "2":
                self.set_model()
            elif choice == "3":
                self.set_temp()
            elif choice == "4":
                self.set_max_tokens()
            elif choice == "5":
                self.show_history()
            elif choice == "6":
                self.save_state()
                console.print("[green]State saved.[/green]")
            elif choice == "7":
                console.print("[yellow]Reloading model...[/yellow]")
                self.load_model()
            elif choice == "8":
                self.finetune_lora()
            elif choice == "9":
                self.save_state()
                console.print("[bold]Goodbye[/bold]")
                break


if __name__ == "__main__":
    hub = Hub()
    hub.run()
