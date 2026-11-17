"""Qwen2.5-0.5B-Instruct generation wrapper (frozen weights, inference only)."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"  # HF commit, verified 2026-11-17

# Greedy for attempt 0 (deterministic baseline). Sampled attempts use the official
# generation_config.json values of this revision (temperature/top_p/top_k/repetition_penalty).
GREEDY_PARAMS = {"do_sample": False, "repetition_penalty": 1.0}
SAMPLE_PARAMS = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.1}
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS_SOLVER = 512
MAX_NEW_TOKENS_POT = 320


@dataclass
class Generation:
    text: str
    finished: bool          # ended with EOS (False = hit max_new_tokens -> truncated)
    input_tokens: int
    output_tokens: int
    latency_s: float        # amortized: batch wall time / sequences in batch (GPU, batched)


def derive_seed(*parts) -> int:
    h = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).digest()
    return int.from_bytes(h[:4], "big")


class QwenSolver:
    """Loads the model once; `generate` is batched with left padding."""

    def __init__(self, device: str | None = None, dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = getattr(torch, dtype) if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=torch_dtype)
        self.model.to(self.device).eval()
        self.dtype = str(torch_dtype).replace("torch.", "")
        self.eos_ids = set(self.model.generation_config.eos_token_id or [])
        if isinstance(self.model.generation_config.eos_token_id, int):
            self.eos_ids = {self.model.generation_config.eos_token_id}

    def info(self) -> dict:
        import transformers
        return {"model_id": MODEL_ID, "revision": MODEL_REVISION, "device": self.device, "dtype": self.dtype,
                "torch": self.torch.__version__, "transformers": transformers.__version__,
                "gpu": self.torch.cuda.get_device_name(0) if self.device == "cuda" else None}

    def render(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, message_lists: list[list[dict]], *, params: dict, max_new_tokens: int,
                 num_return_sequences: int = 1, seed: int | None = None) -> list[list[Generation]]:
        """Returns, for each prompt, a list of `num_return_sequences` generations."""
        torch = self.torch
        texts = [self.render(m) for m in message_lists]
        enc = self.tokenizer(texts, return_tensors="pt", padding=True)
        lengths = enc["attention_mask"].sum(dim=1).tolist()
        if max(lengths) > MAX_INPUT_TOKENS:
            raise ValueError(f"prompt longer than MAX_INPUT_TOKENS={MAX_INPUT_TOKENS}: {max(lengths)}")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        if seed is not None:
            torch.manual_seed(seed)
            if self.device == "cuda":
                torch.cuda.manual_seed_all(seed)
        gen_kwargs = dict(params)
        if not gen_kwargs.get("do_sample"):
            gen_kwargs.update(temperature=None, top_p=None, top_k=None)  # silence sampling defaults
        else:
            # Guard against NaN/inf logits crashing CUDA multinomial sampling (observed once:
            # "illegal memory access" during sampling). No effect when logits are finite.
            gen_kwargs.update(remove_invalid_values=True, renormalize_logits=True)
        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences,
                                      pad_token_id=self.tokenizer.pad_token_id, **gen_kwargs)
        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        prompt_len = enc["input_ids"].shape[1]
        new = out[:, prompt_len:].tolist()
        per_seq = elapsed / len(new)
        results: list[list[Generation]] = []
        for i, n_in in enumerate(lengths):
            group = []
            for j in range(num_return_sequences):
                ids = new[i * num_return_sequences + j]
                n_out, finished = 0, False
                for t in ids:
                    n_out += 1
                    if t in self.eos_ids:
                        finished = True
                        break
                text = self.tokenizer.decode(ids[:n_out], skip_special_tokens=True)
                group.append(Generation(text=text, finished=finished, input_tokens=int(n_in),
                                        output_tokens=n_out, latency_s=per_seq))
            results.append(group)
        return results
