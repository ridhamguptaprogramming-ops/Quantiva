# Quantiva LLM Framework — Build Plan

## Phase 1 — Tokenizer (`quantiva/tokenizer/`)
- [x] base.py (abstract Tokenizer interface)
- [x] bpe.py (BPE trainer from scratch)
- [x] sentencepiece_wrapper.py (SentencePiece support)
- [x] tiktoken_wrapper.py (tiktoken compatibility)
- [x] factory.py (tokenizer factory)
- [x] evaluate.py (tokenizer evaluation)
- [x] __init__.py

## Phase 2 — Model (`quantiva/model/`)
- [ ] config.py (model configuration dataclass)
- [ ] embedding.py (token + rotary position embeddings)
- [ ] normalization.py (LayerNorm + RMSNorm)
- [ ] rotary_embedding.py (RoPE)
- [ ] attention.py (MHA + GQA + Flash Attention + KV Cache + causal mask)
- [ ] mlp.py (GELU + SwiGLU)
- [ ] transformer_block.py (pre/post norm blocks, residual)
- [ ] transformer.py (full transformer stack)
- [ ] gpt.py (GPT LM head, weight init, param count, MFU)
- [ ] __init__.py

## Phase 3 — Data (`quantiva/data/`, `quantiva/datasets/`)
- [ ] dataset.py (token streaming, memmap dataset)
- [ ] dataloader.py (batch sampler, multi-GPU)
- [ ] preprocessing/chunking.py (document chunking)
- [ ] preprocessing/formatting.py (chat templates, SFT formatting)
- [ ] __init__.py files

## Phase 4 — Training (`quantiva/training/`)
- [ ] trainer.py (grad accum, AMP, checkpoint, resume, LR scheduler, clipping, W&B)
- [ ] pretrain.py (pretraining entry)
- [ ] sft.py (supervised fine-tuning)
- [ ] dpo.py (direct preference optimization)
- [ ] grpo.py (group relative policy optimization)
- [ ] lora.py (LoRA/QLoRA)
- [ ] rl.py (RLHF-style reinforcement learning)
- [ ] distributed.py (DDP/FSDP/DeepSpeed helpers)
- [ ] __init__.py

## Phase 5 — Inference (`quantiva/inference/`)
- [ ] sampler.py (temperature/top-k/top-p/min-p/beam)
- [ ] kv_cache.py (KV cache)
- [ ] generate.py (autoregressive generation, speculative decoding, batch)
- [ ] streamer.py (token streaming)
- [ ] quantize.py (quantized inference hooks)
- [ ] __init__.py

## Phase 6 — RAG / Tools / Eval (`quantiva/rag/`, `quantiva/tools/`, `quantiva/evaluation/`)
- [ ] rag/loaders.py (PDF, Markdown)
- [ ] rag/chunking.py
- [ ] rag/embeddings.py
- [ ] rag/retriever.py (FAISS, ChromaDB)
- [ ] rag/rag.py (retrieval pipeline)
- [ ] tools/base.py (tool abstraction)
- [ ] tools/python_executor.py
- [ ] tools/calculator.py
- [ ] tools/web_search.py (hook)
- [ ] tools/image_gen.py (hook)
- [ ] tools/function_calling.py (JSON schema)
- [ ] evaluation/benchmarks.py (perplexity)
- [ ] evaluation/gsm8k.py
- [ ] evaluation/humaneval.py
- [ ] evaluation/mmlu.py

## Phase 7 — API (`quantiva/api/`)
- [ ] server.py (FastAPI app)
- [ ] openai_compat.py (OpenAI-compatible endpoints)
- [ ] websockets.py (streaming)
- [ ] schemas.py (pydantic models)
- [ ] __init__.py

## Phase 8 — Frontend (`frontend/`)
- [ ] Next.js + React + Tailwind + TypeScript chat UI
- [ ] Dark mode, streaming, markdown, syntax highlight, code copy
- [ ] Image upload, drag & drop, voice input, responsive

## Phase 9 — Docker / Configs / Scripts / Tests / Docs
- [ ] Dockerfile (api/train/frontend)
- [ ] docker-compose.yml (Redis, Postgres, Nginx)
- [ ] configs/*.yaml
- [ ] scripts/ (entrypoints, data prep)
- [ ] tests/ (unit tests)
- [ ] docs/ (README, architecture, training, inference, deployment, fine-tuning, API, developer guide)

