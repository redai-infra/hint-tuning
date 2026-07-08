# Hint Tuning: Less Data Makes Better Reasoners

[![arXiv](https://img.shields.io/badge/arXiv-2605.08665-b31b1b.svg)](https://arxiv.org/abs/2605.08665)
[![Model 4B](https://img.shields.io/badge/🤗%20Model-4B-blue)](https://huggingface.co/redai-infra/hint-tuning-4b)
[![Model 7B](https://img.shields.io/badge/🤗%20Model-7B-blue)](https://huggingface.co/redai-infra/hint-tuning-7b)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-hint--tuning--1k-blue)](https://huggingface.co/datasets/redai-infra/hint-tuning-1k)

Official code and data for **Hint Tuning**, a lightweight SFT data construction method that constructs long and short chain-of-thought traces by using the corresponding instruct model as an ideal difficulty probe: the minimal reasoning hint required for the instruct model to solve a problem directly reflects how hard that problem is, and determines the length of CoT assigned to it.


---

## Released Resources

| Resource | Link |
|---|---|
| Hint-Tuning-4B (fine-tuned from [Qwen3-4B-Thinking](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)) | [🤗 HuggingFace](https://huggingface.co/redai-infra/hint-tuning-4b) |
| Hint-Tuning-7B (fine-tuned from [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)) | [🤗 HuggingFace](https://huggingface.co/redai-infra/hint-tuning-7b) |
| hint_tuning_1k dataset | [🤗 HuggingFace](https://huggingface.co/datasets/redai-infra/hint-tuning-1k) |

---

## Data

The `data/` directory contains two files:

| File | Description |
|---|---|
| `data/problems.json` | 1,000 raw problems and gold answers sourced from [s1K-1.1](https://arxiv.org/abs/2501.19393) |
| `data/hint_tuning_1k.json` | The constructed 1K SFT dataset (download below) |

> **Download `hint_tuning_1k.json`:** [🤗 HuggingFace](https://huggingface.co/datasets/redai-infra/hint-tuning-1k)

Each record in `hint_tuning_1k.json` follows the Alpaca format:

```json
{
  "instruction": "Let $f(x) = x^2 + ...$",
  "input": "",
  "output": "<think>\nI may need some deep thinking.\n...\n</think>\n\nThe answer is $\\boxed{42}$."
}
```

The `<think>` prefix encodes the reasoning state assigned during data construction (see below).

---

## Data Construction

The 1,000 problems are drawn from [s1K](https://arxiv.org/abs/2501.19393).
The corresponding instruct model serves as an ideal difficulty probe: the minimal hint prefix from the think model's trace that allows the instruct model to reach the correct answer measures problem difficulty, and directly determines the length of CoT assigned to each problem.

```
Step 1 — Both models attempt all problems independently.

Step 2 — For problems the instruct model cannot solve alone,
         inject cumulative prefixes from the think model's trace
         and ask the instruct model to complete from there.
         Grading (LLM-as-judge) determines the minimal prefix k
         that leads to a correct answer.

Step 3 — Classify each problem:

  instruct correct (k=0)  → State 1 – No-Hint
                              <think>Let me think. ...</think>

  instruct correct (k>0)  → State 2 – Sparse-Hint
                              <think>I may need some deep thinking. [prefix]...</think>

  no prefix worked        → State 3 – Full-Hint (fall back to full think trace)
                              <think>This is a complex or challenging question... [full trace]</think>
```

### Models used in the paper

| Role | Model |
|---|---|
| Think model | [Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) |
| Instruct model | [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) |
| LLM-as-judge grader | Qwen3-4B-Instruct-2507 — local vLLM server |

### Reproducing the dataset

**Dependencies:** vLLM · transformers · openai · datasets

Start the grader server before running any pipeline step:

```bash
CUDA_VISIBLE_DEVICES=4,5 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --tensor-parallel-size 2 --max-model-len 32768 \
  --port 8001 --served-model-name grader
```

#### Step 1 — Both models attempt all problems

```bash
# Think model
python construction/pipeline.py \
  --mode        think \
  --think-model Qwen/Qwen3-4B-Thinking-2507 \
  --dataset     data/problems.json \
  --config      construction/config.yaml \
  --output-dir  output/

# Instruct model (no prefix)
python construction/pipeline.py \
  --mode           instruct \
  --instruct-model Qwen/Qwen3-4B-Instruct-2507 \
  --think-results  output/think_results.json \
  --config         construction/config.yaml \
  --output-dir     output/

# Grade instruct results to identify which problems need a prefix
python construction/pipeline.py \
  --mode          grade \
  --think-results output/think_results.json \
  --instruct-models-config construction/instruct_models.yaml \
  --output-dir    output/
```

#### Step 2 — Find the minimal hint prefix for hard problems



```bash
python construction/pipeline.py \
  --mode           prefix \
  --think-results  output/think_results.json \
  --think-grading  output/llm_grading_think.json \
  --instruct-models-config construction/instruct_models.yaml \
  --config         construction/config.yaml \
  --output-dir     output/
```


#### Step 3 — Classify and merge into SFT format

```bash
python construction/merge.py \
  --think    output/think_results.json \
  --grading  output/llm_grading_think.json \
  --instruct output/instruct_results.json \
  --prefix   output/k_prefix.json \
  --output   data/hint_tuning_1k.json
```

---

## SFT Training

Our experiments use [Relax](https://github.com/redai-infra/Relax), an open-source post-training framework supporting both SFT and RL, [related commit](https://github.com/redai-infra/Relax/commit/5f86ed96786022640c48143bddcb465eb4679682), [run scripts](https://github.com/redai-infra/Relax/tree/main/examples/cot_compression).
The dataset (`hint_tuning_1k.json`) is in **Alpaca format** (`instruction` / `input` / `output` fields).

Training hyperparameters follow [s1](https://arxiv.org/abs/2501.19393). 

---

## Evaluation

We evaluate using [lighteval](https://github.com/huggingface/lighteval) with a vLLM backend.

**Install:** `pip install lighteval[vllm] inspect-ai`

**Benchmarks:** AIME24, AIME25, HMMT25, MATH-500.

```bash
bash evaluation/eval.sh Qwen/hint-tuning-7b output/eval_results
```

The script automatically loads `evaluation/custom_tasks.py` via `--custom-tasks`, which defines the prompt format used at training time:

```
{problem}

Please reason step by step, and put your final answer within \boxed{}.
```

Use this script — not lighteval's built-in task names — to reproduce our numbers. Lighteval's default prompts differ from the above and will produce inconsistent results.

The script also exports `EVAL_MODEL_PATH` so `custom_tasks.py` can load the correct tokenizer for measuring output token length.

**Note on instruction robustness:** The 1K dataset uses a fixed prompt style (math-oriented, `\boxed{}` format). If you want the model to generalize to a wider variety of instruction phrasings, synthesize additional prompt variants on top of the 1K samples before training — e.g. replacing the instruction with paraphrases like `"Solve:"`, `"Think step by step."`, `"Q: … A:"`, etc.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{fan2026hint,
  title={Hint Tuning: Less Data Makes Better Reasoners},
  author={Fan, Siqi and Li, Minghao and Ma, Xiaoqian and Huang, Xiusheng and Chen, Zhuo and Qin, Bowen and Zhang, Liujie and Shang, Shuo and Chen, Weihang},
  journal={arXiv preprint arXiv:2605.08665},
  year={2026}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).

---

## Acknowledgements

We are grateful to the authors of [s1](https://arxiv.org/abs/2501.19393) for curating and open-sourcing the s1K problem set that forms the foundation of our dataset, and to the [Relax](https://arxiv.org/abs/2604.11554) team for building and maintaining the post-training framework used in our experiments.

```bibtex
@inproceedings{muennighoff2025s1,
  title={s1: Simple test-time scaling},
  author={Muennighoff, Niklas and Yang, Zitong and Shi, Weijia and Li, Xiang Lisa and Fei-Fei, Li and Hajishirzi, Hannaneh and Zettlemoyer, Luke and Liang, Percy and Cand{\`e}s, Emmanuel and Hashimoto, Tatsunori B},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={20286--20332},
  year={2025}
}
```

```bibtex
@software{relax2026,
  title  = {Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale},
  author = {Relax Contributors},
  url    = {https://arxiv.org/abs/2604.11554},
  year   = {2026}
}
```


