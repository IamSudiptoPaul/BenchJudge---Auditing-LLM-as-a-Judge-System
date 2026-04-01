# BenchJudge---Auditing-LLM-as-a-Judge-System
The project will create an open-source toolkit, allowing researchers to diagnose the reliability of their AI evaluators, thereby making automated assessment more transparent, trustworthy, and scientifically rigorous. 


🏗 Directory Structure & Design

 Project Structure
```text
BenchJudge/
├── datasets/                   # Raw benchmark JSONL files
├── scripts/                    # Execution pipeline (1.0 to 6.0)
├── outputs/                    # Processed data & logs
│   ├── [dataset_name]/
│   │   ├── model_answers/      # Formatted pairs for judging
│   │   └── judge_scores/       # Final audit results
├── plots/                      # Exported visualization figures
├── requirements.txt            # System dependencies
└── README.md                   # Project documentation
```

🛠 Technical Core & Requirements

To execute the BenchJudge framework, the system integrates a blend of technical and analytical competencies:

Technical Implementation: Built primarily in Python, the framework handles complex data processing, workflow automation, and multi-source API integration. It supports both cloud-based LLM APIs (OpenAI, Anthropic, Gemini) and local model deployment via Hugging Face and Ollama.

Statistical & Quantitative Modeling: The framework is designed to move toward Hierarchical Bayesian Models using probabilistic programming (such as PyStan or NumPyro). This allows for the interpretation of posterior distributions to understand judge reliability rather than relying on static percentages.

Research Rigor: BenchJudge emphasizes systematic experimental design, hypothesis formulation, and version-controlled reproducibility via Git to ensure all meta-evaluations meet scientific standards.


📂 Project Pipeline & Execution Guide

The BenchJudge framework follows a strict four-stage pipeline. Each stage corresponds to a specific script in the scripts/ directory.

1. Data Preparation

Before auditing, raw benchmark files are pre-processed into a unified pairwise format.

Task: Convert MT-Bench, Vicuna, or Chatbot Arena results into a judge-ready jsonl.

Command: ```bash
python scripts/1_data_prep.py

Output: /outputs/[dataset]/model_answers/judge_input_pairs.jsonl

To reproduce an audit, run the scripts in this specific order. Each step generates the data required for the next.

2.4_groq_judge.py: The core auditing script. It runs both the Forward and Reverse passes to collect judge verdicts. Results are saved to outputs/.../judge_scores/.
4_analyze_stats.py: Parses the judge scores to calculate overall Consistency Rates and Bias percentages. Output is displayed in the console.
5_winrates.py: Processes results to generate comparative bar charts of model performance. Saves to plots/win_rates.png.
5.1_radar_plots.py: Visualizes the "Judge Footprint" across different categories (Reasoning, Coding, etc.). Saves to plots/radar_comparison.png.
6_qual_report.py: Extracts specific examples of consistent vs. biased judgments for manual audit. Generates qualitative_audit_report.txt.

2. The Multi-Pass Audit (Core Task)
This is where the actual LLM-as-a-Judge execution happens. The script performs a Forward Pass and a Reverse Pass (swapping Model A and B) to detect bias.

Task: Execute judging using Groq (Llama-3.1, Qwen) or local models.

Command: ```bash
python scripts/2.4_groq_judge.py

Key Feature: Includes automatic rate-limit handling and {ref_answer_1} key-error protection for math prompts.

Output: /outputs/[dataset]/judge_scores/[dataset]_answers_[model]_judge.jsonl

3. Statistical Meta-Analysis
Once the results are gathered, we calculate the consistency and bias metrics.

Task: Generate Consistency Rates, A-Bias vs. B-Bias percentages, and Position Dominance scores.

Command: ```bash
python scripts/4_analyze_consistency.py

Output: Console summary of percentage-based failure modes.

API Configuration
The framework supports cloud-based APIs (Groq, OpenAI, Gemini). Create a .env file in the root directory to manage your keys securely: 
GROQ_API_KEY=your_api_key_here

4. Visualization & Reporting
Transform raw data into dissertation-ready figures.

Task: Generate Win Rate bar charts, Radar charts by category, and Qualitative report samples.

Commands: ```bash
python scripts/5_winrates.py       # Bar charts
python scripts/5.1_radar_plots.py  # Radar charts (Llama vs Qwen)
python scripts/6_qual_report.py    # Text-based audit report

Output: .png charts/ plots and qualitative_audit_report.txt.

BenchJudge isolates structural biases by presenting the same pair of responses to a judge but reversing their order (Position A vs. Position B).
CONSISTENT: The judge selects the same answer regardless of order.
BIASED: The judge changes its verdict based on position, indicating a failure in objective assessment.

The current implementation is optimized for:
Llama-3.1-8B-Instant (via Groq)
Qwen-2.5-Coder-32B (via Groq)
Gemini 1.5 Flash


🔬 Experimental Methodology: The "Audit"

The framework utilizes a Swapped-Pair methodology to isolate structural biases. By presenting the same pair of model responses to a judge but reversing their order (Position A vs. Position B), BenchJudge can calculate exactly how much the judge is influenced by the "slot" rather than the content.

Current Progress & Findings

Recent audits conducted on MT-Bench and Chatbot Arena datasets using local judges (Llama-3.1-8B, Phi-3.5, Qwen-2.5-1.5B) and cloud judges (Gemini 1.5 Flash) revealed several critical failure modes in automated evaluation:

Significant Position Bias: Llama-3.1-8B exhibited a 62.5% Position Bias on MT-Bench. This indicates the model is statistically more likely to select "Model A" simply because it appears first, failing to objectively assess quality when the same answers are swapped.

The "Judgment Collapse": Smaller models like Phi-3.5 proved unsuitable for complex meta-evaluation. In the Chatbot Arena task, it yielded zero matched pairs due to Structural Instability, resulting in malformed JSON or context-window exhaustion when processing a prompt followed by two long-form answers.

Left-Side Bias in Small Models: Testing with Qwen-2.5-1.5B showed a "Win A" rate roughly four times higher than "Win B." This suggests a "lazy" bias where smaller models struggle to process the full context of both answers and default to the first information they consume.

Conservative Bias (The Tie Anomaly): In specific audits, Gemini 1.5 Flash returned a 100% Tie Rate. This suggests a "Conservative Bias" likely triggered by overly strict system prompts or an extraction failure where the judge provides a long explanation but fails to commit to a definitive verdict.


📑 Key Discussion Points for Research

The data gathered through BenchJudge supports several emerging theories in LLM-as-a-Judge research:

Model Size Correlation: There is a direct link between model parameters and "lazy" biases. Models at or below the 8B threshold frequently fall back on position-based heuristics compared to larger frontier models.

Benchmark Sensitivity: Consistency scores often improve on datasets like Chatbot Arena (54.5%) compared to MT-Bench (37.5%). This suggests that the nature of the questions or the specific signal-to-noise ratio in response length can help a judge overcome its inherent structural biases.

Context Exhaustion: The failure of small-context models in comparison tasks is often a byproduct of the "Question + Two Answers" structure, which exceeds their effective reasoning capability, leading to a total collapse of the evaluation framework.


📦 Installation

git clone [https://github.com/IamSudiptoPaul/BenchJudge---Auditing-LLM-as-a-Judge-System.git.git](https://github.com/IamSudiptoPaul/BenchJudge---Auditing-LLM-as-a-Judge-System.git)

pip install -r requirements.txt
