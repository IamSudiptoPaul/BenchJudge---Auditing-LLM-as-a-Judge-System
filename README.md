# BenchJudge---Auditing-LLM-as-a-Judge-System
The project will create an open-source toolkit, allowing researchers to diagnose the reliability of their AI evaluators, thereby making automated assessment more transparent, trustworthy, and scientifically rigorous. 

🛠 Technical Core & Requirements
To execute the BenchJudge framework, the system integrates a blend of technical and analytical competencies:

Technical Implementation: Built primarily in Python, the framework handles complex data processing, workflow automation, and multi-source API integration. It supports both cloud-based LLM APIs (OpenAI, Anthropic, Gemini) and local model deployment via Hugging Face and Ollama.

Statistical & Quantitative Modeling: The framework is designed to move toward Hierarchical Bayesian Models using probabilistic programming (such as PyStan or NumPyro). This allows for the interpretation of posterior distributions to understand judge reliability rather than relying on static percentages.

Research Rigor: BenchJudge emphasizes systematic experimental design, hypothesis formulation, and version-controlled reproducibility via Git to ensure all meta-evaluations meet scientific standards.

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
