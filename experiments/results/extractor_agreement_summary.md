# Extractor robustness: regex (model-free) vs LLM (gpt-5.x) extractor

System-level rank agreement over 5 models (EN, model-free regex): Spearman=0.8, Pearson=0.9219; top model agrees: True. (Avg over EN/ES/PT: Spearman=-0.3 -- regex's light ES/PT patterns make it an EN-first check.)

Instance-level (overall, N=1527): Pearson=0.4914, Spearman=0.5614.

| Model | Regex faith | LLM faith | Regex EN | LLM EN |
|---|---|---|---|---|
| DeepSeek-V3.2 | 0.750 | 0.863 | 0.735 | 0.838 |
| gemini-2.5-pro | 0.803 | 0.824 | 0.925 | 0.901 |
| gpt-5.4-mini | 0.755 | 0.829 | 0.742 | 0.819 |
| gpt-5.5 | 0.783 | 0.851 | 0.753 | 0.843 |
| grok-4.3 | 0.788 | 0.864 | 0.749 | 0.855 |

| Lang | N | Pearson | Spearman |
|---|---|---|---|
| en | 564 | 0.4984 | 0.5857 |
| es | 503 | 0.3728 | 0.4768 |
| pt | 460 | 0.5869 | 0.6193 |

Claim volume: regex=7792, LLM=12855 (regex extracts 61% as many claims).
