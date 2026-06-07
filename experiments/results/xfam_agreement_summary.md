# Cross-family extractor agreement: gpt-5.x vs deepseek-v32

System-level (avg over langs, N=5 models): Spearman=1.0, Pearson=0.9657.

Instance-level (overall, N=1090): Pearson=0.8173, Spearman=0.8383.

| Model | gpt-5.x | xfam |
|---|---|---|
| DeepSeek-V3.2 | 0.863 | 0.861 |
| gemini-2.5-pro | 0.824 | 0.804 |
| gpt-5.4-mini | 0.829 | 0.825 |
| gpt-5.5 | 0.851 | 0.858 |
| grok-4.3 | 0.864 | 0.869 |

| Lang | sys Spearman | inst N | inst Pearson | inst Spearman |
|---|---|---|---|---|
| en | 1.0 | 433 | 0.7646 | 0.8283 |
| es | 1.0 | 343 | 0.8272 | 0.8399 |
| pt | 0.7 | 314 | 0.8603 | 0.8472 |
