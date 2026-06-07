# Coverage vs faithfulness (llm extractor)

| Model | Lang | Precision (faith) | Recall (coverage) | F1 | claims/inst |
|---|---|---|---|---|---|
| DeepSeek-V3.2 | en | 0.847 | 0.883 | 0.864 | 12.62 |
| gemini-2.5-pro | en | 0.767 | 0.080 | 0.145 | 0.55 |
| gpt-5.4-mini | en | 0.868 | 0.486 | 0.623 | 7.51 |
| gpt-5.5 | en | 0.907 | 0.544 | 0.680 | 9.02 |
| grok-4.3 | en | 0.882 | 0.537 | 0.668 | 8.05 |

EN ranking by precision: ['gpt-5.5', 'grok-4.3', 'gpt-5.4-mini', 'DeepSeek-V3.2', 'gemini-2.5-pro']
EN ranking by F1:        ['DeepSeek-V3.2', 'gpt-5.5', 'grok-4.3', 'gpt-5.4-mini', 'gemini-2.5-pro']
Ranking changes when coverage is required: True
