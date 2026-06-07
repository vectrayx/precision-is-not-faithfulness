# Coverage vs faithfulness (llm extractor)

| Model | Lang | Precision (faith) | Recall (coverage) | F1 | claims/inst |
|---|---|---|---|---|---|
| DeepSeek-V3.2 | en | 0.482 | 0.913 | 0.631 | 14.39 |
| gemini-2.5-pro | en | 0.941 | 0.528 | 0.677 | 2.71 |
| gpt-5.4-mini | en | 0.504 | 0.542 | 0.522 | 9.12 |
| gpt-5.5 | en | 0.534 | 0.598 | 0.564 | 9.51 |
| grok-4.3 | en | 0.536 | 0.538 | 0.537 | 8.45 |

EN ranking by precision: ['gemini-2.5-pro', 'grok-4.3', 'gpt-5.5', 'gpt-5.4-mini', 'DeepSeek-V3.2']
EN ranking by F1:        ['gemini-2.5-pro', 'DeepSeek-V3.2', 'gpt-5.5', 'grok-4.3', 'gpt-5.4-mini']
Ranking changes when coverage is required: True
