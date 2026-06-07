# Coverage vs faithfulness (regex extractor)

| Model | Lang | Precision (faith) | Recall (coverage) | F1 | claims/inst |
|---|---|---|---|---|---|
| DeepSeek-V3.2 | en | 0.735 | 0.245 | 0.368 | 2.14 |
| gemini-2.5-pro | en | 0.925 | 0.180 | 0.302 | 1.07 |
| gpt-5.4-mini | en | 0.742 | 0.386 | 0.508 | 3.5 |
| gpt-5.5 | en | 0.753 | 0.448 | 0.561 | 3.93 |
| grok-4.3 | en | 0.749 | 0.179 | 0.289 | 1.61 |

EN ranking by precision: ['gemini-2.5-pro', 'gpt-5.5', 'grok-4.3', 'gpt-5.4-mini', 'DeepSeek-V3.2']
EN ranking by F1:        ['gpt-5.5', 'gpt-5.4-mini', 'DeepSeek-V3.2', 'gemini-2.5-pro', 'grok-4.3']
Ranking changes when coverage is required: True
