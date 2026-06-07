# Coverage vs faithfulness (llm extractor)

| Model | Lang | Precision (faith) | Recall (coverage) | F1 | claims/inst |
|---|---|---|---|---|---|
| DeepSeek-V3.2 | en | 0.824 | 0.614 | 0.704 | 6.37 |
| DeepSeek-V3.2 | es | 0.876 | 0.762 | 0.815 | 8.1 |
| DeepSeek-V3.2 | pt | 0.855 | 0.788 | 0.820 | 8.59 |
| gemini-2.5-pro | en | 0.884 | 0.268 | 0.411 | 1.81 |
| gemini-2.5-pro | es | 0.682 | 0.124 | 0.210 | 0.98 |
| gemini-2.5-pro | pt | 0.751 | 0.126 | 0.216 | 0.91 |
| gpt-5.4-mini | en | 0.812 | 0.395 | 0.531 | 4.53 |
| gpt-5.4-mini | es | 0.840 | 0.459 | 0.594 | 4.52 |
| gpt-5.4-mini | pt | 0.817 | 0.418 | 0.553 | 4.36 |
| gpt-5.5 | en | 0.850 | 0.489 | 0.621 | 5.27 |
| gpt-5.5 | es | 0.874 | 0.498 | 0.635 | 5.14 |
| gpt-5.5 | pt | 0.872 | 0.489 | 0.627 | 5.32 |
| grok-4.3 | en | 0.840 | 0.360 | 0.504 | 3.63 |
| grok-4.3 | es | 0.865 | 0.434 | 0.578 | 4.13 |
| grok-4.3 | pt | 0.867 | 0.483 | 0.620 | 4.51 |

EN ranking by precision: ['gemini-2.5-pro', 'gpt-5.5', 'grok-4.3', 'DeepSeek-V3.2', 'gpt-5.4-mini']
EN ranking by F1:        ['DeepSeek-V3.2', 'gpt-5.5', 'gpt-5.4-mini', 'grok-4.3', 'gemini-2.5-pro']
Ranking changes when coverage is required: True
