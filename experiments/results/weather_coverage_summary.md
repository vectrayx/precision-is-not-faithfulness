# Weather: coverage vs faithfulness (second domain)

| Model | Lang | Precision | Recall | F1 | cl/inst |
|---|---|---|---|---|---|
| DeepSeek-V3.2 | en | 0.865 | 0.843 | 0.854 | 5.47 |
| DeepSeek-V3.2 | es | 0.910 | 0.653 | 0.761 | 4.19 |
| DeepSeek-V3.2 | pt | 0.854 | 0.413 | 0.557 | 2.83 |
| gemini-2.5-pro | en | 0.912 | 0.250 | 0.393 | 1.34 |
| gemini-2.5-pro | es | 0.836 | 0.215 | 0.342 | 1.09 |
| gemini-2.5-pro | pt | 0.868 | 0.208 | 0.336 | 1.03 |
| gpt-5.4-mini | en | 0.944 | 0.462 | 0.620 | 2.85 |
| gpt-5.4-mini | es | 0.916 | 0.442 | 0.596 | 2.83 |
| gpt-5.4-mini | pt | 0.953 | 0.460 | 0.621 | 2.87 |
| gpt-5.5 | en | 0.936 | 0.470 | 0.626 | 2.97 |
| gpt-5.5 | es | 0.932 | 0.458 | 0.615 | 2.85 |
| gpt-5.5 | pt | 0.946 | 0.457 | 0.616 | 2.86 |
| grok-4.3 | en | 0.939 | 0.473 | 0.629 | 2.93 |
| grok-4.3 | es | 0.948 | 0.460 | 0.620 | 2.89 |
| grok-4.3 | pt | 0.974 | 0.450 | 0.616 | 2.82 |

EN ranking by precision: ['gpt-5.4-mini', 'grok-4.3', 'gpt-5.5', 'gemini-2.5-pro', 'DeepSeek-V3.2']
EN ranking by F1:        ['DeepSeek-V3.2', 'grok-4.3', 'gpt-5.5', 'gpt-5.4-mini', 'gemini-2.5-pro']
Ranking changes when coverage required: True
