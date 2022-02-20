# senh
Speech enhancement package to create datasets, evaluate models, train models (To-do).

Main features:
- Createa custom datasets with specific SNR rates and reverb level
- Evaulate models for different cases such as clean noise, reverbarant noise, speaker with specific background SNR
- Train models on custom dataset, options to fine tune models from HuggingFace. Currently supporting ConvTasNet and DPTNet

Dataset for model evaluation can be found here: https://huggingface.co/datasets/cankeles/eval-senh
