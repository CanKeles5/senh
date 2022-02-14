'''
Script to train models from different frameworks

To-Do:
- Train ConvTasNet (Asteroid)
    - Add option to load model & fine tune

- Train DPTNet (Asteroid)

- Train Demucs (Demucs repo)

- Train SSE (self-supervised senh)
    - Refactor code to newer Pytorch versions
    - Add option to only train supervised (not priority right now)
    - Try to see if different model arch. can be used for the same script

----------------------This is not priority
- Train SepFormer (SpeechBrain) -> The code seems broken

- Train custom model (SpeechBrain)

- For all models, have the same API
'''


def train_model(model_name: str):
    model_name = model_name.lower()
    
    if model_name == "convtasnet":
        import training_scripts.convtasnet.train as train_convtasnet
        train_convtasnet.main()
    if model_name == "dptnet":
        import training_scripts.dptnet.train as train_dptnet
        train_dptnet.main()
    elif model_name == "sepformer":
        pass
    elif model_name == "demucs":
        pass
    elif model_name == "custom": #Should provide model somehow, maybe in custom_model.py
        pass
    else
        print(f"Error: model '{model_name}' not defined.")
    pass
    
