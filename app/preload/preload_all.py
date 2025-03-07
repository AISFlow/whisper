import whisper

def load_whisper_models():
    model_names = ['base', 'large', 'medium', 'small', 'tiny', 'turbo']
    models = {}
    for name in model_names:
        try:
            print(f"Loading model: {name}...")
            models[name] = whisper.load_model(name)
            print(f"Model '{name}' loaded successfully.\n")
        except Exception as e:
            print(f"Error loading model '{name}': {e}\n")
    return models

if __name__ == "__main__":
    loaded_models = load_whisper_models()
