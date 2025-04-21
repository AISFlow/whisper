import whisper
import gc

def preload_whisper_models():
    model_names = ['base', 'large', 'medium', 'small', 'tiny', 'turbo']
    for name in model_names:
        try:
            print(f"Loading model: {name}...")
            model = whisper.load_model(name)
            print(f"Model '{name}' loaded successfully.")
            
            del model
            gc.collect()
            print(f"Model '{name}' unloaded.\n")

        except Exception as e:
            print(f"Error loading model '{name}': {e}\n")

if __name__ == "__main__":
    preload_whisper_models()
