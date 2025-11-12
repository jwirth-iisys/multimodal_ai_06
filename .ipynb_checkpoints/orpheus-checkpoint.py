import torch
import soundfile as sf
from chatterbox.tts import ChatterboxTTS
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import IPython
MODEL_REPO = "SebastianBodza/Kartoffelbox-v0.1" 
T3_CHECKPOINT_FILE = "t3_cfg.safetensors"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = ChatterboxTTS.from_pretrained(device=device)

print("Downloading and applying German patch...")
checkpoint_path = hf_hub_download(repo_id=MODEL_REPO, filename=T3_CHECKPOINT_FILE)

t3_state = load_file(checkpoint_path, device="cpu") 

model.t3.load_state_dict(t3_state)
print("Patch applied successfully.")


text = "<chuckle><chuckle><chuckle><chuckle>Tief im verwunschenen Wald, wo die Bäume uralte Geheimnisse flüsterten, lebte ein kleiner Gnom namens Fips, der die Sprache der Tiere verstand."

reference_audio_path = "0.wav"
output_path = "output_cloned_voice.wav"

print("Generating speech...")
with torch.inference_mode():
    wav = model.generate(
        text,
        audio_prompt_path=reference_audio_path,
        exaggeration=0.5, 
        temperature=0.6,  
        cfg_weight=0.3,  
    )

sf.write(output_path, wav.squeeze().cpu().numpy(), model.sr)
print(f"Audio saved to {output_path}")

IPython.display.Audio(output_path)
