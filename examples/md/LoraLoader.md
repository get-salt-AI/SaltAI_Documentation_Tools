# Load LoRA
## Documentation
- Class name: `LoraLoader`
- Category: `loaders`
- Output node: `False`

The LoraLoader node is designed to dynamically load and apply LoRA (Low-Rank Adaptation) adjustments to models and CLIP instances. It enhances or modifies the behavior of these components by injecting learned parameters, allowing for fine-tuned control over their performance and outputs.
## Input types
### Required
- **`model`**
    - Specifies the model to which LoRA adjustments will be applied. It is central to the node's operation as it determines the base model that will be modified.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`clip`**
    - Specifies the CLIP instance to which LoRA adjustments will be applied. This allows for fine-tuning of CLIP's behavior in conjunction with the model.
    - Comfy dtype: `CLIP`
    - Python dtype: `torch.nn.Module`
- **`lora_name`**
    - The name of the LoRA file to be loaded. This file contains the LoRA adjustments to be applied to the model and/or CLIP instance.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`strength_model`**
    - Determines the intensity of the LoRA adjustments applied to the model. It allows for precise control over how much the model's behavior is altered.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`strength_clip`**
    - Determines the intensity of the LoRA adjustments applied to the CLIP instance. It allows for precise control over how much the CLIP's behavior is altered.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The modified model with LoRA adjustments applied.
    - Python dtype: `torch.nn.Module`
- **`clip`**
    - Comfy dtype: `CLIP`
    - The modified CLIP instance with LoRA adjustments applied.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - LoraLoader
    - CLIPTextEncode
    - Reroute
    - VideoLinearCFGGuidance
    - KSampler
    - FaceDetailer
    - ModelSamplingDiscrete
    - ADE_AnimateDiffLoaderWithContext
    - KSampler //Inspire
    - ToBasicPipe



## Source code
```python
class LoraLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP", ),
                              "lora_name": (folder_paths.get_filename_list("loras"), ),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                              "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"

    CATEGORY = "loaders"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)

```
