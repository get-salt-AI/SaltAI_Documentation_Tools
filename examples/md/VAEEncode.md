# VAE Encode
## Documentation
- Class name: `VAEEncode`
- Category: `latent`
- Output node: `False`

This node is designed for encoding images into a latent representation using a specified VAE model. It abstracts the complexity of the encoding process, providing a straightforward way to transform image data into a format suitable for various generative tasks.
## Input types
### Required
- **`pixels`**
    - Represents the image data to be encoded. Its role is crucial as it serves as the direct input for the encoding process, determining the characteristics of the resulting latent representation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`vae`**
    - Specifies the VAE model to be used for encoding. It defines the encoding mechanism and directly influences the quality and attributes of the generated latent representation.
    - Comfy dtype: `VAE`
    - Python dtype: `comfy.sd.VAE`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The encoded latent representation of the input image. It encapsulates the essential features extracted during the encoding process.
    - Python dtype: `Dict[str, torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - KSampler
    - KSamplerAdvanced
    - SetLatentNoiseMask
    - ImpactKSamplerBasicPipe
    - KSampler (Efficient)
    - BNK_Unsampler
    - LatentUpscale
    - KSampler //Inspire
    - DZ_Face_Detailer
    - LatentUpscaleBy



## Source code
```python
class VAEEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "pixels": ("IMAGE", ), "vae": ("VAE", )}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = "latent"

    def encode(self, vae, pixels):
        t = vae.encode(pixels[:,:,:,:3])
        return ({"samples":t}, )

```
