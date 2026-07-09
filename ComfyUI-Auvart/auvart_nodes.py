"""
AUVART COMFYUI CUSTOM NODES
============================
Hidden image art nodes for ComfyUI.

Installation:
  Copy the ComfyUI-Auvart folder to ComfyUI/custom_nodes/
  Restart ComfyUI

Nodes:
  - Auvart Photo Preprocessor: Optimal ControlNet input preparation
  - Auvart Hybrid Image: MIT hybrid image (painting + hidden photo)
  - Auvart FFT Phase Transfer: PTDiffusion-inspired phase manipulation
  - Auvart Photo To Mask: Convert photo to DifferentialDiffusion mask
  - Auvart Squint Simulator: Preview hidden image at different blur levels
"""

import torch
import numpy as np


class AuvartPhotoPreprocessor:
    """
    Preprocesses a photo for optimal ControlNet hidden image input.

    Converts to high-contrast format that QR Monster / Mysee respond best to.
    Gray background (#808080) is QR Monster v2's key feature for better blending.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["grayscale", "binary", "posterize_3level"],),
                "contrast": ("FLOAT", {
                    "default": 1.5, "min": 0.5, "max": 3.0, "step": 0.1,
                    "display": "slider"
                }),
                "blur_radius": ("INT", {
                    "default": 0, "min": 0, "max": 20, "step": 1
                }),
                "invert": ("BOOLEAN", {"default": False}),
                "gray_background": ("BOOLEAN", {"default": True}),
                "gray_threshold": ("INT", {
                    "default": 200, "min": 100, "max": 255, "step": 5
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_image",)
    FUNCTION = "process"
    CATEGORY = "Auvart/Preprocessing"

    def process(self, image, mode, contrast, blur_radius, invert,
                gray_background, gray_threshold):
        # image is [B, H, W, C] float32 [0,1]
        batch = []
        for i in range(image.shape[0]):
            img = image[i].cpu().numpy()  # [H, W, C]

            # Convert to grayscale (luminance weights)
            gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

            # Contrast enhancement
            if contrast != 1.0:
                mean = gray.mean()
                gray = (gray - mean) * contrast + mean
                gray = np.clip(gray, 0, 1)

            # Mode
            if mode == "binary":
                gray = np.where(gray > 0.5, 1.0, 0.0)
            elif mode == "posterize_3level":
                # 3 levels: black (<0.33), gray (0.33-0.67), white (>0.67)
                gray = np.where(gray < 0.33, 0.0,
                                np.where(gray < 0.67, 0.5, 1.0))

            # Invert
            if invert:
                gray = 1.0 - gray

            # Blur (simple box blur approximation)
            if blur_radius > 0:
                from scipy.ndimage import gaussian_filter
                gray = gaussian_filter(gray, sigma=blur_radius)

            # Convert to RGB
            result = np.stack([gray, gray, gray], axis=-1)

            # Gray background: replace bright areas with #808080 (0.502)
            if gray_background:
                threshold = gray_threshold / 255.0
                bright_mask = gray > threshold
                result[bright_mask] = 0.502

            batch.append(result)

        output = np.stack(batch, axis=0)
        return (torch.from_numpy(output).float(),)


class AuvartHybridImage:
    """
    Creates MIT hybrid image: painting visible up close,
    photo revealed when squinting.

    Based on Aude Oliva's research at MIT (2006).
    The painting provides high-frequency detail (visible close up).
    The photo provides low-frequency structure (visible from far / squinting).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "painting": ("IMAGE",),
                "photo": ("IMAGE",),
                "low_cutoff": ("FLOAT", {
                    "default": 12.0, "min": 1.0, "max": 50.0, "step": 1.0,
                    "display": "slider"
                }),
                "high_cutoff": ("FLOAT", {
                    "default": 12.0, "min": 1.0, "max": 50.0, "step": 1.0,
                    "display": "slider"
                }),
                "blend": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "display": "slider"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("hybrid_image",)
    FUNCTION = "create_hybrid"
    CATEGORY = "Auvart/Effects"

    def create_hybrid(self, painting, photo, low_cutoff, high_cutoff, blend):
        from scipy.ndimage import gaussian_filter

        batch = []
        for i in range(painting.shape[0]):
            p_idx = min(i, photo.shape[0] - 1)
            paint = painting[i].cpu().numpy()  # [H, W, C]
            ph = photo[p_idx].cpu().numpy()

            # Resize photo to match painting if needed
            if ph.shape[:2] != paint.shape[:2]:
                from PIL import Image
                ph_pil = Image.fromarray((ph * 255).astype(np.uint8))
                ph_pil = ph_pil.resize((paint.shape[1], paint.shape[0]),
                                       Image.LANCZOS)
                ph = np.array(ph_pil).astype(np.float64) / 255.0

            paint = paint.astype(np.float64)
            ph = ph.astype(np.float64)

            result = np.zeros_like(paint)
            for c in range(3):
                # Low-pass the photo
                photo_low = gaussian_filter(ph[:, :, c], sigma=low_cutoff)

                # High-pass the painting
                paint_blur = gaussian_filter(paint[:, :, c], sigma=high_cutoff)
                paint_high = paint[:, :, c] - paint_blur + 0.5

                # Combine
                result[:, :, c] = paint_high * (1.0 - blend) + photo_low * blend

            result = np.clip(result, 0, 1)
            batch.append(result)

        output = np.stack(batch, axis=0)
        return (torch.from_numpy(output).float(),)


class AuvartFFTPhaseTransfer:
    """
    Transfers phase spectrum from source to target image.
    Core concept from PTDiffusion (CVPR 2025).

    Phase = structure (edges, shapes, positions)
    Magnitude = style (textures, colors, contrast)

    Result: looks like the target (painting) but has the hidden
    structure of the source (photo) embedded.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE",),
                "target": ("IMAGE",),
                "transfer_ratio": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05,
                    "display": "slider"
                }),
                "channel_mode": (["per_channel", "luminance_only"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("transferred_image",)
    FUNCTION = "transfer"
    CATEGORY = "Auvart/Effects"

    def transfer(self, source, target, transfer_ratio, channel_mode):
        batch = []
        for i in range(target.shape[0]):
            s_idx = min(i, source.shape[0] - 1)
            src = source[s_idx].cpu().numpy().astype(np.float64)
            tgt = target[i].cpu().numpy().astype(np.float64)

            # Resize source to match target
            if src.shape[:2] != tgt.shape[:2]:
                from PIL import Image
                src_pil = Image.fromarray((src * 255).astype(np.uint8))
                src_pil = src_pil.resize((tgt.shape[1], tgt.shape[0]),
                                          Image.LANCZOS)
                src = np.array(src_pil).astype(np.float64) / 255.0

            if channel_mode == "luminance_only":
                result = self._transfer_luminance(src, tgt, transfer_ratio)
            else:
                result = self._transfer_per_channel(src, tgt, transfer_ratio)

            result = np.clip(result, 0, 1)
            batch.append(result)

        output = np.stack(batch, axis=0)
        return (torch.from_numpy(output).float(),)

    def _transfer_per_channel(self, src, tgt, ratio):
        result = np.zeros_like(tgt)
        for c in range(3):
            src_fft = np.fft.fft2(src[:, :, c])
            tgt_fft = np.fft.fft2(tgt[:, :, c])

            src_phase = np.angle(src_fft)
            tgt_mag = np.abs(tgt_fft)
            tgt_phase = np.angle(tgt_fft)

            blended_phase = tgt_phase * (1.0 - ratio) + src_phase * ratio
            combined = tgt_mag * np.exp(1j * blended_phase)
            channel = np.fft.ifft2(combined).real

            ch_min, ch_max = tgt[:, :, c].min(), tgt[:, :, c].max()
            if ch_max > ch_min:
                channel = (channel - channel.min()) / (channel.max() - channel.min())
                channel = channel * (ch_max - ch_min) + ch_min
            result[:, :, c] = channel
        return result

    def _transfer_luminance(self, src, tgt, ratio):
        src_y = 0.299 * src[:, :, 0] + 0.587 * src[:, :, 1] + 0.114 * src[:, :, 2]
        tgt_y = 0.299 * tgt[:, :, 0] + 0.587 * tgt[:, :, 1] + 0.114 * tgt[:, :, 2]

        src_fft = np.fft.fft2(src_y)
        tgt_fft = np.fft.fft2(tgt_y)

        blended_phase = (np.angle(tgt_fft) * (1.0 - ratio) +
                         np.angle(src_fft) * ratio)
        combined = np.abs(tgt_fft) * np.exp(1j * blended_phase)
        new_y = np.fft.ifft2(combined).real

        y_min, y_max = tgt_y.min(), tgt_y.max()
        if y_max > y_min:
            new_y = (new_y - new_y.min()) / (new_y.max() - new_y.min())
            new_y = new_y * (y_max - y_min) + y_min

        scale = np.where(tgt_y > 1e-6, new_y / tgt_y, 1.0)
        return tgt * scale[:, :, np.newaxis]


class AuvartPhotoToMask:
    """
    Converts a photo to a mask for DifferentialDiffusion.

    The mask controls per-pixel denoise strength:
    - White (1.0) = full denoise (generate new content)
    - Black (0.0) = no denoise (keep original)

    By using the hidden photo as a mask, different areas get
    different denoise levels, subtly embedding the photo structure.

    EXPERIMENTAL: Nobody has tried this for hidden images yet.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_radius": ("INT", {
                    "default": 5, "min": 0, "max": 30, "step": 1
                }),
                "contrast": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 3.0, "step": 0.1,
                    "display": "slider"
                }),
                "strength": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "display": "slider",
                }),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("denoise_mask",)
    FUNCTION = "convert"
    CATEGORY = "Auvart/Preprocessing"

    def convert(self, image, blur_radius, contrast, strength, invert):
        batch = []
        for i in range(image.shape[0]):
            img = image[i].cpu().numpy()

            # Convert to grayscale
            gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

            # Contrast
            if contrast != 1.0:
                mean = gray.mean()
                gray = (gray - mean) * contrast + mean
                gray = np.clip(gray, 0, 1)

            # Blur
            if blur_radius > 0:
                from scipy.ndimage import gaussian_filter
                gray = gaussian_filter(gray, sigma=blur_radius)

            # Invert
            if invert:
                gray = 1.0 - gray

            # Apply strength: compress range toward 0.5 (middle denoise)
            # strength=0: all pixels = 0.5 (uniform denoise, no hidden image)
            # strength=1: full range 0-1 (maximum hidden image effect)
            gray = 0.5 + (gray - 0.5) * strength

            batch.append(gray)

        output = np.stack(batch, axis=0)
        return (torch.from_numpy(output).float(),)


class AuvartSquintSimulator:
    """
    Simulates squinting / viewing from far away.
    Applies Gaussian blur to reveal the hidden image as it
    would appear in real life.

    Use this as a preview to check if the hidden image works
    before printing or sharing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "squint_level": ("FLOAT", {
                    "default": 8.0, "min": 1.0, "max": 30.0, "step": 0.5,
                    "display": "slider"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("squinted_image",)
    FUNCTION = "simulate"
    CATEGORY = "Auvart/Preview"

    def simulate(self, image, squint_level):
        from scipy.ndimage import gaussian_filter

        batch = []
        for i in range(image.shape[0]):
            img = image[i].cpu().numpy()
            result = np.zeros_like(img)
            for c in range(3):
                result[:, :, c] = gaussian_filter(img[:, :, c],
                                                   sigma=squint_level)
            batch.append(result)

        output = np.stack(batch, axis=0)
        return (torch.from_numpy(output).float(),)


# Node registration
NODE_CLASS_MAPPINGS = {
    "AuvartPhotoPreprocessor": AuvartPhotoPreprocessor,
    "AuvartHybridImage": AuvartHybridImage,
    "AuvartFFTPhaseTransfer": AuvartFFTPhaseTransfer,
    "AuvartPhotoToMask": AuvartPhotoToMask,
    "AuvartSquintSimulator": AuvartSquintSimulator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AuvartPhotoPreprocessor": "Auvart Photo Preprocessor",
    "AuvartHybridImage": "Auvart Hybrid Image",
    "AuvartFFTPhaseTransfer": "Auvart FFT Phase Transfer",
    "AuvartPhotoToMask": "Auvart Photo To Mask",
    "AuvartSquintSimulator": "Auvart Squint Simulator",
}