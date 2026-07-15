"""Simulated dashcam-condition degradations for robustness stress-testing.

Each transform operates on a PIL image (before ToTensor/Normalize) and takes
a `severity` in [0, 1] controlling how strong the effect is; 0 is a no-op.
"""
import math
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


class GlareOverlay:
    """Simulate a specular glare/hotspot, e.g. sun reflection on a sign."""

    def __init__(self, severity: float = 0.5):
        self.severity = severity

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.severity <= 0:
            return img
        w, h = img.size
        cx = random.uniform(0.2, 0.8) * w
        cy = random.uniform(0.2, 0.8) * h
        radius = (0.25 + 0.35 * self.severity) * max(w, h)

        yy, xx = np.mgrid[0:h, 0:w]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        glare = np.clip(1.0 - dist / radius, 0, 1) ** 2 * self.severity

        arr = np.asarray(img).astype(np.float32) + glare[..., None] * 255.0
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


class MotionBlur:
    """Simulate camera/subject motion blur along a random direction."""

    def __init__(self, severity: float = 0.5):
        self.severity = severity

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.severity <= 0:
            return img
        size = max(3, int(round(self.severity * 15)) | 1)
        angle = random.uniform(0, math.pi)
        center = size // 2

        kernel = np.zeros((size, size), dtype=np.float32)
        for i in range(size):
            offset = i - center
            x = int(round(center + offset * math.cos(angle)))
            y = int(round(center + offset * math.sin(angle)))
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1.0
        kernel /= kernel.sum()

        pil_kernel = ImageFilter.Kernel((size, size), kernel.flatten().tolist(), scale=1.0)
        return img.filter(pil_kernel)


class LowLight:
    """Simulate underexposed dashcam footage: dimmer, lower contrast, noisier."""

    def __init__(self, severity: float = 0.5):
        self.severity = severity

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.severity <= 0:
            return img
        img = ImageEnhance.Brightness(img).enhance(1.0 - 0.7 * self.severity)
        img = ImageEnhance.Contrast(img).enhance(1.0 - 0.3 * self.severity)

        arr = np.asarray(img).astype(np.float32)
        noise = np.random.normal(0, 15.0 * self.severity, arr.shape)
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


PERTURBATIONS = {
    "glare": GlareOverlay,
    "motion_blur": MotionBlur,
    "low_light": LowLight,
}
