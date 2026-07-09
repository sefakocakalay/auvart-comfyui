"""
AUVART TOOLKIT - Hidden Image Art Tools
========================================
Standalone Python tools for hidden image / illusion art.

Usage:
  python auvart_toolkit.py hybrid --painting painting.png --photo photo.jpg --output hybrid.png
  python auvart_toolkit.py preprocess --input photo.jpg --output processed.png --mode posterize3
  python auvart_toolkit.py phase --source photo.jpg --target painting.png --output result.png
  python auvart_toolkit.py squint --input result.png --output squint_preview.png --level 8
  python auvart_toolkit.py analyze --input result.png --output analysis.png

Requirements: pip install numpy Pillow scipy matplotlib
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance, ImageOps


# ============================================================
# 1. MIT HYBRID IMAGE GENERATOR
# ============================================================

def create_hybrid_image(painting_path, photo_path, output_path,
                        low_cutoff=12.0, high_cutoff=12.0, blend=0.5,
                        photo_brightness=1.0):
    """
    Creates a hybrid image where the painting is visible up close
    and the photo is revealed when squinting / from far away.

    Based on MIT Aude Oliva's hybrid image research (2006).

    How it works:
    - High-pass filter the painting (keeps fine details, removes broad shapes)
    - Low-pass filter the photo (keeps broad shapes, removes fine details)
    - Combine them

    Close up: eyes see high-frequency detail = painting
    Far away / squint: eyes see low-frequency shapes = photo

    Args:
        painting_path: Path to the AI-generated painting
        photo_path: Path to the photo to hide
        output_path: Where to save the result
        low_cutoff: Gaussian blur radius for low-pass filter (higher = more blur = more hidden)
        high_cutoff: Gaussian blur radius for high-pass filter (higher = less painting detail)
        blend: How much of the photo to blend in (0.0-1.0)
        photo_brightness: Brightness adjustment for the hidden photo (1.0 = normal)
    """
    painting = Image.open(painting_path).convert('RGB')
    photo = Image.open(photo_path).convert('RGB')

    # Resize photo to match painting
    photo = photo.resize(painting.size, Image.LANCZOS)

    # Adjust photo brightness if needed
    if photo_brightness != 1.0:
        enhancer = ImageEnhance.Brightness(photo)
        photo = enhancer.enhance(photo_brightness)

    painting_arr = np.array(painting, dtype=np.float64)
    photo_arr = np.array(photo, dtype=np.float64)

    # LOW-PASS: Blur the photo heavily (keeps only broad shapes)
    photo_low = np.array(
        photo.filter(ImageFilter.GaussianBlur(low_cutoff)),
        dtype=np.float64
    )

    # HIGH-PASS: Painting minus its blurred version (keeps only details)
    painting_blurred = np.array(
        painting.filter(ImageFilter.GaussianBlur(high_cutoff)),
        dtype=np.float64
    )
    painting_high = painting_arr - painting_blurred + 128.0  # shift to mid-gray

    # COMBINE: high-freq painting + low-freq photo
    result = painting_high * (1.0 - blend) + photo_low * blend
    result = np.clip(result, 0, 255).astype(np.uint8)

    output = Image.fromarray(result)
    output.save(output_path, quality=95)
    print(f"Hybrid image saved: {output_path}")
    print(f"  Painting (high-pass cutoff): {high_cutoff}")
    print(f"  Photo (low-pass cutoff): {low_cutoff}")
    print(f"  Blend ratio: {blend}")
    print(f"  Resolution: {output.size[0]}x{output.size[1]}")
    return output


# ============================================================
# 2. FFT PHASE TRANSFER (PTDiffusion Core Concept)
# ============================================================

def fft_phase_transfer(source_path, target_path, output_path,
                       transfer_ratio=0.4, channel_mode='per_channel'):
    """
    Transfers phase spectrum from source (photo) to target (painting).
    This is the core mathematical concept from PTDiffusion (CVPR 2025).

    Phase carries STRUCTURAL information (edges, shapes, positions).
    Magnitude carries STYLE information (textures, colors, contrast).

    By replacing the painting's phase with the photo's phase,
    the result looks like the painting but has the hidden structure
    of the photo embedded.

    Args:
        source_path: Photo to hide (provides phase/structure)
        target_path: Painting (provides magnitude/style)
        output_path: Where to save
        transfer_ratio: How much phase to transfer (0.0=all painting, 1.0=all photo)
        channel_mode: 'per_channel' (RGB separately) or 'luminance_only'
    """
    source = Image.open(source_path).convert('RGB')
    target = Image.open(target_path).convert('RGB')

    # Resize source to match target
    source = source.resize(target.size, Image.LANCZOS)

    source_arr = np.array(source, dtype=np.float64)
    target_arr = np.array(target, dtype=np.float64)

    if channel_mode == 'luminance_only':
        result = _phase_transfer_luminance(source_arr, target_arr, transfer_ratio)
    else:
        result = _phase_transfer_per_channel(source_arr, target_arr, transfer_ratio)

    result = np.clip(result, 0, 255).astype(np.uint8)
    output = Image.fromarray(result)
    output.save(output_path, quality=95)
    print(f"Phase transfer saved: {output_path}")
    print(f"  Source (phase): {source_path}")
    print(f"  Target (magnitude): {target_path}")
    print(f"  Transfer ratio: {transfer_ratio}")
    print(f"  Mode: {channel_mode}")
    return output


def _phase_transfer_per_channel(source, target, ratio):
    """Apply FFT phase transfer on each RGB channel independently."""
    result = np.zeros_like(target)
    for c in range(3):
        src_fft = np.fft.fft2(source[:, :, c])
        tgt_fft = np.fft.fft2(target[:, :, c])

        src_phase = np.angle(src_fft)
        tgt_mag = np.abs(tgt_fft)
        tgt_phase = np.angle(tgt_fft)

        # Blend phases
        blended_phase = tgt_phase * (1.0 - ratio) + src_phase * ratio

        # Reconstruct: target magnitude + blended phase
        combined = tgt_mag * np.exp(1j * blended_phase)
        channel = np.fft.ifft2(combined).real

        # Normalize to original range
        ch_min, ch_max = target[:, :, c].min(), target[:, :, c].max()
        if ch_max > ch_min:
            channel = (channel - channel.min()) / (channel.max() - channel.min())
            channel = channel * (ch_max - ch_min) + ch_min

        result[:, :, c] = channel
    return result


def _phase_transfer_luminance(source, target, ratio):
    """Apply FFT phase transfer on luminance only, preserve chrominance."""
    # Convert to YCbCr-like decomposition
    # Y = 0.299*R + 0.587*G + 0.114*B
    src_y = 0.299 * source[:, :, 0] + 0.587 * source[:, :, 1] + 0.114 * source[:, :, 2]
    tgt_y = 0.299 * target[:, :, 0] + 0.587 * target[:, :, 1] + 0.114 * target[:, :, 2]

    # Phase transfer on luminance
    src_fft = np.fft.fft2(src_y)
    tgt_fft = np.fft.fft2(tgt_y)

    src_phase = np.angle(src_fft)
    tgt_mag = np.abs(tgt_fft)
    tgt_phase = np.angle(tgt_fft)

    blended_phase = tgt_phase * (1.0 - ratio) + src_phase * ratio
    combined = tgt_mag * np.exp(1j * blended_phase)
    new_y = np.fft.ifft2(combined).real

    # Normalize
    y_min, y_max = tgt_y.min(), tgt_y.max()
    if y_max > y_min:
        new_y = (new_y - new_y.min()) / (new_y.max() - new_y.min())
        new_y = new_y * (y_max - y_min) + y_min

    # Apply luminance change to RGB (preserve color ratios)
    scale = np.where(tgt_y > 1e-6, new_y / tgt_y, 1.0)
    result = target * scale[:, :, np.newaxis]
    return result


# ============================================================
# 3. PHOTO PREPROCESSOR FOR CONTROLNET
# ============================================================

def preprocess_photo(input_path, output_path, mode='grayscale',
                     contrast=1.5, blur_radius=0, invert=False,
                     gray_background=True, gray_threshold=200):
    """
    Preprocesses a photo for optimal ControlNet hidden image input.

    The key insight: QR Monster and similar ControlNets work best with
    high-contrast patterns. This preprocessor creates the optimal input.

    Modes:
    - grayscale: Simple high-contrast grayscale
    - binary: Pure black and white (threshold at 128)
    - posterize3: Three levels (black/gray/white) - matches Mysee training

    Args:
        input_path: Original photo
        output_path: Processed output
        mode: 'grayscale', 'binary', 'posterize3'
        contrast: Contrast enhancement factor (1.0 = no change, 2.0 = double)
        blur_radius: Gaussian blur radius (0 = none)
        invert: Whether to invert (white-on-black instead of black-on-white)
        gray_background: Replace white areas with #808080 (QR Monster v2 feature)
        gray_threshold: Brightness threshold for gray background replacement
    """
    img = Image.open(input_path).convert('RGB')

    # Step 1: Convert to grayscale
    img = ImageOps.grayscale(img)

    # Step 2: Enhance contrast
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)

    # Step 3: Apply mode
    if mode == 'binary':
        img = img.point(lambda x: 255 if x > 128 else 0)
    elif mode == 'posterize3':
        # 3-level: black (<85), gray (85-170), white (>170)
        # This matches ControlNet Mysee's training data
        img = img.point(lambda x: 0 if x < 85 else (128 if x < 170 else 255))

    # Step 4: Invert if requested
    if invert:
        img = ImageOps.invert(img)

    # Step 5: Apply blur
    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))

    # Step 6: Convert back to RGB
    img = img.convert('RGB')

    # Step 7: Gray background (#808080)
    if gray_background:
        arr = np.array(img)
        # Find pixels brighter than threshold (white/light areas)
        brightness = arr.mean(axis=2)
        light_mask = brightness > gray_threshold
        arr[light_mask] = [128, 128, 128]
        img = Image.fromarray(arr)

    img.save(output_path, quality=95)
    print(f"Preprocessed photo saved: {output_path}")
    print(f"  Mode: {mode}")
    print(f"  Contrast: {contrast}")
    print(f"  Blur: {blur_radius}")
    print(f"  Invert: {invert}")
    print(f"  Gray background: {gray_background} (threshold: {gray_threshold})")
    return img


# ============================================================
# 4. SQUINT SIMULATOR (Multi-Scale Blur Preview)
# ============================================================

def simulate_squint(input_path, output_path, level=8.0):
    """
    Simulates squinting / viewing from far away by applying Gaussian blur.
    This reveals the hidden image as it would appear in real life.

    Args:
        input_path: Generated image to test
        output_path: Squint preview output
        level: Blur intensity (higher = more squint / farther away)
    """
    img = Image.open(input_path).convert('RGB')
    squinted = img.filter(ImageFilter.GaussianBlur(level))
    squinted.save(output_path, quality=95)
    print(f"Squint simulation saved: {output_path}")
    print(f"  Level: {level}")
    return squinted


def simulate_squint_series(input_path, output_dir, levels=None):
    """
    Creates a series of squint simulations at different levels.
    Useful for finding the optimal viewing distance.
    """
    if levels is None:
        levels = [2, 4, 6, 8, 12, 16, 20, 30]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(input_path).convert('RGB')
    results = []

    for level in levels:
        squinted = img.filter(ImageFilter.GaussianBlur(level))
        out_path = output_dir / f"squint_level_{level:02d}.png"
        squinted.save(out_path, quality=95)
        results.append((level, out_path))
        print(f"  Level {level}: {out_path}")

    print(f"\nSquint series saved to: {output_dir}")
    print(f"  {len(levels)} images at levels: {levels}")
    return results


# ============================================================
# 5. FREQUENCY ANALYSIS (Visualize Hidden Image Quality)
# ============================================================

def analyze_frequencies(input_path, output_path):
    """
    Analyzes the frequency content of an image to visualize
    how well the hidden image is embedded.

    Creates a 2x2 grid:
    - Top-left: Original image
    - Top-right: FFT magnitude spectrum (log scale)
    - Bottom-left: Low-frequency content (hidden image should appear here)
    - Bottom-right: High-frequency content (painting details)

    Requires matplotlib: pip install matplotlib
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib required. Install: pip install matplotlib")
        return None

    img = Image.open(input_path).convert('L')
    arr = np.array(img, dtype=np.float64)

    # FFT
    fft = np.fft.fft2(arr)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shift))

    # Low-pass filter (reveals hidden image)
    rows, cols = arr.shape
    crow, ccol = rows // 2, cols // 2
    mask_low = np.zeros((rows, cols), dtype=np.float64)
    r = min(rows, cols) // 8  # cutoff radius
    y, x = np.ogrid[-crow:rows - crow, -ccol:cols - ccol]
    mask_area = x * x + y * y <= r * r
    mask_low[mask_area] = 1.0

    fft_low = fft_shift * mask_low
    low_freq = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_low)))

    # High-pass filter (painting details)
    mask_high = 1.0 - mask_low
    fft_high = fft_shift * mask_high
    high_freq = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_high)))

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    axes[0, 0].imshow(arr, cmap='gray')
    axes[0, 0].set_title('Original', fontsize=14)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(magnitude, cmap='hot')
    axes[0, 1].set_title('FFT Magnitude Spectrum', fontsize=14)
    axes[0, 1].axis('off')

    axes[1, 0].imshow(low_freq, cmap='gray')
    axes[1, 0].set_title('Low Frequency (Hidden Image)', fontsize=14)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(high_freq, cmap='gray')
    axes[1, 1].set_title('High Frequency (Painting Detail)', fontsize=14)
    axes[1, 1].axis('off')

    plt.suptitle('Auvart Frequency Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Frequency analysis saved: {output_path}")
    print(f"  Low-freq energy: {low_freq.sum():.0f}")
    print(f"  High-freq energy: {high_freq.sum():.0f}")
    print(f"  Ratio (low/high): {low_freq.sum() / max(high_freq.sum(), 1):.3f}")
    print(f"  Higher ratio = hidden image more dominant")


# ============================================================
# 6. BATCH VARIANT GENERATOR
# ============================================================

def create_variants(painting_path, photo_path, output_dir,
                    blend_values=None, phase_values=None):
    """
    Creates multiple variants with different settings for A/B comparison.
    Generates both hybrid images and phase transfer variants.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if blend_values is None:
        blend_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    if phase_values is None:
        phase_values = [0.2, 0.3, 0.4, 0.5, 0.6]

    print("=== HYBRID IMAGE VARIANTS ===")
    for blend in blend_values:
        out = output_dir / f"hybrid_blend_{blend:.1f}.png"
        create_hybrid_image(painting_path, photo_path, str(out), blend=blend)

    print("\n=== FFT PHASE TRANSFER VARIANTS ===")
    for ratio in phase_values:
        out = output_dir / f"phase_ratio_{ratio:.1f}.png"
        fft_phase_transfer(photo_path, painting_path, str(out),
                           transfer_ratio=ratio)

    print(f"\nAll variants saved to: {output_dir}")
    print(f"  {len(blend_values)} hybrid + {len(phase_values)} phase = "
          f"{len(blend_values) + len(phase_values)} total")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Auvart Toolkit - Hidden Image Art Tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python auvart_toolkit.py hybrid --painting art.png --photo face.jpg -o hybrid.png
  python auvart_toolkit.py preprocess --input face.jpg -o processed.png --mode posterize3 --gray-bg
  python auvart_toolkit.py phase --source face.jpg --target art.png -o result.png --ratio 0.4
  python auvart_toolkit.py squint --input result.png -o squint.png --level 8
  python auvart_toolkit.py squint-series --input result.png --output-dir ./squints/
  python auvart_toolkit.py analyze --input result.png -o analysis.png
  python auvart_toolkit.py variants --painting art.png --photo face.jpg --output-dir ./variants/
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Hybrid
    p_hybrid = subparsers.add_parser('hybrid', help='Create MIT hybrid image')
    p_hybrid.add_argument('--painting', required=True, help='Path to painting')
    p_hybrid.add_argument('--photo', required=True, help='Path to photo to hide')
    p_hybrid.add_argument('-o', '--output', required=True, help='Output path')
    p_hybrid.add_argument('--low-cutoff', type=float, default=12.0)
    p_hybrid.add_argument('--high-cutoff', type=float, default=12.0)
    p_hybrid.add_argument('--blend', type=float, default=0.5)
    p_hybrid.add_argument('--brightness', type=float, default=1.0)

    # Preprocess
    p_prep = subparsers.add_parser('preprocess', help='Preprocess photo for ControlNet')
    p_prep.add_argument('--input', required=True, help='Input photo')
    p_prep.add_argument('-o', '--output', required=True, help='Output path')
    p_prep.add_argument('--mode', choices=['grayscale', 'binary', 'posterize3'],
                        default='grayscale')
    p_prep.add_argument('--contrast', type=float, default=1.5)
    p_prep.add_argument('--blur', type=int, default=0)
    p_prep.add_argument('--invert', action='store_true')
    p_prep.add_argument('--gray-bg', action='store_true', default=True)
    p_prep.add_argument('--no-gray-bg', action='store_true')
    p_prep.add_argument('--gray-threshold', type=int, default=200)

    # Phase Transfer
    p_phase = subparsers.add_parser('phase', help='FFT phase transfer')
    p_phase.add_argument('--source', required=True, help='Photo (phase source)')
    p_phase.add_argument('--target', required=True, help='Painting (magnitude source)')
    p_phase.add_argument('-o', '--output', required=True, help='Output path')
    p_phase.add_argument('--ratio', type=float, default=0.4)
    p_phase.add_argument('--mode', choices=['per_channel', 'luminance_only'],
                         default='per_channel')

    # Squint
    p_squint = subparsers.add_parser('squint', help='Simulate squinting')
    p_squint.add_argument('--input', required=True)
    p_squint.add_argument('-o', '--output', required=True)
    p_squint.add_argument('--level', type=float, default=8.0)

    # Squint Series
    p_series = subparsers.add_parser('squint-series', help='Create squint level series')
    p_series.add_argument('--input', required=True)
    p_series.add_argument('--output-dir', required=True)

    # Analyze
    p_analyze = subparsers.add_parser('analyze', help='Frequency analysis')
    p_analyze.add_argument('--input', required=True)
    p_analyze.add_argument('-o', '--output', required=True)

    # Variants
    p_var = subparsers.add_parser('variants', help='Generate comparison variants')
    p_var.add_argument('--painting', required=True)
    p_var.add_argument('--photo', required=True)
    p_var.add_argument('--output-dir', required=True)

    args = parser.parse_args()

    if args.command == 'hybrid':
        create_hybrid_image(args.painting, args.photo, args.output,
                            args.low_cutoff, args.high_cutoff,
                            args.blend, args.brightness)
    elif args.command == 'preprocess':
        gray_bg = args.gray_bg and not args.no_gray_bg
        preprocess_photo(args.input, args.output, args.mode,
                         args.contrast, args.blur, args.invert,
                         gray_bg, args.gray_threshold)
    elif args.command == 'phase':
        fft_phase_transfer(args.source, args.target, args.output,
                           args.ratio, args.mode)
    elif args.command == 'squint':
        simulate_squint(args.input, args.output, args.level)
    elif args.command == 'squint_series' or args.command == 'squint-series':
        simulate_squint_series(args.input, args.output_dir)
    elif args.command == 'analyze':
        analyze_frequencies(args.input, args.output)
    elif args.command == 'variants':
        create_variants(args.painting, args.photo, args.output_dir)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()