#!/usr/bin/env python3
"""Small desktop image-redaction utility built with Tkinter and Pillow.

Run without arguments to open the file picker, or pass an image path to load it:

    python tools/ImageRedactorUI.py photo.jpg
"""

from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass
from pathlib import Path
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps, ImageTk


@dataclass(frozen=True)
class RedactionSettings:
    noise: int = 50
    brightness: int = 35
    rotation: float = 0.0
    scale: int = 50
    pixel_size: int = 10
    blur: float = 0.0
    match_resistance: int = 0
    crop_horizontal: int = 0
    crop_vertical: int = 0


def apply_match_resistance(image: Image.Image, strength: int) -> Image.Image:
    """Apply a subtle, deterministic fingerprint-resistant transformation.

    Reverse-image services are deliberately robust, so this can only make an
    exact/near-exact match harder; it cannot guarantee that an image is never
    found. The transform combines tiny channel displacement with a structured
    high-frequency pattern, which survives ordinary PNG export.
    """
    strength = max(0, min(100, int(strength)))
    if strength == 0:
        return image

    shift = max(1, round(strength / 25))
    red, green, blue = image.split()
    red = ImageChops.offset(red, shift, 0)
    blue = ImageChops.offset(blue, -shift, 0)
    altered = Image.merge("RGB", (red, green, blue))

    # Add a low-opacity diagonal pattern. Unlike metadata, this changes the
    # rendered pixels and therefore remains present when a site re-encodes it.
    spacing = max(5, 18 - strength // 8)
    amplitude = max(2, round(strength / 12))
    pattern = Image.new("L", altered.size, 128)
    drawing = ImageDraw.Draw(pattern)
    for index, start in enumerate(range(-altered.height, altered.width, spacing)):
        value = 128 + amplitude if index & 1 else 128 - amplitude
        drawing.line((start, 0, start + altered.height, altered.height), fill=value, width=1)
    pattern_rgb = Image.merge("RGB", (pattern, pattern, pattern))
    return ImageChops.add(altered, pattern_rgb, scale=1.0, offset=-128)


def process_image(source: Image.Image, settings: RedactionSettings) -> Image.Image:
    """Return a processed copy without changing the source image."""
    image = ImageOps.exif_transpose(source).convert("RGBA")

    # Crop percentages are applied to each matching edge. Keeping them below
    # 50% guarantees that at least one row and column remain.
    crop_x = round(image.width * max(0, min(45, settings.crop_horizontal)) / 100)
    crop_y = round(image.height * max(0, min(45, settings.crop_vertical)) / 100)
    if crop_x or crop_y:
        image = image.crop(
            (crop_x, crop_y, image.width - crop_x, image.height - crop_y)
        )

    alpha = image.getchannel("A")
    rgb = image.convert("RGB")

    # Pixelation is a useful redaction layer and is much faster than changing
    # each pixel in Python. A value of 1 leaves the image untouched.
    if settings.pixel_size > 1:
        small_size = (
            max(1, rgb.width // settings.pixel_size),
            max(1, rgb.height // settings.pixel_size),
        )
        rgb = rgb.resize(small_size, Image.Resampling.BILINEAR).resize(
            rgb.size, Image.Resampling.NEAREST
        )

    if settings.noise > 0:
        noise = Image.effect_noise(rgb.size, settings.noise).convert("RGB")
        # effect_noise is centered near 128. The offset recenters it and then
        # applies the requested brightness while ImageChops safely clamps RGB.
        rgb = ImageChops.add(rgb, noise, scale=1.0, offset=settings.brightness - 128)
    elif settings.brightness:
        factor = max(0.0, 1.0 + settings.brightness / 100.0)
        rgb = ImageEnhance.Brightness(rgb).enhance(factor)

    if settings.blur > 0:
        rgb = rgb.filter(ImageFilter.GaussianBlur(settings.blur))

    if settings.match_resistance > 0:
        rgb = apply_match_resistance(rgb, settings.match_resistance)

    image = rgb.convert("RGBA")
    image.putalpha(alpha)
    if settings.rotation:
        image = image.rotate(
            settings.rotation,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=(0, 0, 0, 0),
        )

    scale = max(1, settings.scale) / 100.0
    target = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(target, Image.Resampling.LANCZOS)


def suggested_output_path(source_path: Path) -> Path:
    return source_path.with_name(f"{source_path.stem}_redacted.png")


def save_png(image: Image.Image, path: Path) -> None:
    """Save atomically so a failed write cannot leave a partial output file."""
    path = path.with_suffix(".png")
    temporary = path.with_name(f".{path.name}.tmp")
    image.save(temporary, format="PNG", optimize=True)
    temporary.replace(path)


class ImageRedactorApp:
    PREVIEW_SIZE = (820, 620)

    def __init__(self, root, initial_path: Path | None = None) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.root.title("Image Redactor")
        self.root.geometry("1120x760")
        self.root.minsize(900, 620)

        self.source_path: Path | None = None
        self.source_image: Image.Image | None = None
        self.result_image: Image.Image | None = None
        self.preview_photo = None
        self.render_generation = 0

        self.noise = tk.IntVar(value=50)
        self.brightness = tk.IntVar(value=35)
        self.rotation = tk.DoubleVar(value=0)
        self.scale = tk.IntVar(value=50)
        self.pixel_size = tk.IntVar(value=10)
        self.blur = tk.DoubleVar(value=0)
        self.match_resistance = tk.IntVar(value=0)
        self.crop_horizontal = tk.IntVar(value=0)
        self.crop_vertical = tk.IntVar(value=0)
        self.status = tk.StringVar(value="Open an image to begin")

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main = ttk.Frame(root, padding=12)
        main.grid(sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(main)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        ttk.Button(toolbar, text="Open image", command=self.open_image).pack(side="left")
        self.save_button = ttk.Button(
            toolbar, text="Save redacted copy", command=self.save_image, state="disabled"
        )
        self.save_button.pack(side="left", padx=8)
        ttk.Button(toolbar, text="Reset", command=self.reset_controls).pack(side="left")

        preview_frame = ttk.Frame(main, relief="sunken", padding=4)
        preview_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        self.preview = ttk.Label(preview_frame, text="No image selected", anchor="center")
        self.preview.grid(sticky="nsew")

        controls = ttk.LabelFrame(main, text="Effects", padding=12)
        controls.grid(row=1, column=1, sticky="ns")
        controls.columnconfigure(0, weight=1)

        rows = [
            ("Noise", self.noise, 0, 150, 1),
            ("Brightness", self.brightness, -100, 100, 1),
            ("Rotation", self.rotation, -180, 180, 1),
            ("Output size (%)", self.scale, 10, 200, 1),
            ("Pixel size", self.pixel_size, 1, 50, 1),
            ("Blur", self.blur, 0, 20, 0.5),
            ("Reverse-search resistance*", self.match_resistance, 0, 100, 1),
            ("Crop left/right (% each)", self.crop_horizontal, 0, 45, 1),
            ("Crop top/bottom (% each)", self.crop_vertical, 0, 45, 1),
        ]
        for row, (label, variable, start, end, step) in enumerate(rows):
            ttk.Label(controls, text=label).grid(row=row * 2, column=0, sticky="w")
            scale = ttk.Scale(controls, from_=start, to=end, variable=variable)
            scale.grid(row=row * 2 + 1, column=0, sticky="ew", pady=(0, 12))
            scale.configure(command=lambda _value: self.queue_render())
            ttk.Spinbox(
                controls,
                from_=start,
                to=end,
                increment=step,
                textvariable=variable,
                width=7,
                command=self.queue_render,
            ).grid(row=row * 2 + 1, column=1, padx=(8, 0))
            variable.trace_add("write", lambda *_args: self.queue_render())

        ttk.Label(
            controls,
            text="* Experimental: can reduce matching,\nbut no edit can guarantee prevention.",
            foreground="#666666",
            justify="left",
        ).grid(row=len(rows) * 2, column=0, columnspan=2, sticky="w", pady=(2, 0))

        ttk.Label(main, textvariable=self.status, anchor="w").grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0)
        )

        self.root.bind("<Control-o>", lambda _event: self.open_image())
        self.root.bind("<Control-s>", lambda _event: self.save_image())
        if initial_path:
            self.load_image(initial_path)

    def settings(self) -> RedactionSettings:
        return RedactionSettings(
            noise=int(self.noise.get()),
            brightness=int(self.brightness.get()),
            rotation=float(self.rotation.get()),
            scale=int(self.scale.get()),
            pixel_size=int(self.pixel_size.get()),
            blur=float(self.blur.get()),
            match_resistance=int(self.match_resistance.get()),
            crop_horizontal=int(self.crop_horizontal.get()),
            crop_vertical=int(self.crop_vertical.get()),
        )

    def reset_controls(self) -> None:
        defaults = RedactionSettings()
        self.noise.set(defaults.noise)
        self.brightness.set(defaults.brightness)
        self.rotation.set(defaults.rotation)
        self.scale.set(defaults.scale)
        self.pixel_size.set(defaults.pixel_size)
        self.blur.set(defaults.blur)
        self.match_resistance.set(defaults.match_resistance)
        self.crop_horizontal.set(defaults.crop_horizontal)
        self.crop_vertical.set(defaults.crop_vertical)
        self.queue_render()

    def open_image(self) -> None:
        from tkinter import filedialog

        selected = filedialog.askopenfilename(
            title="Open image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self.load_image(Path(selected))

    def load_image(self, path: Path) -> None:
        from tkinter import messagebox

        try:
            with Image.open(path) as opened:
                opened.load()
                self.source_image = opened.copy()
            self.source_path = path
            self.status.set(f"Loaded {path.name} — {self.source_image.width}×{self.source_image.height}")
            self.queue_render()
        except (OSError, ValueError) as error:
            messagebox.showerror("Could not open image", str(error))

    def queue_render(self) -> None:
        if self.source_image is None:
            return
        try:
            settings = self.settings()
        except (ValueError, TypeError):
            return
        self.render_generation += 1
        generation = self.render_generation
        source = self.source_image.copy()
        self.status.set("Updating preview…")

        def worker() -> None:
            try:
                result = process_image(source, settings)
                self.root.after(0, lambda: self.finish_render(generation, result))
            except Exception as error:  # report unexpected Pillow errors in the UI
                self.root.after(
                    0, lambda captured=error: self.render_failed(generation, captured)
                )

        threading.Thread(target=worker, daemon=True).start()

    def finish_render(self, generation: int, result: Image.Image) -> None:
        if generation != self.render_generation:
            return
        self.result_image = result
        preview = result.copy()
        preview.thumbnail(self.PREVIEW_SIZE, Image.Resampling.LANCZOS)
        checker = Image.new("RGBA", preview.size, "#d8d8d8")
        display = Image.alpha_composite(checker, preview).convert("RGB")
        self.preview_photo = ImageTk.PhotoImage(display)
        self.preview.configure(image=self.preview_photo, text="")
        self.save_button.configure(state="normal")
        self.status.set(f"Ready — output {result.width}×{result.height}")

    def render_failed(self, generation: int, error: Exception) -> None:
        from tkinter import messagebox

        if generation == self.render_generation:
            self.status.set("Preview failed")
            messagebox.showerror("Could not process image", str(error))

    def save_image(self) -> None:
        from tkinter import filedialog, messagebox

        if self.result_image is None or self.source_path is None:
            return
        suggested = suggested_output_path(self.source_path)
        selected = filedialog.asksaveasfilename(
            title="Save redacted image",
            initialdir=str(suggested.parent),
            initialfile=suggested.name,
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
        )
        if not selected:
            return
        output = Path(selected).with_suffix(".png")
        try:
            save_png(self.result_image, output)
            self.status.set(f"Saved {output}")
            messagebox.showinfo("Saved", f"Redacted copy saved to:\n{output}")
        except OSError as error:
            messagebox.showerror("Could not save image", str(error))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Redact an image with a graphical preview.")
    parser.add_argument("image", nargs="?", type=Path, help="image to open")
    return parser.parse_args()


def main() -> None:
    import tkinter as tk

    args = parse_args()
    root = tk.Tk()
    ImageRedactorApp(root, args.image)
    root.mainloop()


if __name__ == "__main__":
    main()
