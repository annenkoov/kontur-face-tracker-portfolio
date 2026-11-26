"""Tkinter user interface for calibration, live tracking, and journal."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from datetime import datetime
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox, ttk
import time
import threading
from statistics import median

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageTk
from PIL import ImageDraw

from .events import EventTracker, TrackerState
from .evaluation import self_rating_summary, top_expressions
from .labels import LABELS, LabeledSampleStore, observation_is_fresh
from .personalization import default_adapter_path, load_adapter
from .patterns import FacialPatternTracker, PATTERN_LABELS
from .action_units import AU_LABELS_RU
from .scoring import CalibrationProfile, CalibrationSampler, Scorer, apply_personal_sharpness_gate
from .session import TrackingSession
from .storage import JournalStore
from .temporal import ExpressionSmoother
from .worker import AnalysisWorker, WorkerMessage
from .vision import (EMOTION_MODEL_VERSION, FACIAL_CONTOUR_CONNECTIONS,
                     consensus_expression_probabilities)


BG = "#080914"
PANEL = "#11172d"
PANEL_ALT = "#192340"
PANEL_HI = "#202c4d"
TEXT = "#f5f7ff"
MUTED = "#aeb8d4"
ACCENT = "#8b7cff"
GREEN = "#48e1b7"
AMBER = "#ffc36b"
RED = "#ff7188"
LIVE_MAX_AGE_SECONDS = 1.8
CARD_BORDER = "#53618f"
VIOLET = "#a18bff"
TEAL = "#48e5e1"
EXPRESSION_RU = {
    "Anger": "Злость", "Contempt": "Презрение", "Disgust": "Отвращение", "Fear": "Страх",
    "Happiness": "Радость", "Neutral": "Нейтральное", "Sadness": "Грусть", "Surprise": "Удивление",
}


@lru_cache(maxsize=8)
def _rounded_mask(width: int, height: int, radius: int) -> Image.Image:
    """Кэширует геометрию маски: видеокадры одного размера переиспользуют её."""
    scale = 2
    mask_size = (width * scale, height * scale)
    mask = Image.new("L", mask_size, 0)
    clamped_radius = max(0, min(int(radius), min(width, height) // 2)) * scale
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, mask_size[0] - 1, mask_size[1] - 1),
        radius=clamped_radius, fill=255)
    return mask.resize((width, height), Image.Resampling.LANCZOS)


def apply_rounded_mask(image: Image.Image, radius: int) -> Image.Image:
    """Возвращает копию изображения с сглаженными прозрачными углами."""
    rgba = image.convert("RGBA")
    width, height = rgba.size
    if width < 1 or height < 1:
        return rgba
    mask = _rounded_mask(width, height, int(radius))
    rgba.putalpha(ImageChops.multiply(rgba.getchannel("A"), mask))
    return rgba


def rounded_icon_tile(icon: ImageTk.PhotoImage, size: int, fill: str,
                      radius: int = 11) -> ImageTk.PhotoImage:
    """Создаёт PNG-подобную плашку: фон и пиктограмма скругляются вместе."""
    tile = Image.new("RGBA", (size, size), fill)
    icon_image = ImageTk.getimage(icon).convert("RGBA")
    icon_image.thumbnail((size // 2, size // 2), Image.Resampling.LANCZOS)
    tile.alpha_composite(icon_image, ((size - icon_image.width) // 2,
                                     (size - icon_image.height) // 2))
    return ImageTk.PhotoImage(apply_rounded_mask(tile, radius))


def _rounded_surface_image(width: int, height: int, *, fill: str, border: str,
                           radius: int, padding: int, accent: str | None) -> Image.Image:
    """Статичная матовая поверхность с бликом только в безопасной рамке."""
    width, height = max(8, int(width)), max(8, int(height))
    radius = min(max(0, int(radius)), width // 2, height // 2)
    scale = 2
    size = (width * scale, height * scale)
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (1, 1, size[0] - 2, size[1] - 2), radius=radius * scale, fill=255)
    base = np.array([int(fill[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32)
    pixels = np.broadcast_to(base, (size[1], size[0], 3)).copy()
    if accent and padding > 0:
        glow = np.array([int(accent[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32)
        x = np.arange(size[0], dtype=np.float32)[None, :]
        y = np.arange(size[1], dtype=np.float32)[:, None]
        width_scaled, height_scaled = size
        pad = max(1, min(int(padding) * scale, width_scaled // 2, height_scaled // 2))
        left = np.clip((pad - x) / pad, 0, 1)
        right = np.clip((x - (width_scaled - pad)) / pad, 0, 1)
        top = np.clip((pad - y) / pad, 0, 1)
        bottom = np.clip((y - (height_scaled - pad)) / pad, 0, 1)
        rim = np.maximum(np.maximum(left, right), np.maximum(top, bottom))
        amount = (rim * .16)[..., None]
        pixels = pixels * (1 - amount) + glow[None, None, :] * amount
    surface = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="RGB").convert("RGBA")
    surface.putalpha(mask)
    shadow_mask = mask.filter(ImageFilter.GaussianBlur(10 * scale))
    shadow = Image.new("RGBA", size, (100, 93, 255, 0))
    shadow.putalpha(shadow_mask.point(lambda value: round(value * .20)))
    composed = Image.alpha_composite(shadow, surface)
    draw = ImageDraw.Draw(composed)
    draw.rounded_rectangle((2, 2, size[0] - 3, size[1] - 3),
                            radius=radius * scale, outline=border, width=scale)
    draw.line((radius * 1.2, 3 * scale, size[0] - radius * 1.2, 3 * scale),
              fill=(190, 205, 255, 54), width=scale)
    return composed.resize((width, height), Image.Resampling.LANCZOS)


class RoundedCard(tk.Canvas):
    """A lightweight rounded Tk card, drawn without extra UI dependencies."""

    def __init__(self, master, *, fill=PANEL, border=CARD_BORDER, radius=16, padding=12,
                 accent=None, width=240, height=120, **kwargs):
        super().__init__(master, bg=BG, bd=0, highlightthickness=0, **kwargs)
        # Явный размер убирает стандартную большую высоту Canvas в Tk,
        # которая иначе раздувает панели и может вытеснить график.
        self.configure(width=width, height=height)
        self.fill = fill
        self.border = border
        self.radius = radius
        self.padding = padding
        self.accent = accent
        self.content = tk.Frame(self, bg=fill)
        # Размер встроенного content-контейнера задаёт Canvas-окно карточки;
        # не позволяем дочерним подписям пересогласовывать внешний размер.
        self.content.grid_propagate(False)
        self._surface_photo = None
        self._surface_size = None
        self._window = self.create_window(padding, padding, anchor="nw", window=self.content)
        self._redraw_pending = False
        self._redraw_job = None
        self.bind("<Configure>", self._schedule_redraw, add="+")
        self._schedule_redraw()

    def _schedule_redraw(self, _event=None):
        # При перетаскивании границы Windows посылает много Configure подряд.
        # Сбрасываем таймер и рисуем карточку один раз после короткой паузы.
        if self._redraw_job is not None:
            try:
                self.after_cancel(self._redraw_job)
            except tk.TclError:
                pass
        self._redraw_pending = True
        self._redraw_job = self.after(35, self._redraw_card)

    def _redraw_card(self, _event=None):
        self._redraw_pending = False
        self._redraw_job = None
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return
        width = max(2, self.winfo_width())
        height = max(2, self.winfo_height())
        # Tk может сначала прислать временный Configure размером 1×1 при
        # построении окна. Не рисуем Pillow-контур, пока геометрия не готова.
        if width < 8 or height < 8:
            return
        radius = min(self.radius, width // 2, height // 2)
        self._draw_surface(width, height, radius)

        inset = self.padding
        self.coords(self._window, inset, inset)
        self.itemconfigure(self._window, width=max(1, width - inset * 2),
                           height=max(1, height - inset * 2))

        self.tag_raise("card-border")

    def _draw_surface(self, width: int, height: int, radius: int) -> None:
        """Рисует матовую стеклянную панель: мягкий tint, светлый кант и ореол."""
        key = (width, height, radius, self.fill, self.border, self.accent)
        if key == self._surface_size:
            return
        self._surface_size = key
        self._surface_photo = ImageTk.PhotoImage(_rounded_surface_image(
            width, height, fill=self.fill, border=self.border, radius=radius,
            padding=self.padding, accent=self.accent))
        self.delete("card-surface")
        self.create_image(0, 0, image=self._surface_photo, anchor="nw", tags="card-surface")
        self.tag_lower("card-surface")


class RoundedPanel(tk.Frame):
    """Автовысотная скруглённая панель для содержимого с pack-геометрией."""

    def __init__(self, master, *, fill=PANEL, border=CARD_BORDER, radius=16,
                 padding=14, accent=VIOLET, **kwargs):
        super().__init__(master, bg=BG, bd=0, highlightthickness=0, **kwargs)
        self.fill = fill
        self.border = border
        self.radius = radius
        self.padding = padding
        self.accent = accent
        self._photo = None
        self._last_size = None
        self._redraw_job = None
        self._background = tk.Canvas(self, bg=BG, bd=0, highlightthickness=0)
        self._background.place(x=0, y=0, relwidth=1, relheight=1)
        self.content = tk.Frame(self, bg=fill, bd=0, highlightthickness=0)
        self.content.pack(fill="both", expand=True, padx=padding, pady=padding)
        self.bind("<Configure>", self._schedule_redraw, add="+")
        self._schedule_redraw()

    def _schedule_redraw(self, _event=None):
        if self._redraw_job is not None:
            try:
                self.after_cancel(self._redraw_job)
            except tk.TclError:
                pass
        self._redraw_job = self.after(35, self._redraw)

    def _redraw(self):
        self._redraw_job = None
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return
        width, height = self.winfo_width(), self.winfo_height()
        if width < 8 or height < 8 or (width, height) == self._last_size:
            return
        self._last_size = (width, height)
        photo = ImageTk.PhotoImage(_rounded_surface_image(
            width, height, fill=self.fill, border=self.border, radius=self.radius,
            padding=self.padding, accent=self.accent))
        self._photo = photo
        self._background.delete("rounded-panel-surface")
        self._background.create_image(0, 0, image=photo, anchor="nw",
                                      tags="rounded-panel-surface")


class StyledButton(tk.Frame):
    """Сглаженная кнопка-контейнер с отдельной семантической Tk-кнопкой."""

    def __init__(self, master, *, text, icon, command, variant="secondary", surface=PANEL,
                 text_color=TEXT, font=("Segoe UI Semibold", 10), initial_width=240,
                 initial_height=56, **kwargs):
        self._variant = variant
        self._surface = surface
        self._state = kwargs.pop("state", tk.NORMAL)
        self._hover = False
        self._focused = False
        self._selected = False
        self._photo = None
        self._icon_active = icon
        icon_source = ImageTk.getimage(icon).convert("RGBA")
        icon_muted = Image.new("RGBA", icon_source.size, "#8792a6")
        icon_muted.putalpha(icon_source.getchannel("A"))
        self._icon_disabled = ImageTk.PhotoImage(icon_muted)
        self._drawn_size = None
        self._resize_job = None
        self._text = text
        self._text_color = text_color
        self._command = command
        super().__init__(master, bg=surface, bd=0, highlightthickness=0,
                         width=initial_width, height=initial_height, takefocus=0,
                         cursor="hand2", **kwargs)
        self.pack_propagate(False)
        self._button = tk.Button(
            self, text=text, image="", compound="none", command=self._invoke_if_enabled, bg=surface,
            fg=text_color, activebackground=surface, activeforeground=text_color,
            disabledforeground="#8792a6", bd=0, relief="flat", highlightthickness=0,
            padx=0, pady=0, cursor="hand2", takefocus=1, state=tk.NORMAL)
        self._button.place(x=0, y=0, relwidth=1, relheight=1)
        self._content = tk.Frame(self, bg=surface, bd=0, highlightthickness=0)
        self._content.place(relx=.5, rely=.5, anchor="center")
        self._icon_label = tk.Label(self._content, image=icon, bg=surface, bd=0)
        self._icon_label.pack(side="left", padx=(0, 9))
        self._text_label = tk.Label(self._content, text=text, bg=surface, fg=text_color,
                                    font=font, bd=0, anchor="center")
        self._text_label.pack(side="left")
        self._targets = (self, self._content, self._icon_label, self._text_label)
        for target in self._targets:
            target.bind("<Configure>", self._on_resize, add="+")
            target.bind("<Enter>", self._on_enter, add="+")
            target.bind("<Leave>", self._on_leave, add="+")
        for target in (self._content, self._icon_label, self._text_label):
            target.bind("<Button-1>", self._on_child_click, add="+")
        self._button.bind("<Configure>", self._on_resize, add="+")
        self._button.bind("<Enter>", self._on_enter, add="+")
        self._button.bind("<Leave>", self._on_leave, add="+")
        self._button.bind("<FocusIn>", self._on_button_focus, add="+")
        self._button.bind("<FocusOut>", self._on_button_focus, add="+")
        self._draw_surface(initial_width, initial_height)

    def configure(self, cnf=None, **kwargs):
        kwargs = dict(kwargs)
        state = kwargs.pop("state", None)
        text = kwargs.pop("text", None)
        if text is not None and hasattr(self, "_text_label"):
            self._text = text
            self._text_label.configure(text=text)
            self._button.configure(text=text)
        result = super().configure(cnf, **kwargs)
        if state is not None and hasattr(self, "_state"):
            self._state = state
            # Отключённое действие не должно получать клавиатурный фокус:
            # иначе контур фокуса выглядит как ложное состояние активности.
            # Оставляем нативную кнопку в NORMAL: Tk добавляет stipple к
            # disabled-image и превращает гладкий градиент в точечную сетку.
            # Доступность действия отдельно проверяется в _invoke_if_enabled.
            self._button.configure(state=tk.NORMAL, takefocus=1 if state == tk.NORMAL else 0)
            if state != tk.NORMAL:
                self._focused = False
            self._text_label.configure(fg=self._text_color if state == tk.NORMAL else "#8792a6")
            self._icon_label.configure(
                image=self._icon_active if state == tk.NORMAL else self._icon_disabled)
            self.configure(cursor="hand2" if state == tk.NORMAL else "arrow")
            self._refresh_surface()
        return result

    def _invoke_if_enabled(self):
        if self._state == tk.NORMAL and self._command is not None:
            return self._command()

    config = configure

    def cget(self, key):
        if key == "text":
            return self._text
        if key == "state":
            return self._state
        return super().cget(key)

    def set_selected(self, selected: bool) -> None:
        self._selected = bool(selected)
        self._refresh_surface(force=True)

    def _on_resize(self, event):
        if event.width <= 1 or event.height <= 1:
            return
        if self._resize_job is not None:
            try:
                self.after_cancel(self._resize_job)
            except tk.TclError:
                pass
        self._resize_job = self.after(35, self._draw_resized_surface)

    def _draw_resized_surface(self):
        self._resize_job = None
        try:
            if self.winfo_exists():
                self._draw_surface(self.winfo_width(), self.winfo_height())
        except tk.TclError:
            return

    def _on_enter(self, _event):
        if self._state == tk.NORMAL:
            self._hover = True
            self._refresh_surface()

    def _on_leave(self, _event):
        self.after_idle(self._check_pointer)

    def _check_pointer(self):
        try:
            if not self.winfo_exists():
                return
            pointer = self.winfo_containing(self.winfo_pointerx(), self.winfo_pointery())
            inside = False
            while pointer is not None:
                if pointer == self:
                    inside = True
                    break
                pointer = getattr(pointer, "master", None)
            if inside != self._hover:
                self._hover = inside
                self._refresh_surface()
        except tk.TclError:
            return

    def _on_child_click(self, _event):
        if self._state == tk.NORMAL:
            self._button.focus_set()
            self._button.invoke()
            return "break"

    def _on_focus(self, _event):
        self._refresh_surface()

    def _on_button_focus(self, event):
        self._focused = event.type == tk.EventType.FocusIn
        self._refresh_surface(force=True)

    def _refresh_surface(self, force=False):
        try:
            if self.winfo_exists():
                if force:
                    self._drawn_size = None
                self._draw_surface(self.winfo_width(), self.winfo_height())
        except tk.TclError:
            return

    def _draw_surface(self, width, height):
        width, height = max(64, int(width)), max(40, int(height))
        signature = (width, height, self._state, self._hover, self._selected, self._focused)
        if signature == self._drawn_size:
            return
        self._drawn_size = signature
        scale = 3
        size = (width * scale, height * scale)
        fill = self._surface
        outline = "#6476a8" if self._state == tk.NORMAL else "#303b55"
        selected_fill = self._variant == "primary" or (self._variant == "nav" and self._selected)
        if selected_fill:
            outline = "#b4aaff" if self._state == tk.NORMAL else "#5a5685"
            if self._variant == "nav":
                start = end = "#5b50d4"
            else:
                # Отключённый CTA остаётся узнаваемо фиолетовым, но не выглядит
                # доступным действием до завершения калибровки.
                start = end = "#6658f2" if self._state == tk.NORMAL else "#39345f"
            if self._hover and self._state == tk.NORMAL:
                start = end = "#786aff"
            radius = min((18 if self._variant == "primary" else height // 2) * scale, size[1] // 2)
            mask = Image.new("L", size, 0)
            ImageDraw.Draw(mask).rounded_rectangle((1, 1, size[0] - 2, size[1] - 2),
                                                    radius=radius, fill=255)
            color = tuple(int(start[i:i + 2], 16) for i in (1, 3, 5))
            # Подсветку оставляем ВНУТРИ силуэта кнопки: внешний Gaussian blur
            # обрезался границами прямоугольной картинки и давал квадратные
            # вспышки в углах. Размываем тонкое внутреннее кольцо и снова
            # ограничиваем его скруглённой маской.
            inset = 5 * scale
            inner_mask = Image.new("L", size, 0)
            ImageDraw.Draw(inner_mask).rounded_rectangle(
                (1 + inset, 1 + inset, size[0] - 2 - inset, size[1] - 2 - inset),
                radius=max(1, radius - inset), fill=255)
            ring_mask = ImageChops.subtract(mask, inner_mask)
            glow_mask = ImageChops.multiply(
                ring_mask.filter(ImageFilter.GaussianBlur(3 * scale)), mask)
            glow = Image.new("RGBA", size, (*color, 0))
            glow.putalpha(glow_mask.point(lambda value: round(value * .24)))
            face = Image.new("RGBA", size, (*color, 0))
            face.putalpha(mask)
            # Подсветка накладывается на лицевую поверхность, но ни один её
            # пиксель не может выйти за закруглённый силуэт.
            image = Image.alpha_composite(face, glow)
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle((2, 2, size[0] - 3, size[1] - 3),
                                   radius=radius, outline=outline, width=scale)
            draw.line((radius * 1.2, 3 * scale, size[0] - radius * 1.2, 3 * scale),
                      fill=(231, 232, 255, 82), width=scale)
        else:
            image = Image.new("RGBA", size, (0, 0, 0, 0))
            if self._hover and self._state == tk.NORMAL:
                fill = "#26355b" if self._variant != "nav" else "#18203a"
            if self._variant == "nav":
                fill, outline = "#0d1326", "#2b3658"
            elif self._variant == "badge":
                fill, outline = "#10232d", "#367b78"
            ImageDraw.Draw(image).rounded_rectangle(
                (2, 2, size[0] - 3, size[1] - 3), radius=min(16 * scale, size[1] // 2),
                fill=fill, outline=outline, width=scale)
        if self._focused and self._state == tk.NORMAL:
            ImageDraw.Draw(image).rounded_rectangle(
                (5 * scale, 5 * scale, size[0] - 5 * scale, size[1] - 5 * scale),
                radius=max(4 * scale, (height // 2 - 4) * scale),
                outline="#f0f2ff", width=2 * scale)
        content_fill = start if selected_fill else fill
        # Снаружи форма прозрачна по углам, поэтому сам контейнер и слой фона
        # сохраняют цвет окна. Только внутренняя полоса с текстом получает цвет
        # заливки — иначе она закрывает скругление прямоугольником.
        self.configure(bg=self._surface)
        self._button.configure(bg=self._surface)
        for widget in (self._content, self._icon_label, self._text_label):
            widget.configure(bg=content_fill)
        self._photo = ImageTk.PhotoImage(image.resize((width, height), Image.Resampling.LANCZOS))
        # Нативная кнопка остаётся NORMAL ради чистого PIL-градиента; клики в
        # отключённом состоянии отсекает _invoke_if_enabled.
        self._button.configure(image=self._photo, compound="none")


def _render_icon(name: str, color: str, size: int = 20) -> ImageTk.PhotoImage:
    """Создаёт чёткую локальную иконку без зависимости от символьного шрифта."""
    scale = 3
    image = Image.new("RGBA", (size * scale, size * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    factor = size * scale / 24

    def point(x, y):
        return (round(x * factor), round(y * factor))

    def line(points, width=1.8):
        draw.line([point(x, y) for x, y in points], fill=color,
                  width=max(1, round(width * factor)), joint="curve")

    def rect(box, radius=0, width=1.8):
        coords = tuple(round(v * factor) for v in box)
        if radius:
            draw.rounded_rectangle(coords, radius=round(radius * factor),
                                   outline=color, width=max(1, round(width * factor)))
        else:
            draw.rectangle(coords, outline=color, width=max(1, round(width * factor)))

    if name == "home":
        line([(3, 10), (12, 3), (21, 10)])
        line([(5, 9), (5, 20), (10, 20), (10, 14), (14, 14), (14, 20), (19, 20), (19, 9)])
    elif name == "history":
        line([(3, 7), (3, 12), (8, 12)])
        draw.arc(tuple(round(v * factor) for v in (3, 3, 21, 21)), start=40, end=320,
                 fill=color, width=max(1, round(1.8 * factor)))
        line([(12, 7), (12, 12), (16, 14)])
    elif name == "layers":
        line([(3, 8), (12, 3), (21, 8), (12, 13), (3, 8)])
        line([(4, 12), (12, 17), (20, 12)])
        line([(4, 16), (12, 21), (20, 16)])
    elif name == "target":
        draw.ellipse(tuple(round(v * factor) for v in (4, 4, 20, 20)), outline=color,
                     width=max(1, round(1.8 * factor)))
        draw.ellipse(tuple(round(v * factor) for v in (9, 9, 15, 15)), outline=color,
                     width=max(1, round(1.8 * factor)))
        line([(12, 1), (12, 5)]); line([(12, 19), (12, 23)])
        line([(1, 12), (5, 12)]); line([(19, 12), (23, 12)])
    elif name == "play":
        line([(7, 4), (19, 12), (7, 20), (7, 4)])
    elif name == "database":
        draw.ellipse(tuple(round(v * factor) for v in (4, 3, 20, 9)), outline=color,
                     width=max(1, round(1.8 * factor)))
        line([(4, 6), (4, 18)])
        line([(20, 6), (20, 18)])
        draw.arc(tuple(round(v * factor) for v in (4, 9, 20, 15)), start=0, end=180,
                 fill=color, width=max(1, round(1.8 * factor)))
        draw.arc(tuple(round(v * factor) for v in (4, 15, 20, 21)), start=0, end=180,
                 fill=color, width=max(1, round(1.8 * factor)))
    elif name == "stop":
        rect((5, 5, 19, 19), radius=2)
    elif name == "activity":
        line([(2, 13), (7, 13), (10, 6), (14, 19), (17, 11), (22, 11)])
    elif name == "person":
        draw.ellipse(tuple(round(v * factor) for v in (8, 2, 16, 10)), outline=color,
                     width=max(1, round(1.8 * factor)))
        draw.arc(tuple(round(v * factor) for v in (4, 10, 20, 24)), start=180, end=360,
                 fill=color, width=max(1, round(1.8 * factor)))
        line([(4, 17), (4, 21), (20, 21), (20, 17)])
    elif name == "video":
        rect((2, 5, 15, 19), radius=2)
        line([(15, 9), (22, 5), (22, 19), (15, 15)])
    elif name in {"lock", "privacy"}:
        draw.arc(tuple(round(v * factor) for v in (6, 2, 18, 15)), start=180, end=360,
                 fill=color, width=max(1, round(1.8 * factor)))
        rect((4, 11, 20, 22), radius=2)
        draw.ellipse(tuple(round(v * factor) for v in (11, 15, 13, 17)), fill=color)
        line([(12, 17), (12, 19)], width=1.4)
        if name == "privacy":
            draw.ellipse(tuple(round(v * factor) for v in (0, 10, 5, 15)), fill=TEAL)
    else:
        raise ValueError(f"Unknown icon: {name}")

    return ImageTk.PhotoImage(image.resize((size, size), Image.Resampling.LANCZOS))


def prepare_preview_frame(frame, face_box=None, landmarks=()):
    """Нарисовать реальные контуры на немирроренной копии исходного кадра."""
    display = frame.copy()
    height, width = display.shape[:2]
    points = []
    if landmarks:
        points = [
            (int(round(np.clip(x, 0.0, 1.0) * (width - 1))),
             int(round(np.clip(y, 0.0, 1.0) * (height - 1))))
            for x, y in landmarks
        ]
        thickness = max(1, round(min(width, height) / 720))
        for start, end in FACIAL_CONTOUR_CONNECTIONS:
            if start < len(points) and end < len(points):
                cv2.line(display, points[start], points[end], (160, 128, 255), thickness,
                         lineType=cv2.LINE_AA)
        radius = max(1, min(3, round(min(width, height) / 240)))
        visible_points = sorted({index for edge in FACIAL_CONTOUR_CONNECTIONS for index in edge
                                 if index < len(points)})
        for index in visible_points:
            cv2.circle(display, points[index], radius, (232, 222, 255), -1,
                       lineType=cv2.LINE_AA)
    elif face_box:
        x1, y1, x2, y2 = face_box
        cv2.rectangle(display, (x1, y1), (x2, y2), (140, 120, 255), 2,
                      lineType=cv2.LINE_AA)
    return display


def responsive_wraplength(maximum: int, available: int) -> int:
    """Не даёт динамическому переносу выйти за дизайн-лимит или ширину ячейки."""
    maximum = max(80, int(maximum))
    available = max(80, int(available))
    return min(maximum, available)


def _available_wrap_width(widget) -> int:
    """Оценить реальное место для подписи с учётом ячейки и соседей."""
    master = widget.master
    manager = widget.winfo_manager()
    if manager == "grid":
        info = widget.grid_info()
        column, row = int(info.get("column", 0)), int(info.get("row", 0))
        column_span = max(1, int(info.get("columnspan", 1)))
        row_span = max(1, int(info.get("rowspan", 1)))
        box = master.grid_bbox(column, row, column + column_span - 1,
                               row + row_span - 1)
        cell_width = box[2] if box else 0
        return max(80, cell_width - 12)

    parent_width = master.winfo_width()
    if manager == "pack":
        info = widget.pack_info()
        own_pad = _horizontal_padding(info.get("padx", 0))
        side = info.get("side", "top")
        if side in ("left", "right"):
            sibling_width = 0
            for sibling in master.winfo_children():
                if sibling is widget or sibling.winfo_manager() != "pack":
                    continue
                sibling_info = sibling.pack_info()
                sibling_side = sibling_info.get("side", "top")
                if sibling_side in ("left", "right"):
                    sibling_width += sibling.winfo_width()
                    sibling_width += _horizontal_padding(sibling_info.get("padx", 0))
            parent_width -= sibling_width + own_pad
        else:
            parent_width -= 24 + own_pad
    else:
        parent_width -= 24
    return max(80, parent_width)


def _horizontal_padding(value) -> int:
    if isinstance(value, (tuple, list)):
        parts = value
    else:
        parts = str(value).split()
    numbers = []
    for part in parts:
        try:
            numbers.append(int(part))
        except (TypeError, ValueError):
            continue
    if len(numbers) == 1:
        return numbers[0] * 2
    return sum(numbers[:2])


def format_confidence_calibration(metrics: dict) -> str:
    """Render optional confidence diagnostics without making the report view brittle."""
    try:
        ece = float(metrics["top_label_ece"])
        brier = float(metrics["brier_score"])
        nll = float(metrics["nll"])
    except (KeyError, TypeError, ValueError):
        return "—"
    if not all(math.isfinite(value) for value in (ece, brier, nll)):
        return "—"
    return f"ECE {ece:.3f} · Brier {brier:.3f} · NLL {nll:.2f}"


class TrackerApp:
    def __init__(self, root: tk.Tk, store: JournalStore | None = None):
        self.root = root
        self.root_path = Path(__file__).resolve().parents[2]
        self.store = store or JournalStore()
        self.sample_store = LabeledSampleStore()
        self.report_directory = self.sample_store.root.parent
        self.benchmark_report_path = self.report_directory / "benchmark-report.json"
        self.personalization_report_path = self.report_directory / "personalization-report.json"
        self.adapter_path = default_adapter_path()
        self.personal_adapter = load_adapter(self.adapter_path, expected_model="enet_b0_8_best_vgaf",
                                             expected_preprocessing="tight")
        self.task_messages: queue.Queue[tuple] = queue.Queue()
        self.task_busy = False
        self.task_cancel = threading.Event()
        self.task_process: subprocess.Popen | None = None
        self.messages: queue.Queue[WorkerMessage] = queue.Queue(maxsize=2)
        self.worker: AnalysisWorker | None = None
        self._worker_generation = 0
        self._worker_stop_pending = False
        self.calibration_observations = []
        self.calibration_sampler = CalibrationSampler()
        self.profile: CalibrationProfile | None = None
        self.scorer: Scorer | None = None
        self.pattern_tracker: FacialPatternTracker | None = None
        self.event_tracker: EventTracker | None = None
        self.session: TrackingSession | None = None
        self.calibrating = False
        self.tracking = False
        self.latest_frame = None
        self.latest_observation = None
        self._last_preview_monotonic = 0.0
        self._analysis_frame_times = deque()
        # Частота записи 0.48 с; 660 точек дают запас примерно на пять минут.
        self._trend_samples = deque(maxlen=660)
        self._trend_period_seconds = 60
        self._trend_photo = None
        self._trend_bg_photo = None
        self._trend_bg_size = None
        self._last_trend_sample_at = 0.0
        self._trend_draw_pending = False
        self._active_tab = "Главная"
        self._wraplength_caps = {}
        self._wrap_refresh_after = None
        self._last_root_size = None
        self._compact_dashboard = None
        self._icons = {}
        self.photo = None
        self.current_index: float | None = None
        self.latest_score = None
        self._blind_rating_dialog: tk.Toplevel | None = None
        self._blind_rating_save_button: ttk.Button | None = None
        self._blind_rating_var: tk.StringVar | None = None
        self.current_expression: float | None = None
        self.current_deviation: float | None = None
        self.episode_open = False
        self.expression_smoother = ExpressionSmoother()
        self._build_style()
        self._build_ui()
        self.root.after_idle(self._fit_dashboard)
        self._draw_gauge(None)
        self._schedule_trend_draw()
        self._schedule_wrap_refresh()
        self.root.bind("<Configure>", self._on_root_configure, add="+")
        self.live_tab.bind("<Configure>", self._schedule_wrap_refresh, add="+")
        self.journal_tab.bind("<Configure>", self._schedule_wrap_refresh, add="+")
        self.details_canvas.bind("<Configure>", self._schedule_wrap_refresh, add="+")
        self.refresh_journal()
        self.refresh_accuracy()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(80, self._poll_messages)

    def _build_style(self):
        self.root.title("Контур — трекер мимических изменений")
        # Стартовый размер повторяет масштаб референса, но подстраивается под экран.
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        width = min(1584, max(1080, int(screen_width * 0.86)), screen_width - 48)
        height = min(990, max(700, int(screen_height * 0.90)), screen_height - 64)
        x = max(0, (screen_width - width) // 2)
        y = max(0, (screen_height - height) // 2)
        self.root.state("normal")
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(min(1080, width), min(700, height))
        self.root.configure(bg=BG)
        style = ttk.Style(self.root)
        style.theme_use("clam")
        # Стандартные ttk-кнопки на других вкладках и в диалогах используют те
        # же сглаженные формы и оттенки, что навигация и нижняя панель главного экрана.
        self._ttk_button_images = []

        def button_surface(fill: str, outline: str):
            scale, width, height, radius = 4, 160, 44, 12
            image = Image.new("RGBA", (width * scale, height * scale), (0, 0, 0, 0))
            ImageDraw.Draw(image).rounded_rectangle(
                (2, 2, width * scale - 3, height * scale - 3), radius=radius * scale,
                fill=fill, outline=outline, width=2 * scale)
            photo = ImageTk.PhotoImage(image.resize((width, height), Image.Resampling.LANCZOS))
            self._ttk_button_images.append(photo)
            return photo

        default_images = (
            button_surface(PANEL_ALT, "#46557a"),
            button_surface("#243653", "#788cff"),
            button_surface("#202b40", "#303b55"),
        )
        accent_images = (
            button_surface("#746cff", "#9389ff"),
            button_surface("#877dff", "#b0aaff"),
            button_surface("#25304a", "#35415f"),
        )
        style.element_create("Rounded.Button.border", "image", default_images[0],
                             ("active", default_images[1]), ("disabled", default_images[2]),
                             border=12, sticky="nsew")
        style.element_create("Accent.Button.border", "image", accent_images[0],
                             ("active", accent_images[1]), ("disabled", accent_images[2]),
                             border=12, sticky="nsew")
        button_label = ("Button.padding", {"sticky": "nsew", "children": [
            ("Button.label", {"sticky": "nsew"})]})
        style.layout("TButton", [("Rounded.Button.border", {"sticky": "nsew", "children": [
            ("Button.focus", {"sticky": "nsew", "children": [button_label]})]})])
        style.layout("Accent.TButton", [("Accent.Button.border", {"sticky": "nsew", "children": [
            ("Button.focus", {"sticky": "nsew", "children": [button_label]})]})])
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Alt.TFrame", background=PANEL_ALT)
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Panel.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("PanelMuted.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Segoe UI Semibold", 21))
        style.configure("Value.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI Semibold", 38))
        style.configure("TButton", font=("Segoe UI Semibold", 10), padding=(14, 9), background=PANEL_ALT,
                        foreground=TEXT, borderwidth=0, relief="flat", focuscolor="#8b82ff")
        style.map("TButton", foreground=[("disabled", "#8792a6"), ("!disabled", TEXT)])
        style.configure("Glass.TCombobox", fieldbackground="#17213a", background="#17213a",
                        foreground=TEXT, arrowcolor=TEAL, bordercolor="#465680",
                        lightcolor="#465680", darkcolor="#293554", padding=(9, 5),
                        relief="flat", insertcolor=TEXT)
        style.map("Glass.TCombobox", fieldbackground=[("readonly", "#17213a"), ("active", "#202d4a")],
                  foreground=[("readonly", TEXT)], selectbackground=[("readonly", "#292755")],
                  selectforeground=[("readonly", TEXT)])
        self.root.option_add("*TCombobox*Listbox.background", "#17213a")
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", "#574fc2")
        self.root.option_add("*TCombobox*Listbox.selectForeground", TEXT)
        style.configure("Accent.TButton", background="#7168ff", foreground="white", borderwidth=0,
                        relief="flat", focuscolor="#7168ff")
        style.map("Accent.TButton", background=[("disabled", "#202b42"), ("active", "#8c82ff"),
                                                  ("!disabled", "#7168ff")],
                  foreground=[("disabled", "#8490a5"), ("!disabled", "white")])
        style.configure("Outline.TButton", background=PANEL_ALT, foreground=TEXT, padding=(12, 8),
                        borderwidth=1, relief="flat", bordercolor="#596994", darkcolor="#18243a",
                        lightcolor="#596994", focuscolor=PANEL_ALT)
        style.map("Outline.TButton", background=[("active", "#293b5b"), ("disabled", "#1a2435")])
        style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(0, 0, 0, 8))
        style.configure("TNotebook.Tab", background=PANEL, foreground=MUTED, padding=(16, 8),
                        font=("Segoe UI Semibold", 10), borderwidth=0, focuscolor=BG)
        style.map("TNotebook.Tab", background=[("selected", PANEL_ALT), ("active", "#18253a")],
                  foreground=[("selected", TEXT), ("active", TEXT)], relief=[("selected", "flat")])
        style.configure("Treeview", background=PANEL, foreground=TEXT, fieldbackground=PANEL,
                        rowheight=30, borderwidth=0, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", background=PANEL_ALT, foreground=MUTED,
                        font=("Segoe UI Semibold", 9), borderwidth=0)
        style.map("Treeview", background=[("selected", "#2b3a5a")])
        style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=PANEL_ALT, borderwidth=0)
        style.configure("Vertical.TScrollbar", background=PANEL_ALT, troughcolor=BG,
                        bordercolor=BG, arrowcolor=MUTED)

    def _icon(self, name: str, color: str = TEXT, size: int = 20):
        key = (name, color, size)
        if key not in self._icons:
            self._icons[key] = _render_icon(name, color, size)
        return self._icons[key]

    def _set_camera_status(self, text: str, color: str) -> None:
        self.camera_status.configure(text=text, foreground=color)
        self.camera_status_dot.itemconfigure("dot", fill=color)

    def _set_landmark_status(self, observation=None) -> None:
        points = getattr(observation, "landmarks", ()) if observation is not None else ()
        present = bool(points and observation.present)
        text = f"Сетка лица · {len(points)}" if present else "Сетка лица · ожидание"
        color = TEAL if present else MUTED
        if (self.landmark_status_label.cget("text") != text or
                self.landmark_status_label.cget("fg") != color):
            self.landmark_status_label.configure(text=text, fg=color)

    def _build_ui(self):
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="both", expand=True, padx=20, pady=16)
        header = tk.Frame(outer, bg=BG, height=96)
        header.pack(fill="x", pady=(0, 12))
        header.pack_propagate(False)
        # Два уровня повторяют иерархию референса: бренд с названием сверху,
        # навигация и приватность — отдельной линией ниже.
        header.grid_columnconfigure(0, weight=1)
        header.grid_columnconfigure(1, weight=0)
        header.grid_columnconfigure(2, weight=1)
        header.grid_rowconfigure(0, weight=1)
        header.grid_rowconfigure(1, weight=1)

        identity = tk.Frame(header, bg=BG)
        identity.grid(row=0, column=0, columnspan=3, sticky="w")
        logo = tk.Canvas(identity, width=38, height=34, bg=BG, bd=0, highlightthickness=0)
        logo.pack(side="left", padx=(0, 10), pady=12)
        logo.create_polygon(4, 22, 12, 9, 16, 11, 11, 24, fill=VIOLET, outline="")
        logo.create_polygon(12, 25, 20, 5, 25, 7, 19, 26, fill="#7771ff", outline="")
        logo.create_polygon(20, 26, 28, 11, 34, 12, 27, 27, fill="#5d78e8", outline="")
        tk.Label(identity, text="КОНТУР", bg=BG, fg=TEXT,
                 font=("Segoe UI Semibold", 15)).pack(side="left")
        tk.Frame(identity, bg="#414766", width=1, height=32).pack(side="left", padx=16)
        self.window_title_label = tk.Label(identity, text="Мимика и самонаблюдение", bg=BG, fg=TEXT,
                                           font=("Segoe UI Semibold", 16))
        self.window_title_label.pack(side="left")

        nav_slot = tk.Frame(header, bg=BG)
        nav_slot.grid(row=1, column=0, columnspan=3, sticky="nsew")
        nav = tk.Frame(nav_slot, bg=BG, highlightthickness=0)
        nav.place(relx=0.5, rely=0.5, anchor="center")
        self.nav_buttons = {}
        self._nav_full_labels = {"Главное": "Главная", "История": "История",
                                 "Подробно": "Подробно"}
        self._nav_compact_labels = {"Главное": "Главная", "История": "История",
                                    "Подробно": "Детали"}
        nav_icons = {"Главное": "home", "История": "history", "Подробно": "layers"}
        for key in ("Главное", "История", "Подробно"):
            title = self._nav_full_labels[key]
            button = StyledButton(
                nav, text=title, icon=self._icon(nav_icons[key], MUTED, 18),
                command=lambda name=key: self._select_tab(name), variant="nav", surface=BG,
                font=("Segoe UI Semibold", 10), initial_width=148, initial_height=42)
            button.pack(side="left", padx=3, pady=3)
            self.nav_buttons[key] = button

        self.device_text_label = StyledButton(
            header, text="Обработка на устройстве", icon=self._icon("privacy", "#c7f5ee", 18),
            command=self.show_privacy, variant="badge", surface=BG, text_color="#c7f5ee",
            font=("Segoe UI", 9), initial_width=250, initial_height=42)
        self.device_text_label.grid(row=1, column=2, sticky="e", pady=4)
        content_host = tk.Frame(outer, bg=BG)
        content_host.pack(fill="both", expand=True)
        content_host.grid_rowconfigure(0, weight=1)
        content_host.grid_columnconfigure(0, weight=1)
        self.live_tab = tk.Frame(content_host, bg=BG)
        self.journal_tab = tk.Frame(content_host, bg=BG)
        self.accuracy_tab = tk.Frame(content_host, bg=BG)
        self._tab_frames = {"Главное": self.live_tab, "История": self.journal_tab,
                            "Подробно": self.accuracy_tab}
        for frame in self._tab_frames.values():
            frame.grid(row=0, column=0, sticky="nsew")
        details_shell = ttk.Frame(self.accuracy_tab)
        details_shell.pack(fill="both", expand=True)
        self.details_canvas = tk.Canvas(details_shell, bg=BG, highlightthickness=0)
        details_scrollbar = ttk.Scrollbar(details_shell, orient="vertical", command=self.details_canvas.yview,
                                          style="Vertical.TScrollbar")
        self.details_canvas.configure(yscrollcommand=details_scrollbar.set)
        details_scrollbar.pack(side="right", fill="y")
        self.details_canvas.pack(side="left", fill="both", expand=True)
        self.details_content = ttk.Frame(self.details_canvas)
        self.details_window = self.details_canvas.create_window((0, 0), window=self.details_content, anchor="nw")
        self.details_content.bind(
            "<Configure>", lambda _event: self.details_canvas.configure(
                scrollregion=self.details_canvas.bbox("all")))
        self.details_canvas.bind(
            "<Configure>", lambda event: self.details_canvas.itemconfigure(self.details_window, width=event.width))
        self.root.bind_all("<MouseWheel>", self._scroll_details, add="+")
        ttk.Label(self.details_content, text="ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ",
                  font=("Segoe UI Semibold", 14)).pack(anchor="w", pady=(4, 2))
        ttk.Label(self.details_content,
                  text="Основное наблюдение и управление находятся во вкладке «Главное». Здесь собраны детали модели, "
                       "мимические паттерны, личные примеры и проверка распознавания.",
                  style="Muted.TLabel", wraplength=820, justify="left").pack(anchor="w", pady=(0, 12))
        self._build_live()
        self._build_journal()
        self._build_accuracy()
        self._select_tab("Главное")

    def _fit_header(self, event=None) -> None:
        if not hasattr(self, "window_title_label"):
            return
        try:
            if not self.window_title_label.winfo_exists():
                return
            width = self.root.winfo_width() if event is None else event.width
            compact = width < 1240
            title = ("" if width < 1140 else "Самонаблюдение") if compact else "Мимика и самонаблюдение"
            badge = "Локально" if compact else "Обработка на устройстве"
            if self.window_title_label.cget("text") != title:
                self.window_title_label.configure(text=title)
            if self.device_text_label.cget("text") != badge:
                self.device_text_label.configure(text=badge)
            labels = self._nav_compact_labels if compact else self._nav_full_labels
            for key, button in self.nav_buttons.items():
                if button.winfo_exists() and button.cget("text") != labels[key]:
                    button.configure(text=labels[key])
        except tk.TclError:
            return

    def _fit_dashboard(self, event=None) -> None:
        """Уплотняет композицию по высоте, чтобы низкое окно не обрезало карточки."""
        if not hasattr(self, "score_card"):
            return
        try:
            height = self.root.winfo_height() if event is None else event.height
            compact = height < 850
            if compact == self._compact_dashboard:
                return
            self._compact_dashboard = compact
            if compact:
                self.gauge_canvas.configure(width=148, height=148)
                self.index_label.configure(font=("Segoe UI Semibold", 30))
                self.trend_canvas.configure(height=88)
                self.state_card.configure(height=88)
                self.score_card.configure(height=190)
                self.trend_card.configure(height=140)
                self.actions_card.configure(height=98)
                self.observation_card.configure(height=98)
                for row, floor in enumerate((80, 180, 132)):
                    self.right_stack.grid_rowconfigure(row, minsize=floor)
            else:
                self.gauge_canvas.configure(width=190, height=190)
                self.index_label.configure(font=("Segoe UI Semibold", 38))
                self.trend_canvas.configure(height=150)
                self.state_card.configure(height=128)
                self.score_card.configure(height=244)
                self.trend_card.configure(height=190)
                self.actions_card.configure(height=108)
                self.observation_card.configure(height=108)
                for row, floor in enumerate((112, 205, 195)):
                    self.right_stack.grid_rowconfigure(row, minsize=floor)
            self._draw_gauge(self.current_index)
            self._schedule_wrap_refresh()
        except tk.TclError:
            return

    def _on_root_configure(self, event) -> None:
        """Реагировать на изменение размера, но игнорировать перемещение окна."""
        if event.widget is not self.root:
            return
        size = (event.width, event.height)
        if size == self._last_root_size:
            return
        self._last_root_size = size
        self._fit_header(event)
        self._fit_dashboard(event)
        self._schedule_wrap_refresh(event)

    def _scroll_details(self, event):
        if self._active_tab == "Подробно":
            self.details_canvas.yview_scroll(int(-event.delta / 120), "units")
            return "break"

    def _select_tab(self, name: str) -> None:
        if name not in self._tab_frames:
            return
        self._active_tab = name
        self._tab_frames[name].tkraise()
        for key, button in self.nav_buttons.items():
            selected = key == name
            button.set_selected(selected)
            button._icon_label.configure(image=self._icon(
                {"Главное": "home", "История": "history", "Подробно": "layers"}[key],
                TEXT if selected else MUTED, 18))

    def _build_live(self):
        # Не добавляем декоративное полотно под карточками: его линии были
        # заметны в промежутках между панелями и воспринимались как сетка.
        self.live_tab.grid_columnconfigure(0, weight=52, uniform="dashboard")
        self.live_tab.grid_columnconfigure(1, weight=48, uniform="dashboard")
        self.live_tab.grid_rowconfigure(0, weight=1)
        self.live_tab.grid_rowconfigure(1, weight=0)

        # Левая колонка — главный объект экрана. Статус камеры ложится поверх
        # нижней кромки превью, как компактная стеклянная полоса в референсе.
        self.preview_card = RoundedCard(self.live_tab, padding=13, radius=20, accent="#4689cf",
                                        width=600, height=440)
        self.preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        preview_panel = self.preview_card.content
        preview_panel.grid_rowconfigure(0, weight=1)
        preview_panel.grid_columnconfigure(0, weight=1)
        self.preview = tk.Label(
            preview_panel,
            text="КАМЕРА ОСТАНОВЛЕНА\n\nНажми «Создать базу», чтобы начать",
            image=self._icon("video", "#7481ae", 44), compound="top",
            bg=self.preview_card.fill, fg="#bdc7e4", font=("Segoe UI Semibold", 15), justify="center",
            padx=10, pady=10,
        )
        self.preview.grid(row=0, column=0, sticky="nsew")
        preview_caption_card = RoundedCard(
            preview_panel, fill="#11182b", border="#485b86", radius=14,
            accent="#3caab9", padding=12, width=520, height=54)
        preview_caption_card.place(relx=.018, rely=.982, relwidth=.964, anchor="sw", height=54)
        preview_caption = preview_caption_card.content
        preview_caption.grid_columnconfigure(1, weight=1)
        preview_caption.grid_columnconfigure(3, weight=1)
        self.camera_status_dot = tk.Canvas(preview_caption, width=12, height=12, bg="#11182b",
                                           highlightthickness=0, bd=0)
        self.camera_status_dot.grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.camera_status_dot.create_oval(2, 2, 10, 10, fill=MUTED, outline="", tags="dot")
        self.camera_status = tk.Label(preview_caption, text="Камера выключена", bg="#11182b",
                                      fg=MUTED, font=("Segoe UI Semibold", 10), anchor="w")
        self.camera_status.grid(row=0, column=1, sticky="w")
        self.landmark_status_label = tk.Label(
            preview_caption, text="Ключевые точки · ожидание", bg="#11182b", fg=MUTED,
            font=("Segoe UI Semibold", 9), anchor="center",
        )
        self.landmark_status_label.grid(row=0, column=2, sticky="ew", padx=(8, 8))
        self.camera_meta_label = tk.Label(preview_caption, text="Ожидаем кадр", bg="#11182b",
                                          fg=MUTED, font=("Segoe UI", 9), anchor="e")
        self.camera_meta_label.grid(row=0, column=3, sticky="e")

        # Правая половина повторяет стек карточек из референса.
        right = tk.Frame(self.live_tab, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        self.right_stack = right
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=17, minsize=112)
        right.grid_rowconfigure(1, weight=34, minsize=205)
        right.grid_rowconfigure(2, weight=33, minsize=195)

        state_card = RoundedCard(right, padding=14, radius=20, accent="#705bfa", width=320, height=128)
        self.state_card = state_card
        state_card.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        state_card.content.grid_columnconfigure(0, weight=1)
        tk.Label(state_card.content, text="СЕЙЧАС", bg=PANEL, fg=VIOLET,
                 font=("Segoe UI Semibold", 10)).grid(row=0, column=0, sticky="w", pady=(0, 5))
        state_row = tk.Frame(state_card.content, bg=PANEL)
        state_row.grid(row=1, column=0, sticky="ew")
        state_row.grid_columnconfigure(1, weight=1)
        face_icon = tk.Canvas(state_row, width=48, height=48, bg=PANEL,
                              highlightthickness=0, bd=0)
        face_icon.grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 10))
        # Контур лица в угловой рамке подсказывает, что здесь показывается
        # состояние распознавания, а не медицинская оценка.
        face_icon.create_line(7, 17, 7, 8, 16, 8, fill=TEXT, width=2, capstyle="round")
        face_icon.create_line(32, 8, 41, 8, 41, 17, fill=TEXT, width=2, capstyle="round")
        face_icon.create_line(7, 31, 7, 40, 16, 40, fill=TEXT, width=2, capstyle="round")
        face_icon.create_line(32, 40, 41, 40, 41, 31, fill=TEXT, width=2, capstyle="round")
        face_icon.create_oval(17, 19, 19, 21, fill=VIOLET, outline="")
        face_icon.create_oval(29, 19, 31, 21, fill=VIOLET, outline="")
        face_icon.create_arc(18, 19, 30, 31, start=205, extent=130, style="arc",
                             outline=VIOLET, width=2)
        self.state_label = tk.Label(state_row, text="Ожидание лица", bg=PANEL, fg=TEXT,
                                    font=("Segoe UI Semibold", 17), anchor="w")
        self.state_label.grid(row=0, column=1, sticky="w")
        self.state_hint = tk.Label(state_row, text="Сначала откалибруй личную базу", bg=PANEL, fg=MUTED,
                                   font=("Segoe UI", 10), wraplength=340, justify="left", anchor="w")
        self.state_hint.grid(row=1, column=1, sticky="w", pady=(3, 0))
        self.quick_calibrate_btn = StyledButton(
            state_row, text="Калибровка", icon=self._icon("target", TEXT, 22),
            command=self.start_calibration, variant="secondary", surface=PANEL,
            font=("Segoe UI Semibold", 9), initial_width=150, initial_height=48)
        self.quick_calibrate_btn.grid(row=0, column=2, rowspan=2, sticky="e", padx=(12, 0))

        score_card = RoundedCard(right, padding=14, radius=20, accent="#855fff", width=320, height=244)
        self.score_card = score_card
        score_card.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        score_card.content.grid_rowconfigure(1, weight=1)
        score_card.content.grid_columnconfigure(1, weight=1)
        tk.Label(score_card.content, text="ИЗМЕНЕНИЕ МИМИКИ", bg=PANEL, fg=VIOLET,
                 font=("Segoe UI Semibold", 10)).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 7))
        self.gauge_canvas = tk.Canvas(score_card.content, width=190, height=190, bg=PANEL,
                                     highlightthickness=0, bd=0)
        self.gauge_canvas.grid(row=1, column=0, sticky="w", padx=(0, 14))
        self.index_label = tk.Label(self.gauge_canvas, text="—", bg=PANEL, fg=TEXT,
                                    font=("Segoe UI Semibold", 38))
        self.gauge_value_window = self.gauge_canvas.create_window(90, 78, window=self.index_label)
        self.gauge_caption = self.gauge_canvas.create_text(90, 116, text="из 100", fill=MUTED,
                                                           font=("Segoe UI", 10), tags="gauge-caption")
        self._gauge_photo = None
        self._gauge_image_item = self.gauge_canvas.create_image(90, 90, tags="gauge-ring")
        self.gauge_arc = None
        self.gauge_canvas.bind("<Configure>", lambda _event: self._draw_gauge(self.current_index), add="+")
        metrics = tk.Frame(score_card.content, bg=PANEL)
        metrics.grid(row=1, column=1, sticky="nsew")
        metrics.grid_columnconfigure(0, weight=1)
        metrics.grid_rowconfigure(0, weight=1)
        metrics.grid_rowconfigure(7, weight=1)
        expression_header = tk.Frame(metrics, bg=PANEL)
        expression_header.grid(row=1, column=0, sticky="w")
        tk.Label(expression_header, image=self._icon("person", VIOLET, 20), bg=PANEL).pack(side="left", padx=(0, 8))
        tk.Label(expression_header, text="УСТОЙЧИВАЯ МИМИКА", bg=PANEL, fg=MUTED,
                 font=("Segoe UI Semibold", 9)).pack(side="left")
        self.expression_stable_label = tk.Label(metrics, text="Недостаточно данных", bg=PANEL, fg=TEXT,
                                                font=("Segoe UI Semibold", 14), anchor="w", justify="left",
                                                wraplength=360)
        self.expression_stable_label.grid(row=2, column=0, sticky="w", pady=(5, 8))
        tk.Frame(metrics, bg="#34415f", height=1).grid(row=3, column=0, sticky="ew", pady=(0, 8))
        quality_header = tk.Frame(metrics, bg=PANEL)
        quality_header.grid(row=4, column=0, sticky="w")
        tk.Label(quality_header, image=self._icon("video", TEAL, 20), bg=PANEL).pack(side="left", padx=(0, 8))
        tk.Label(quality_header, text="КАЧЕСТВО КАДРА", bg=PANEL, fg=MUTED,
                 font=("Segoe UI Semibold", 9)).pack(side="left")
        quality_row = tk.Frame(metrics, bg=PANEL)
        quality_row.grid(row=5, column=0, sticky="ew", pady=(3, 0))
        self.quality_label = tk.Label(quality_row, text="Ожидание сигнала", bg=PANEL, fg=TEXT,
                                      font=("Segoe UI Semibold", 14), anchor="w", justify="left",
                                      wraplength=360)
        self.quality_label.pack(side="left", anchor="w")
        self.quality_bars_canvas = tk.Canvas(quality_row, width=82, height=20, bg=PANEL,
                                             highlightthickness=0, bd=0)
        self.quality_bars_canvas.pack(side="right", padx=(6, 0))
        self._draw_quality_bars(None)

        trend_card = RoundedCard(right, padding=14, radius=20, accent="#3fcbd5", width=320, height=190)
        self.trend_card = trend_card
        trend_card.grid(row=2, column=0, sticky="nsew")
        trend_card.content.grid_rowconfigure(1, weight=1)
        trend_card.content.grid_columnconfigure(0, weight=1)
        trend_header = tk.Frame(trend_card.content, bg=PANEL)
        trend_header.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        tk.Label(trend_header, text="ДИНАМИКА СЕССИИ", bg=PANEL, fg=VIOLET,
                 font=("Segoe UI Semibold", 10)).pack(side="left")
        self.trend_period_var = tk.StringVar(value="Последние 60 секунд")
        self.trend_period_menu = ttk.Combobox(
            trend_header, textvariable=self.trend_period_var, state="readonly",
            values=("Последние 60 секунд", "Последние 5 минут"), width=22,
            style="Glass.TCombobox")
        self.trend_period_menu.pack(side="right")
        self.trend_period_menu.bind("<<ComboboxSelected>>", self._set_trend_period, add="+")
        self.trend_canvas = tk.Canvas(trend_card.content, height=150, bg=PANEL,
                                     highlightthickness=0, bd=0)
        self.trend_canvas.grid(row=1, column=0, sticky="nsew")
        self.trend_canvas.bind("<Configure>", self._schedule_trend_draw, add="+")

        # Нижняя строка действий и краткого объяснения.
        actions_card = RoundedCard(self.live_tab, padding=12, radius=20, accent="#645cff",
                                   width=420, height=108)
        self.actions_card = actions_card
        actions_card.grid(row=1, column=0, sticky="ew", padx=(0, 10), pady=(12, 0))
        actions = actions_card.content
        actions.grid_rowconfigure(0, weight=1)
        for column in range(3):
            actions.grid_columnconfigure(column, weight=1, uniform="action")
        self.start_btn = StyledButton(actions, text="Начать сессию", icon=self._icon("play", TEXT, 24),
                                       variant="primary", surface=PANEL,
                                       command=self.start_session, state="disabled",
                                       font=("Segoe UI Semibold", 11))
        self.start_btn.grid(row=0, column=1, sticky="ew", padx=4)
        self.calibrate_btn = StyledButton(actions, text="Создать базу", icon=self._icon("database", TEXT, 24),
                                           variant="secondary", surface=PANEL,
                                           command=self.start_calibration,
                                           font=("Segoe UI Semibold", 11))
        self.calibrate_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4), ipady=5)
        self.finish_btn = StyledButton(actions, text="Завершить", icon=self._icon("stop", MUTED, 24),
                                        variant="secondary", surface=PANEL,
                                        command=self.finish_session, state="disabled",
                                        font=("Segoe UI Semibold", 11))
        self.finish_btn.grid(row=0, column=2, sticky="ew", padx=(4, 0), ipady=5)
        self.start_btn.grid_configure(ipady=5)
        self.calibration_progress = ttk.Progressbar(actions, maximum=50, mode="determinate")
        self.calibration_progress.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(7, 4))
        self.calibration_label = ttk.Label(
            actions,
            text="Для базы смотри прямо с обычным выражением; нужно около пяти секунд хорошего света и фокуса.",
            style="Muted.TLabel", wraplength=520, justify="left")
        self.calibration_label.grid(row=2, column=0, columnspan=3, sticky="w")
        self.calibration_progress.grid_remove()
        self.calibration_label.grid_remove()

        observation_card = RoundedCard(self.live_tab, padding=14, radius=20, accent="#3ec9d4",
                                       width=420, height=108)
        self.observation_card = observation_card
        observation_card.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=(12, 0))
        observation = observation_card.content
        observation.grid_columnconfigure(1, weight=1)
        self._pulse_icon_photo = rounded_icon_tile(
            self._icon("activity", VIOLET, 24), 40, "#20224a", radius=12)
        pulse = tk.Label(observation, image=self._pulse_icon_photo,
                         bg=PANEL, bd=0, highlightthickness=0)
        pulse.grid(row=0, column=0, rowspan=2, sticky="nsw", padx=(0, 10))
        tk.Label(observation, text="НАБЛЮДЕНИЕ", bg=PANEL, fg=VIOLET,
                 font=("Segoe UI Semibold", 9)).grid(row=0, column=1, sticky="w")
        self.observation_message = tk.Label(
            observation, text="Создай личную базу, чтобы начать.",
            bg=PANEL, fg=TEXT, font=("Segoe UI", 11), wraplength=440,
            justify="left", anchor="w")
        self.observation_message.grid(row=1, column=1, sticky="ew", pady=(4, 0))

        detail_surface = RoundedPanel(self.details_content, fill=PANEL, radius=16,
                                      padding=14, accent="#705bfa")
        detail_surface.pack(fill="x", pady=(0, 10))
        detail = detail_surface.content
        tk.Label(detail, text="УСТОЙЧИВЫЙ ВЫВОД · СГЛАЖЕНО ВО ВРЕМЕНИ", bg=PANEL, fg=MUTED,
                 font=("Segoe UI Semibold", 9)).pack(anchor="w", pady=(0, 5))
        self.expression_stable_detail_label = tk.Label(detail, text="Недостаточно данных", bg=PANEL, fg=TEXT,
                                                       font=("Segoe UI Semibold", 12), justify="left", anchor="w")
        self.expression_stable_detail_label.pack(fill="x", pady=(0, 10))
        tk.Label(detail, text="МГНОВЕННЫЙ ВЫВОД · TOP-3", bg=PANEL, fg=MUTED,
                 font=("Segoe UI Semibold", 9)).pack(anchor="w", pady=(0, 5))
        self.expression_label = tk.Label(detail, text="Ожидание лица", bg=PANEL, fg=TEXT,
                                         font=("Segoe UI Semibold", 12), justify="left", anchor="w")
        self.expression_label.pack(fill="x")
        self.expression_alternatives = tk.Label(detail, text="Альтернативы: —", bg=PANEL, fg=MUTED,
                                                font=("Segoe UI", 9), justify="left", anchor="w")
        self.expression_alternatives.pack(fill="x", pady=(3, 3))
        tk.Label(detail,
                 text=("Проценты — нормированные оценки модели (устойчивый вывод сглажен во времени), "
                       "не проверенная вероятность правильного распознавания выражения."),
                 bg=PANEL, fg=MUTED, font=("Segoe UI", 9), wraplength=320,
                 justify="left", anchor="w").pack(fill="x", pady=(0, 8))
        self.model_label = tk.Label(detail, text="Модели: HSEmotion + FER+ · локально", bg=PANEL, fg=MUTED,
                                    font=("Segoe UI", 9), justify="left", anchor="w", wraplength=820)
        self.model_label.pack(fill="x", pady=(0, 12))
        self.quality_diagnostics_label = tk.Label(detail, text="Качество кадра: ждём наблюдение",
                                                  bg=PANEL, fg=MUTED, font=("Segoe UI", 9),
                                                  justify="left", anchor="w", wraplength=820)
        self.quality_diagnostics_label.pack(fill="x", pady=(0, 12))
        self._refresh_model_status()
        tk.Label(detail, text="ИЗ ЧЕГО СЛОЖЕН ИНДЕКС", bg=PANEL, fg=MUTED,
                 font=("Segoe UI Semibold", 9)).pack(anchor="w", pady=(0, 10))
        self.expr_value = self._metric_row(detail, "Сдвиг выражений", "E · к личной базе")
        self.dev_value = self._metric_row(detail, "Отклонение от базы", "D · калибровка")
        tk.Label(detail, text="E — расстояние между текущим и личным базовым распределением выражений.",
                 bg=PANEL, fg=MUTED, font=("Segoe UI", 9), wraplength=820,
                 justify="left", anchor="w").pack(fill="x", pady=(7, 0))
        tk.Label(detail, text="Индекс = 50% E + 50% D · веса — экспериментальная эвристика", bg=PANEL, fg=ACCENT,
                 font=("Segoe UI Semibold", 9)).pack(anchor="w", pady=(13, 0))
        tk.Label(detail,
                 text=("Разговор и зевок могут менять мимические признаки. Если хочешь прервать такое наблюдение, "
                       "нажми «Прервать текущую сессию» и начни новую, когда будешь готов."),
                 bg=PANEL, fg=MUTED, font=("Segoe UI", 9), wraplength=320,
                 justify="left", anchor="w").pack(fill="x", pady=(5, 0))

        pattern_surface = RoundedPanel(self.details_content, fill=PANEL, radius=16,
                                       padding=14, accent="#855fff")
        pattern_surface.pack(fill="x", pady=(0, 10))
        pattern_panel = pattern_surface.content
        tk.Label(pattern_panel, text="МИМИЧЕСКИЕ ПАТТЕРНЫ · ОТДЕЛЬНО ОТ ИНДЕКСА",
                 bg=PANEL, fg=MUTED, font=("Segoe UI Semibold", 9)).pack(anchor="w")
        tk.Label(pattern_panel, text="Сравнение с твоей базой; активность должна сохраняться во времени. "
                 "Это наблюдаемые движения лица, не диагноз стресса.", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 9), wraplength=320, justify="left").pack(anchor="w", pady=(4, 7))
        self.pattern_rows: dict[str, tk.Label] = {}
        for key, title in PATTERN_LABELS.items():
            row = tk.Frame(pattern_panel, bg=PANEL)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=title, bg=PANEL, fg=TEXT, font=("Segoe UI", 9), anchor="w").pack(side="left")
            value = tk.Label(row, text="Нужна калибровка", bg=PANEL, fg=MUTED,
                             font=("Segoe UI Semibold", 9), anchor="e")
            value.pack(side="right")
            self.pattern_rows[key] = value

        au_surface = RoundedPanel(self.details_content, fill=PANEL, radius=16,
                                  padding=14, accent="#3fcbd5")
        au_surface.pack(fill="x", pady=(0, 10))
        au_panel = au_surface.content
        tk.Label(au_panel, text="FACS ACTION UNITS · МОДЕЛЬНЫЕ ОЦЕНКИ",
                 bg=PANEL, fg=MUTED, font=("Segoe UI Semibold", 9)).pack(anchor="w")
        tk.Label(au_panel, text="Py-Feat оценивает активность 20 движений лица. Значение 0–1 — выход модели, "
                 "не ручная FACS-разметка и не оценка стресса.", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 9), wraplength=320, justify="left").pack(anchor="w", pady=(4, 7))
        self.au_status_label = tk.Label(au_panel, text="Загрузка AU-модели при первом кадре",
                                        bg=PANEL, fg=MUTED, font=("Segoe UI", 9), anchor="w",
                                        wraplength=320, justify="left")
        self.au_status_label.pack(fill="x", pady=(0, 5))
        self.au_rows: dict[str, tk.Label] = {}
        for key, title in AU_LABELS_RU.items():
            row = tk.Frame(au_panel, bg=PANEL)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"{key} · {title}", bg=PANEL, fg=TEXT,
                     font=("Segoe UI", 9), anchor="w").pack(side="left")
            value = tk.Label(row, text="—", bg=PANEL, fg=MUTED,
                             font=("Segoe UI Semibold", 9), anchor="e")
            value.pack(side="right")
            self.au_rows[key] = value

        label_surface = RoundedPanel(self.details_content, fill=PANEL, radius=16,
                                     padding=14, accent="#705bfa")
        label_surface.pack(fill="x", pady=(0, 10))
        label_panel = label_surface.content
        tk.Label(label_panel, text="ЛИЧНЫЕ ПРИМЕРЫ ДЛЯ ПРОВЕРКИ", bg=PANEL, fg=MUTED,
                 font=("Segoe UI Semibold", 9)).pack(anchor="w")
        tk.Label(label_panel, text="Сохраняется только обрезка лица и только после нажатия.",
                 bg=PANEL, fg=TEXT, font=("Segoe UI", 9), wraplength=320, justify="left").pack(anchor="w", pady=(4, 8))
        label_choices = [f"{EXPRESSION_RU[label]} ({label})" for label in LABELS]
        self.sample_label_var = tk.StringVar(value=label_choices[5])
        self.sample_label_combo = ttk.Combobox(label_panel, state="readonly", values=label_choices,
                                                textvariable=self.sample_label_var)
        self.sample_label_combo.pack(fill="x")
        self.save_sample_btn = ttk.Button(label_panel, text="Сохранить пример", command=self.save_labeled_sample,
                                          state="disabled")
        self.save_sample_btn.pack(fill="x", pady=(8, 0))
        self.sample_counts_label = tk.Label(label_panel, text="", bg=PANEL, fg=MUTED,
                                            font=("Segoe UI", 9), justify="left", wraplength=320)
        self.sample_counts_label.pack(anchor="w", pady=(7, 0))
        self.delete_samples_btn = ttk.Button(label_panel, text="Удалить все примеры",
                                             command=self.delete_labeled_samples)
        self.delete_samples_btn.pack(fill="x", pady=(6, 0))
        self.sample_folder_label = tk.Label(label_panel, text=str(self.sample_store.root), bg=PANEL, fg=MUTED,
                                            font=("Segoe UI", 9), justify="left", wraplength=320)
        self.sample_folder_label.pack(anchor="w", pady=(4, 0))
        self.refresh_sample_counts()

        self.pause_btn = ttk.Button(self.details_content, text="Прервать текущую сессию",
                                    command=self.pause_session, state="disabled")
        self.pause_btn.pack(fill="x", pady=(8, 0))
        rating_surface = RoundedPanel(self.details_content, fill=PANEL, radius=16,
                                      padding=14, accent="#3ec9d4")
        rating_surface.pack(fill="x", pady=(8, 0))
        rating_panel = rating_surface.content
        tk.Label(rating_panel, text="МОЯ ОЦЕНКА СЕССИИ", bg=PANEL, fg=MUTED,
                 font=("Segoe UI Semibold", 9)).pack(anchor="w", pady=(0, 7))
        rating_row = ttk.Frame(rating_panel, style="Panel.TFrame")
        rating_row.pack(fill="x")
        ttk.Label(rating_row, text="Самочувствие 0–10", style="PanelMuted.TLabel").pack(side="left")
        self.self_rating_var = tk.StringVar(value="")
        self.self_rating_combo = ttk.Combobox(rating_row, state="readonly", width=4,
                                              values=[str(value) for value in range(11)],
                                              textvariable=self.self_rating_var)
        self.self_rating_combo.pack(side="left", padx=(8, 6))
        self.self_rating_btn = ttk.Button(rating_row, text="Сохранить", command=self.save_self_rating,
                                          state="disabled")
        self.self_rating_btn.pack(side="left")
        self.blind_rating_btn = ttk.Button(self.details_content, text="Слепая оценка · сначала я, потом индекс",
                                            command=self.start_blind_rating, state="disabled")
        self.blind_rating_btn.pack(fill="x", pady=(7, 0))
        self.self_rating_status = ttk.Label(rating_panel, text="Своя оценка необязательна.",
                                            style="PanelMuted.TLabel", wraplength=820, justify="left")
        self.self_rating_status.pack(anchor="w", pady=(6, 0))

    @staticmethod
    def _metric_row(parent, title, subtitle):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", pady=3)
        tk.Label(row, text=title, bg=PANEL, fg=TEXT, font=("Segoe UI Semibold", 10)).pack(side="left")
        tk.Label(row, text=subtitle, bg=PANEL, fg=MUTED, font=("Segoe UI", 9)).pack(side="left", padx=(8, 0))
        value = tk.Label(row, text="—", bg=PANEL, fg=TEXT, font=("Segoe UI Semibold", 12))
        value.pack(side="right")
        return value

    def _build_journal(self):
        bar = ttk.Frame(self.journal_tab)
        bar.pack(fill="x", pady=(0, 12))
        ttk.Label(bar, text="СЕССИИ И ЭПИЗОДЫ", font=("Segoe UI Semibold", 12)).pack(side="left")
        ttk.Button(bar, text="Обновить", command=self.refresh_journal).pack(side="right")
        self.delete_btn = ttk.Button(bar, text="Удалить историю", command=self.delete_history)
        self.delete_btn.pack(side="right", padx=(0, 8))
        columns = ("when", "duration", "coverage", "episodes", "ratings", "state")
        self.journal = ttk.Treeview(self.journal_tab, columns=columns, show="headings", selectmode="browse")
        for column, title, width in (("when", "Начало сессии", 240), ("duration", "Длительность", 160),
                                     ("coverage", "Покрытие", 135),
                                     ("episodes", "Эпизоды повышенного сигнала", 180),
                                     ("ratings", "Мои отметки", 100), ("state", "Завершение", 150)):
            self.journal.heading(column, text=title)
            self.journal.column(column, width=width, anchor="w")
        self.journal.pack(fill="both", expand=True)
        ttk.Label(self.journal_tab,
                  text="Покрытие — доля времени с последовательными кадрами, пригодными для расчёта. Это не точность индекса.",
                  style="Muted.TLabel", wraplength=820, justify="left").pack(anchor="w", pady=(6, 0))
        self.journal.bind("<<TreeviewSelect>>", self._show_selected_session)
        self.episode_details = ttk.Label(self.journal_tab, text="Выбери сессию, чтобы посмотреть эпизоды.",
                                         style="Muted.TLabel", justify="left", wraplength=820)
        self.episode_details.pack(fill="x", pady=(12, 0))
        self.self_rating_summary_label = ttk.Label(self.journal_tab, text="Личных отметок пока нет.",
                                                   style="Muted.TLabel", justify="left")
        self.self_rating_summary_label.pack(fill="x", pady=(8, 0))
        self.journal_tab.bind("<Configure>", self._resize_rating_summary, add="+")

    def _resize_rating_summary(self, event) -> None:
        if event.widget is self.journal_tab and hasattr(self, "self_rating_summary_label"):
            self.self_rating_summary_label.configure(wraplength=max(240, min(1000, event.width - 40)))

    def _schedule_wrap_refresh(self, _event=None) -> None:
        """Объединяет resize-события и ждёт завершения раскладки Tk."""
        try:
            if not self.root.winfo_exists():
                return
        except tk.TclError:
            return
        pending = getattr(self, "_wrap_refresh_after", None)
        if pending is not None:
            try:
                self.root.after_cancel(pending)
            except tk.TclError:
                pass
        self._wrap_refresh_after = self.root.after(45, self._refresh_wrap_lengths)

    def _refresh_wrap_lengths(self) -> None:
        self._wrap_refresh_after = None
        try:
            if not self.root.winfo_exists():
                return
            # Геометрия grid/pack пересчитывается в idle-цикле; сначала
            # завершаем его, иначе получим размеры предыдущего кадра окна.
            self.root.update_idletasks()
        except tk.TclError:
            return

        def visit(parent):
            for widget in parent.winfo_children():
                try:
                    configured = int(widget.cget("wraplength"))
                except (tk.TclError, TypeError, ValueError):
                    configured = 0
                if configured:
                    # Храним авторский максимум: перенос никогда не становится
                    # шире выбранной ячейки и не разрастается обратно на большом окне.
                    maximum = self._wraplength_caps.setdefault(widget, configured)
                    available = _available_wrap_width(widget)
                    wraplength = responsive_wraplength(maximum, available)
                    if configured != wraplength:
                        widget.configure(wraplength=wraplength)
                visit(widget)

        visit(self.root)

    def _build_accuracy(self):
        ttk.Label(self.details_content, text="ПРОВЕРКА РАСПОЗНАВАНИЯ", font=("Segoe UI Semibold", 14)).pack(anchor="w")
        ttk.Label(self.details_content,
                  text="Сравнение запускается локально на примерах, которые ты сам сохранил и разметил. "
                       "Значения появятся после накопления разнообразных сессий.",
                  style="Muted.TLabel", wraplength=820, justify="left").pack(anchor="w", pady=(5, 12))
        ttk.Label(self.details_content, text="ЛИЧНАЯ ПРОВЕРКА ЛИЦЕВОГО СИГНАЛА",
                  font=("Segoe UI Semibold", 10)).pack(anchor="w", pady=(0, 4))
        ttk.Label(self.details_content,
                  text="Оценки 0–10 сохраняются только по твоему нажатию рядом со свежим лицевым сигналом. "
                       "Они не участвуют в формуле. Описательная связь появится после нескольких сессий; "
                       "это личная проверка, не диагноз.",
                  style="Muted.TLabel", wraplength=820, justify="left").pack(anchor="w")
        self.adapter_status_label = ttk.Label(self.details_content, text="Персональная адаптация не активна",
                                              font=("Segoe UI Semibold", 10))
        self.adapter_status_label.pack(anchor="w", pady=(0, 10))
        actions = ttk.Frame(self.details_content)
        actions.pack(fill="x", pady=(0, 10))
        self.benchmark_btn = ttk.Button(actions, text="Сравнить модели", style="Accent.TButton",
                                        command=self.start_benchmark)
        self.benchmark_btn.pack(side="left")
        self.personalize_btn = ttk.Button(actions, text="Оценить адаптацию", command=self.start_personalization)
        self.personalize_btn.pack(side="left", padx=(8, 0))
        self.reset_adapter_btn = ttk.Button(actions, text="Сбросить адаптацию", command=self.reset_adapter)
        self.reset_adapter_btn.pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Обновить отчёт", command=self.refresh_accuracy).pack(side="right")
        self.accuracy_task_status = ttk.Label(self.details_content, text="Отчёт хранится локально на этом компьютере.",
                                              style="Muted.TLabel")
        self.accuracy_task_status.pack(anchor="w", pady=(0, 8))
        columns = ("model", "preprocessing", "examples", "sessions", "f1", "accuracy", "confidence", "latency")
        self.accuracy_table = ttk.Treeview(self.details_content, columns=columns, show="headings", height=8)
        headings = (("model", "Модель", 200), ("preprocessing", "Кадр", 100),
                    ("examples", "Примеры", 90), ("sessions", "Сессия×метка", 110),
                    ("f1", "Macro-F1", 100), ("accuracy", "Accuracy", 100),
                    ("confidence", "ECE / Brier / NLL", 220),
                    ("latency", "CPU мс / P95", 130))
        for key, title, width in headings:
            self.accuracy_table.heading(key, text=title)
            self.accuracy_table.column(key, width=width, anchor="w")
        self.accuracy_table.pack(fill="both", expand=True)
        ttk.Label(self.details_content, text="ТВОИ РАЗМЕЧЕННЫЕ ПРИМЕРЫ", font=("Segoe UI Semibold", 10)).pack(
            anchor="w", pady=(10, 5))
        sample_columns = ("id", "label", "created", "quality", "session")
        self.sample_table = ttk.Treeview(self.details_content, columns=sample_columns, show="headings", height=5)
        for key, title, width in (("id", "ID", 90), ("label", "Метка", 140), ("created", "Сохранён", 220),
                                  ("quality", "Качество", 90), ("session", "Capture-сессия", 110)):
            self.sample_table.heading(key, text=title)
            self.sample_table.column(key, width=width, anchor="w")
        self.sample_table.pack(fill="x")
        sample_actions = ttk.Frame(self.details_content)
        sample_actions.pack(fill="x", pady=(5, 0))
        ttk.Button(sample_actions, text="Исправить метку", command=self.relabel_selected_sample).pack(side="left")
        ttk.Button(sample_actions, text="Удалить пример", command=self.delete_selected_sample).pack(side="left", padx=(8, 0))
        ttk.Label(self.details_content,
                  text="Показатели описывают совпадение модели с твоими метками. Они не измеряют стресс "
                       "и не доказывают внутреннюю эмоцию. ECE/Brier/NLL проверяют согласованность вероятностей "
                       "с твоими жёсткими метками; на малом числе сессий ECE нестабилен, а неоднозначные "
                       "выражения одной меткой описать нельзя.",
                  style="Muted.TLabel", wraplength=820, justify="left").pack(anchor="w", pady=(10, 0))

    def refresh_accuracy(self):
        if not hasattr(self, "accuracy_table"):
            return
        for item in self.accuracy_table.get_children():
            self.accuracy_table.delete(item)
        try:
            report = json.loads(self.benchmark_report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report = None
        if report is not None:
            for row in report.get("comparisons", []):
                calibration = row.get("confidence_calibration", {})
                confidence_text = format_confidence_calibration(calibration)
                self.accuracy_table.insert("", "end", values=(
                    row.get("model_version", "—"), row.get("preprocessing", "—"),
                    row.get("examples", 0), row.get("session_label_units", 0),
                    f"{row.get('macro_f1', 0):.3f}", f"{row.get('accuracy', 0):.3f}",
                    confidence_text,
                    f"{row.get('mean_latency_ms', 0):.1f} / {row.get('p95_latency_ms', 0):.1f}",
                ))
            generated = report.get("generated_at", "")
            self.accuracy_task_status.configure(text=f"Отчёт: {generated or self.benchmark_report_path}")
        else:
            self.accuracy_task_status.configure(text=f"Отчёта пока нет. Размеченные примеры: {len(self.sample_store.list_samples())}")
        if self.personal_adapter:
            self.adapter_status_label.configure(
                text=(f"Персональная поправка активна · предварительный macro-F1 "
                      f"{self.personal_adapter.baseline_macro_f1:.3f} → "
                      f"{self.personal_adapter.adapted_macro_f1:.3f} на отложенных сессиях. "
                      "Не независимая гарантия качества."), foreground=GREEN)
        else:
            self.adapter_status_label.configure(text="Персональная адаптация не активна · используется общая модель",
                                                 foreground=MUTED)
        self.refresh_sample_table()

    def refresh_self_rating_summary(self):
        if not hasattr(self, "self_rating_summary_label"):
            return
        rows = self.store.list_all_self_ratings()
        lines = ["Описательная связь мимического сигнала с самооценкой · режимы отдельно:"]
        for mode, title in (("blind", "Без показа индекса"), ("visible", "С видимым индексом")):
            summary = self_rating_summary(rows, mode=mode)
            if summary["rho"] is None:
                detail = f"{summary['ratings']} отметок · {summary['sessions']} информативных сессий · пока мало данных"
            else:
                detail = (f"{summary['ratings']} отметок · {summary['sessions']} сессий · "
                          f"медиана внутрисессионного r = {summary['rho']:+.2f} "
                          f"(разброс {summary['session_rho_min']:+.2f}…{summary['session_rho_max']:+.2f})")
            lines.append(f"{title}: {detail}.")
        lines.append("Сессии имеют равный вес; описательная N-of-1 связь, не точность и не диагноз. "
                     "Нужны ≥10 отметок в ≥3 информативных сессиях, минимум 3 отметки в каждой; "
                     "соседние оценки могут быть зависимы.")
        text = "\n".join(lines)
        self.self_rating_summary_label.configure(text=text)

    def refresh_sample_table(self):
        if not hasattr(self, "sample_table"):
            return
        for item in self.sample_table.get_children():
            self.sample_table.delete(item)
        for sample in self.sample_store.list_samples():
            local_time = sample.created_at.replace("T", " ").replace("+00:00", " UTC")
            self.sample_table.insert("", "end", iid=sample.sample_id, values=(
                sample.sample_id[:8], EXPRESSION_RU.get(sample.label, sample.label), local_time,
                f"{sample.quality:.0%}", sample.session_id[:8],
            ))

    def _invalidate_derived_analysis(self):
        for path in (self.adapter_path, self.benchmark_report_path, self.benchmark_report_path.with_suffix(".csv"),
                     self.personalization_report_path):
            path.unlink(missing_ok=True)
        self.personal_adapter = None
        self._refresh_model_status()
        self.refresh_accuracy()

    def relabel_selected_sample(self):
        if self.task_busy:
            messagebox.showinfo("Идёт обработка", "Дождись завершения анализа перед изменением разметки.")
            return
        if self.tracking or self.calibrating:
            messagebox.showinfo("Камера активна", "Приостанови камеру, чтобы редактировать разметку.")
            return
        selection = self.sample_table.selection()
        if not selection:
            messagebox.showinfo("Выбери пример", "Выбери сохранённый пример в таблице ниже.")
            return
        sample = next(item for item in self.sample_store.list_samples() if item.sample_id == selection[0])
        dialog = tk.Toplevel(self.root)
        dialog.title("Исправить метку выражения")
        dialog.transient(self.root)
        dialog.grab_set()
        ttk.Label(dialog, text="Выбери выражение, которое ты хотел отметить:").pack(padx=18, pady=(18, 8))
        choices = [f"{EXPRESSION_RU[label]} ({label})" for label in LABELS]
        selected = tk.StringVar(value=f"{EXPRESSION_RU[sample.label]} ({sample.label})")
        combo = ttk.Combobox(dialog, state="readonly", values=choices, textvariable=selected, width=30)
        combo.pack(padx=18, fill="x")

        def save():
            label = selected.get().rsplit("(", 1)[-1].rstrip(")")
            self.sample_store.update_label(sample.sample_id, label)
            self.refresh_sample_counts()
            self._invalidate_derived_analysis()
            dialog.destroy()
            messagebox.showinfo("Метка исправлена", "Сравнение и адаптер очищены. Запусти сравнение заново.")

        ttk.Button(dialog, text="Сохранить метку", style="Accent.TButton", command=save).pack(
            padx=18, pady=18, fill="x")

    def delete_selected_sample(self):
        if self.task_busy:
            messagebox.showinfo("Идёт обработка", "Дождись завершения анализа перед удалением примера.")
            return
        if self.tracking or self.calibrating:
            messagebox.showinfo("Камера активна", "Приостанови камеру, чтобы удалить размеченный пример.")
            return
        selection = self.sample_table.selection()
        if not selection:
            messagebox.showinfo("Выбери пример", "Выбери сохранённый пример в таблице ниже.")
            return
        if not messagebox.askyesno("Удалить пример?", "Удалить выбранную обрезку лица и её метку? Это нельзя отменить."):
            return
        self.sample_store.delete_sample(selection[0])
        self.refresh_sample_counts()
        self._invalidate_derived_analysis()

    def _refresh_model_status(self):
        if hasattr(self, "model_label"):
            suffix = " · личная поправка активна" if self.personal_adapter else ""
            self.model_label.configure(text=f"Модели: HSEmotion + FER+ · локально{suffix}")

    def _tools_available(self) -> bool:
        if self.tracking or self.calibrating:
            messagebox.showinfo("Сначала останови камеру", "Останови сессию перед сравнением или обучением.")
            return False
        if self.task_busy:
            messagebox.showinfo("Задача выполняется", "Дождись окончания текущего локального расчёта.")
            return False
        return True

    def start_benchmark(self):
        if not self._tools_available():
            return
        count = len(self.sample_store.list_samples())
        if count == 0:
            messagebox.showinfo("Нет размеченных примеров", "Сначала запусти сессию и сохрани несколько примеров с выбранными метками.")
            return
        args = [str(self.root_path / "tools" / "benchmark_models.py"), "--samples", str(self.sample_store.root),
                "--output", str(self.benchmark_report_path)]
        self._launch_local_tool("benchmark", args, "Сравниваю модели на локальных примерах…")

    def start_personalization(self):
        if not self._tools_available():
            return
        if not self.benchmark_report_path.is_file():
            messagebox.showinfo("Сначала сравни модели", "Персональная адаптация использует отчёт локального сравнения. Сначала запусти «Сравнить модели».")
            return
        args = [str(self.root_path / "tools" / "train_personal_adapter.py"), str(self.benchmark_report_path),
                "--report", str(self.personalization_report_path), "--adapter", str(self.adapter_path)]
        self._launch_local_tool("personalization", args, "Проверяю адаптацию на отложенных capture-сессиях…")

    def _launch_local_tool(self, task: str, arguments: list[str], status: str):
        self.task_busy = True
        self.task_cancel.clear()
        self.accuracy_task_status.configure(text=status, foreground=AMBER)
        self.benchmark_btn.configure(state="disabled")
        self.personalize_btn.configure(state="disabled")
        self.reset_adapter_btn.configure(state="disabled")

        def run():
            environment = os.environ.copy()
            source_path = str(self.root_path / "src")
            environment["PYTHONPATH"] = source_path + (os.pathsep + environment["PYTHONPATH"]
                                                         if environment.get("PYTHONPATH") else "")
            environment["PYTHONUTF8"] = "1"
            process = None
            try:
                process = subprocess.Popen([sys.executable, *arguments], cwd=self.root_path, env=environment,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                           encoding="utf-8", errors="replace")
                self.task_process = process
                while True:
                    try:
                        stdout, stderr = process.communicate(timeout=0.25)
                        break
                    except subprocess.TimeoutExpired:
                        if self.task_cancel.is_set():
                            process.terminate()
                            stdout, stderr = process.communicate(timeout=2)
                            break
                self.task_messages.put((task, process.returncode or 0, stdout, stderr))
            except Exception as exc:
                self.task_messages.put((task, 1, "", str(exc)))
            finally:
                self.task_process = None

        threading.Thread(target=run, name=f"kontur-{task}", daemon=True).start()

    def _handle_task_messages(self):
        while True:
            try:
                task, return_code, stdout, stderr = self.task_messages.get_nowait()
            except queue.Empty:
                break
            self.task_busy = False
            self.benchmark_btn.configure(state="normal")
            self.personalize_btn.configure(state="normal")
            self.reset_adapter_btn.configure(state="normal")
            if return_code != 0:
                detail = (stderr or stdout or "Неизвестная ошибка")[-1200:]
                self.accuracy_task_status.configure(text="Локальная задача завершилась с ошибкой", foreground=RED)
                messagebox.showerror("Ошибка обработки", detail)
                continue
            if task == "benchmark":
                self.refresh_accuracy()
                report = json.loads(self.benchmark_report_path.read_text(encoding="utf-8"))
                comparisons = report.get("comparisons", [])
                best = max(comparisons, key=lambda row: row["macro_f1"]) if comparisons else None
                if best:
                    message = (f"Лучшая текущая строка отчёта: {best['model_version']} · {best['preprocessing']}\n"
                               f"Macro-F1 {best['macro_f1']:.3f} · сессий×меток {best['session_label_units']}\n\n"
                               "Это предварительная оценка только по твоим размеченным примерам.")
                else:
                    message = "Отчёт создан, но пока нет данных для сравнения."
                self.accuracy_task_status.configure(text="Сравнение завершено; отчёт сохранён локально.", foreground=GREEN)
                messagebox.showinfo("Сравнение готово", message)
            elif task == "personalization":
                result = json.loads(self.personalization_report_path.read_text(encoding="utf-8"))
                if result.get("activated"):
                    self.personal_adapter = load_adapter(self.adapter_path, expected_model="enet_b0_8_best_vgaf",
                                                         expected_preprocessing="tight")
                    self._refresh_model_status()
                self.refresh_accuracy()
                self.accuracy_task_status.configure(text=result.get("reason", "Адаптация оценена"),
                                                    foreground=GREEN if result.get("activated") else MUTED)
                messagebox.showinfo("Оценка адаптации", result.get("reason", "Адаптация оценена"))

    def reset_adapter(self):
        if not self._tools_available():
            return
        if messagebox.askyesno("Сбросить адаптацию?", "Убрать персональную поправку и вернуться к общей модели?"):
            self.adapter_path.unlink(missing_ok=True)
            self.personal_adapter = None
            self.expression_smoother.reset()
            self._refresh_model_status()
            self.refresh_accuracy()

    def start_calibration(self):
        if self.task_busy:
            self.state_hint.configure(text="Дождись завершения локального сравнения")
            return
        if self.worker and self.worker.running:
            self.state_hint.configure(text="Дождись, пока остановится камера")
            return
        if self.tracking:
            return
        # Invalidate any queued messages from a worker that is finishing a prior task.
        self._worker_generation += 1
        self.calibration_observations.clear()
        self.calibration_sampler.start(time.monotonic())
        self.calibrating = True
        # Вспомогательная инструкция занимает место только во время калибровки.
        self.calibration_progress.grid()
        self.calibration_label.grid()
        self.calibration_progress.configure(value=0)
        self.calibrate_btn.configure(state="disabled")
        self.calibration_label.configure(
            text="Смотри прямо с обычным выражением, обеспечь хороший свет и фокус; не закрывай лицо руками или предметами…"
        )
        self.observation_message.configure(text="Собираю твою личную базу.")
        self.state_label.configure(text="Калибровка", fg=ACCENT)
        self.state_hint.configure(text="Смотри прямо и не разговаривай около пяти секунд.")
        self._start_worker()

    def _set_rating_buttons(self, state: str) -> None:
        if hasattr(self, "self_rating_btn"):
            self.self_rating_btn.configure(state=state)
        if hasattr(self, "blind_rating_btn"):
            self.blind_rating_btn.configure(state=state)

    def _refresh_blind_rating_save_state(self) -> None:
        button = self._blind_rating_save_button
        rating_var = self._blind_rating_var
        if button is None or rating_var is None:
            return
        try:
            if not button.winfo_exists():
                return
        except tk.TclError:
            return
        ready = bool(rating_var.get()) and self._rating_snapshot() is not None
        button.configure(state="normal" if ready else "disabled")

    def start_session(self):
        if self.task_busy:
            self.state_hint.configure(text="Дождись завершения локального сравнения")
            return
        if self.profile is None:
            return
        if self.worker and self.worker.running:
            self.state_hint.configure(text="Дождись, пока остановится камера")
            return
        self.scorer = Scorer(self.profile)
        self.pattern_tracker = FacialPatternTracker(self.profile)
        self._show_patterns(None)
        self.event_tracker = EventTracker()
        self.expression_smoother.reset()
        self.session = TrackingSession(self.store, self.event_tracker)
        self.session.start(datetime.now().timestamp())
        self.refresh_journal()
        self.tracking = True
        self.observation_message.configure(text="Сигнал меняется плавно.")
        self.episode_open = False
        self.latest_score = None
        self.current_index = None
        self._trend_samples.clear()
        self._last_trend_sample_at = 0.0
        self._draw_gauge(None)
        self._schedule_trend_draw()
        self._set_rating_buttons("disabled")
        self.self_rating_status.configure(text="Дождись свежего лицевого сигнала.")
        self.start_btn.configure(state="disabled")
        self.pause_btn.configure(state="normal", text="Прервать текущую сессию")
        self.finish_btn.configure(state="normal")
        self.calibrate_btn.configure(state="disabled")
        self.calibration_label.configure(text="Сессия идёт; устойчивые эпизоды появятся в журнале")
        self.state_hint.configure(text="Индекс сглаживается; одиночные движения игнорируются")
        self._start_worker()

    def pause_session(self):
        if self.tracking:
            self.tracking = False
            self.current_index = None
            self.latest_score = None
            self._draw_gauge(None)
            self.quality_label.configure(text="Нет активного кадра", fg=MUTED)
            self._draw_quality_bars(None)
            self._set_rating_buttons("disabled")
            self._worker_generation += 1
            if self.session:
                self.session.finish(datetime.now().timestamp(), interrupted=True)
            self.session = None
            stopped = self._stop_worker()
            self.pause_btn.configure(state="disabled", text="Прервать текущую сессию")
            self.finish_btn.configure(state="disabled")
            self.start_btn.configure(state="normal" if stopped else "disabled")
            self.calibrate_btn.configure(state="normal")
            self.state_label.configure(text="Пауза", fg=MUTED)
            self.state_hint.configure(text="Сессия прервана; нажми «Начать», чтобы создать новую.")
            self.observation_message.configure(text="Наблюдение приостановлено.")
            self._clear_expression()
            self.refresh_journal()

    def finish_session(self):
        if not self.tracking:
            return
        self.tracking = False
        self.current_index = None
        self.latest_score = None
        self._draw_gauge(None)
        self.quality_label.configure(text="Нет активного кадра", fg=MUTED)
        self._draw_quality_bars(None)
        self._set_rating_buttons("disabled")
        self._worker_generation += 1
        if self.session:
            self.session.finish(datetime.now().timestamp(), interrupted=False)
        self.session = None
        stopped = self._stop_worker()
        self.pause_btn.configure(state="disabled", text="Прервать текущую сессию")
        self.finish_btn.configure(state="disabled")
        self.start_btn.configure(state="normal" if stopped else "disabled")
        self.calibrate_btn.configure(state="normal")
        self.state_label.configure(text="Сессия завершена", fg=GREEN)
        self.state_hint.configure(text="Можно запустить новую сессию")
        self.observation_message.configure(text="Сессия завершена.")
        self._clear_expression()
        self.refresh_journal()

    def _start_worker(self):
        if self.worker:
            if self.worker.running:
                return
            self.worker = None
        self._worker_generation += 1
        self._analysis_frame_times.clear()
        self.worker = AnalysisWorker(self.messages, generation=self._worker_generation)
        self.worker.start()
        self._set_camera_status("Камера запускается…", AMBER)
        self.camera_meta_label.configure(text="Подключение к видеопотоку")

    def _record_analysis_rate(self, captured_monotonic: float | None = None) -> float | None:
        try:
            captured_at = float(captured_monotonic)
        except (TypeError, ValueError):
            captured_at = time.monotonic()
        if not math.isfinite(captured_at):
            captured_at = time.monotonic()
        if self._analysis_frame_times and captured_at <= self._analysis_frame_times[-1]:
            self._analysis_frame_times.clear()
        self._analysis_frame_times.append(captured_at)
        cutoff = captured_at - 5.0
        while self._analysis_frame_times and self._analysis_frame_times[0] < cutoff:
            self._analysis_frame_times.popleft()
        if len(self._analysis_frame_times) < 2:
            return None
        span = self._analysis_frame_times[-1] - self._analysis_frame_times[0]
        return (len(self._analysis_frame_times) - 1) / span if span > 0 else None

    def _draw_gauge(self, value: float | None) -> None:
        """Рисует плавный светящийся gauge без выдуманного значения в пустом состоянии."""
        if not hasattr(self, "gauge_canvas"):
            return
        self.index_label.configure(text="—" if value is None else f"{value:.0f}")
        canvas_width = self.gauge_canvas.winfo_width()
        canvas_height = self.gauge_canvas.winfo_height()
        if canvas_width <= 1:
            canvas_width = int(self.gauge_canvas.cget("width"))
        if canvas_height <= 1:
            canvas_height = int(self.gauge_canvas.cget("height"))
        canvas_width, canvas_height = max(1, canvas_width), max(1, canvas_height)
        scale = 3
        image_size = (canvas_width * scale, canvas_height * scale)
        image = Image.new("RGBA", image_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        cx, cy = canvas_width * scale / 2, canvas_height * scale / 2
        radius = max(12 * scale, min(canvas_width, canvas_height) * scale * .385)
        ring_width = max(7 * scale, round(min(canvas_width, canvas_height) * scale * .061))
        bounds = (cx - radius, cy - radius, cx + radius, cy + radius)
        draw.ellipse(bounds, outline="#293654", width=ring_width)
        # Тонкий внутренний кант и метки придают приборный вид, оставаясь тише шкалы.
        inset = ring_width * .9
        draw.ellipse((bounds[0] + inset, bounds[1] + inset,
                      bounds[2] - inset, bounds[3] - inset), outline="#1b2741", width=scale)
        for tick in range(32):
            angle = math.radians(-90 + tick * 360 / 32)
            major = tick % 8 == 0
            inner = radius + ring_width * .72
            outer = inner + (6 if major else 3) * scale
            tick_color = "#7883a8" if major else "#374363"
            draw.line((cx + math.cos(angle) * inner, cy + math.sin(angle) * inner,
                       cx + math.cos(angle) * outer, cy + math.sin(angle) * outer),
                      fill=tick_color, width=scale if major else max(1, scale - 1))

        if value is not None:
            progress = max(0.0, min(100.0, float(value))) / 100.0
            end_angle = -90 + 360 * progress
            step_count = max(2, int(180 * progress))
            points = []
            for index in range(step_count + 1):
                angle = math.radians(-90 + (end_angle + 90) * index / step_count)
                points.append((cx + math.cos(angle) * radius, cy + math.sin(angle) * radius))
            glow_mask = Image.new("L", image_size, 0)
            if len(points) > 1:
                ImageDraw.Draw(glow_mask).line(points, fill=210, width=ring_width + 10 * scale, joint="curve")
                glow_mask = glow_mask.filter(ImageFilter.GaussianBlur(6 * scale))
                glow = Image.new("RGBA", image_size, (112, 93, 255, 0))
                glow.putalpha(glow_mask.point(lambda alpha: round(alpha * .64)))
                image = Image.alpha_composite(image, glow)
                draw = ImageDraw.Draw(image)
                violet = (143, 117, 255)
                cyan = (66, 226, 231)
                for index, (start, finish) in enumerate(zip(points, points[1:])):
                    mix = index / max(1, len(points) - 2)
                    color = tuple(round(violet[channel] * (1 - mix) + cyan[channel] * mix)
                                  for channel in range(3))
                    draw.line((start, finish), fill=(*color, 255), width=ring_width, joint="curve")
                for point, mix in ((points[0], 0.0), (points[-1], 1.0)):
                    color = tuple(round(violet[channel] * (1 - mix) + cyan[channel] * mix)
                                  for channel in range(3))
                    cap = ring_width / 2
                    draw.ellipse((point[0] - cap, point[1] - cap,
                                  point[0] + cap, point[1] + cap), fill=(*color, 255))
            if progress > 0:
                endpoint = points[-1]
                dot_radius = ring_width * .42
                draw.ellipse((endpoint[0] - dot_radius, endpoint[1] - dot_radius,
                              endpoint[0] + dot_radius, endpoint[1] + dot_radius),
                             fill="#b5ffff", outline="#ffffff", width=scale)

        self._gauge_photo = ImageTk.PhotoImage(image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS))
        self.gauge_canvas.itemconfigure(self._gauge_image_item, image=self._gauge_photo)
        cx, cy = canvas_width / 2, canvas_height / 2
        self.gauge_canvas.coords(self.gauge_value_window, cx, cy - 12)
        self.gauge_canvas.coords(self.gauge_caption, cx, cy + 27)
        self.gauge_canvas.tag_raise("gauge-caption")

    def _draw_quality_bars(self, value: float | None) -> None:
        if not hasattr(self, "quality_bars_canvas"):
            return
        canvas = self.quality_bars_canvas
        canvas.delete("all")
        filled = 0 if value is None else round(max(0.0, min(1.0, value)) * 5)
        for index, height in enumerate((7, 9, 12, 15, 18)):
            x0 = 2 + index * 15
            y0 = 19 - height
            color = TEAL if index < filled else "#303951"
            canvas.create_rectangle(x0, y0, x0 + 10, 19, fill=color, outline="")

    def _set_stable_label(self, text: str) -> None:
        self.expression_stable_label.configure(text=text)
        if hasattr(self, "expression_stable_detail_label"):
            self.expression_stable_detail_label.configure(text=text)

    def _record_trend_sample(self, value: float | None) -> None:
        now = time.monotonic()
        # Рисуем график дважды в секунду максимум; это не добавляет нагрузки к видеопотоку.
        if now - self._last_trend_sample_at < 0.48:
            return
        self._last_trend_sample_at = now
        sample = None
        if value is not None:
            try:
                parsed = float(value)
                sample = parsed if math.isfinite(parsed) else None
            except (TypeError, ValueError):
                sample = None
        self._trend_samples.append((now, sample))
        cutoff = now - 300.0
        while self._trend_samples and self._trend_samples[0][0] < cutoff:
            self._trend_samples.popleft()
        self._schedule_trend_draw()

    def _set_trend_period(self, _event=None) -> None:
        self._trend_period_seconds = 300 if self.trend_period_var.get().endswith("5 минут") else 60
        self._schedule_trend_draw()

    @staticmethod
    def _smooth_spline(points: list[tuple[float, float]], steps: int = 8) -> list[tuple[float, float]]:
        """Catmull–Rom spline для плавной линии без скачивания JS-библиотеки."""
        if len(points) < 3:
            return points
        smooth = [points[0]]
        for index in range(len(points) - 1):
            p0 = points[index - 1] if index > 0 else points[index]
            p1 = points[index]
            p2 = points[index + 1]
            p3 = points[index + 2] if index + 2 < len(points) else p2
            for step in range(1, steps + 1):
                t = step / steps
                t2, t3 = t * t, t * t * t
                coordinates = []
                for axis in range(2):
                    v0, v1, v2, v3 = p0[axis], p1[axis], p2[axis], p3[axis]
                    coordinate = .5 * ((2 * v1) + (-v0 + v2) * t
                                       + (2 * v0 - 5 * v1 + 4 * v2 - v3) * t2
                                       + (-v0 + 3 * v1 - 3 * v2 + v3) * t3)
                    coordinates.append(coordinate)
                smooth.append((coordinates[0], coordinates[1]))
        return smooth

    def _schedule_trend_draw(self, _event=None) -> None:
        if not hasattr(self, "trend_canvas") or self._trend_draw_pending:
            return
        try:
            if not self.root.winfo_exists():
                return
        except tk.TclError:
            return
        self._trend_draw_pending = True
        self.root.after(120, self._draw_trend)

    def _draw_trend(self) -> None:
        self._trend_draw_pending = False
        if not hasattr(self, "trend_canvas"):
            return
        try:
            if not self.trend_canvas.winfo_exists():
                return
        except tk.TclError:
            return
        canvas = self.trend_canvas
        canvas.delete("all")
        width = max(240, canvas.winfo_width())
        height = max(120, canvas.winfo_height())
        left, right, top, bottom = 38, width - 10, 12, height - 25
        plot_height = max(30, bottom - top)
        plot_width = max(60, right - left)

        # Градиент занимает всю поверхность графика, а не узкую полосу по
        # краям. Скруглённая альфа-маска сохраняет мягкие внутренние углы.
        if self._trend_bg_size != (width, height):
            bg_scale = 2
            bg_size = (width * bg_scale, height * bg_scale)
            base = np.array([17, 23, 45], dtype=np.float32)
            violet = np.array([83, 65, 178], dtype=np.float32)
            teal = np.array([22, 130, 145], dtype=np.float32)
            gx = np.linspace(0.0, 1.0, bg_size[0], dtype=np.float32)[None, :, None]
            gy = np.linspace(0.0, 1.0, bg_size[1], dtype=np.float32)[:, None, None]
            # Цвет плавно проходит из фиолетового слева сверху в бирюзовый
            # справа снизу; смесь заметна по всей площади, но сетка остаётся читаемой.
            blend = np.clip(gx * .58 + gy * .42, 0.0, 1.0)
            gradient = violet * (1.0 - blend) + teal * blend
            strength = (.26 + .06 * np.sin((gx[..., 0] + gy[..., 0]) * math.pi))[..., None]
            bg_pixels = base[None, None, :] * (1.0 - strength) + gradient * strength
            bg_rgba = np.dstack((np.broadcast_to(bg_pixels, (bg_size[1], bg_size[0], 3)),
                                 np.full((bg_size[1], bg_size[0]), 255, dtype=np.uint8)))
            bg_image = Image.fromarray(np.clip(bg_rgba, 0, 255).astype(np.uint8), mode="RGBA")
            bg_mask = Image.new("L", bg_size, 0)
            ImageDraw.Draw(bg_mask).rounded_rectangle(
                (0, 0, bg_size[0] - 1, bg_size[1] - 1), radius=14 * bg_scale, fill=255)
            bg_image.putalpha(bg_mask)
            self._trend_bg_photo = ImageTk.PhotoImage(
                bg_image.resize((width, height), Image.Resampling.LANCZOS))
            self._trend_bg_size = (width, height)
        canvas.create_image(0, 0, image=self._trend_bg_photo, anchor="nw",
                            tags="trend-background")
        period = self._trend_period_seconds
        now = time.monotonic()
        for tick in (0, 25, 50, 75, 100):
            y = bottom - plot_height * tick / 100
            canvas.create_line(left, y, right, y, fill="#35415f", dash=(1, 6), width=1)

        visible_samples = [(timestamp, value) for timestamp, value in self._trend_samples
                           if timestamp >= now - period]
        raw_segments: list[list[tuple[float, float]]] = []
        current: list[tuple[float, float]] = []
        for timestamp, value in visible_samples:
            if value is None:
                if current:
                    raw_segments.append(current)
                    current = []
                continue
            elapsed = max(0.0, min(float(period), timestamp - (now - period)))
            x = left + plot_width * elapsed / period
            y = bottom - plot_height * max(0.0, min(100.0, float(value))) / 100.0
            current.append((x, y))
        if current:
            raw_segments.append(current)

        # Градиентная область и неоновый spline рисуются в одном кэшируемом
        # Pillow-слое только по данным сессии; пустое состояние остаётся честным.
        if raw_segments:
            scale = 2
            layer_size = (width * scale, height * scale)
            overlay = Image.new("RGBA", layer_size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            xs = np.arange(layer_size[0], dtype=np.float32)[None, :]
            color_mix = np.clip(xs / max(1, layer_size[0] - 1), 0, 1)[..., None]
            violet = np.array([135, 106, 255], dtype=np.float32)
            cyan = np.array([57, 218, 229], dtype=np.float32)
            rgb = np.broadcast_to(violet * (1 - color_mix) + cyan * color_mix,
                                  (layer_size[1], layer_size[0], 3)).astype(np.uint8)
            for segment_index, raw in enumerate(raw_segments):
                smooth = self._smooth_spline(raw, steps=8)
                scaled = [(round(x * scale), round(y * scale)) for x, y in smooth]
                start_scaled = (round(raw[0][0] * scale), round(bottom * scale))
                end_scaled = (round(raw[-1][0] * scale), round(bottom * scale))
                if len(scaled) >= 2:
                    area_mask = Image.new("L", layer_size, 0)
                    ImageDraw.Draw(area_mask).polygon([*scaled, end_scaled, start_scaled], fill=255)
                    vertical_alpha = np.clip(
                        92 * (1 - np.arange(layer_size[1], dtype=np.float32)[:, None]
                              / max(1, bottom * scale - top * scale)) ** 1.35,
                        0, 92)
                    alpha = np.asarray(area_mask, dtype=np.float32) * vertical_alpha / 255.0
                    rgba = np.dstack((rgb, np.clip(alpha, 0, 255).astype(np.uint8)))
                    fill_layer = Image.fromarray(rgba, mode="RGBA")
                    overlay = Image.alpha_composite(overlay, fill_layer)
                    overlay_draw = ImageDraw.Draw(overlay)

                    line_mask = Image.new("L", layer_size, 0)
                    ImageDraw.Draw(line_mask).line(scaled, fill=255, width=5 * scale, joint="curve")
                    glow_mask = line_mask.filter(ImageFilter.GaussianBlur(5 * scale))
                    glow = Image.new("RGBA", layer_size, (131, 105, 255, 0))
                    glow.putalpha(glow_mask.point(lambda alpha_value: round(alpha_value * .76)))
                    overlay = Image.alpha_composite(overlay, glow)
                    overlay_draw = ImageDraw.Draw(overlay)
                    for point_index, (start_point, end_point) in enumerate(zip(scaled, scaled[1:])):
                        mix = point_index / max(1, len(scaled) - 2)
                        color = tuple(round(violet[channel] * (1 - mix) + cyan[channel] * mix)
                                      for channel in range(3))
                        overlay_draw.line((start_point, end_point), fill=(*color, 255),
                                          width=2 * scale, joint="curve")

                # Конечная точка показывается у свежего сигнала, разрывы None не соединяются.
                marker = raw[-1]
                if segment_index == len(raw_segments) - 1 and visible_samples[-1][1] is not None:
                    mx, my = marker[0] * scale, marker[1] * scale
                    halo = 8 * scale
                    overlay_draw.ellipse((mx - halo, my - halo, mx + halo, my + halo),
                                         fill=(107, 95, 255, 72))
                    overlay_draw.ellipse((mx - 4 * scale, my - 4 * scale,
                                          mx + 4 * scale, my + 4 * scale),
                                         fill="#53e7e2", outline="#ecffff", width=scale)
            self._trend_photo = ImageTk.PhotoImage(
                overlay.resize((width, height), Image.Resampling.LANCZOS))
            canvas.create_image(0, 0, image=self._trend_photo, anchor="nw", tags="trend-layer")

        for tick in (0, 25, 50, 75, 100):
            y = bottom - plot_height * tick / 100
            canvas.create_text(left - 9, y, text=str(tick), fill="#8c98b8",
                               font=("Segoe UI", 8), anchor="e")
        tick_count = 4
        for tick in range(tick_count + 1):
            elapsed = period * tick / tick_count
            x = left + plot_width * tick / tick_count
            if tick == tick_count:
                label = "Сейчас"
            elif period == 60:
                label = f"−{int(period - elapsed)} с"
            else:
                minutes, seconds = divmod(int(period - elapsed), 60)
                label = f"−{minutes}:{seconds:02d}"
            endpoint_anchor = "w" if tick == 0 else ("e" if tick == tick_count else "center")
            canvas.create_text(x, bottom + 13, text=label, fill="#9ba7c6",
                               font=("Segoe UI", 8), anchor=endpoint_anchor)
        if not raw_segments:
            canvas.create_oval(width / 2 - 4, (top + bottom) / 2 - 4,
                               width / 2 + 4, (top + bottom) / 2 + 4,
                               fill=TEAL, outline="")
            canvas.create_text(width / 2 + 14, (top + bottom) / 2,
                               text="Динамика появится после старта сессии",
                               fill=MUTED, font=("Segoe UI Semibold", 9), anchor="w")

    def _stop_worker(self, timeout: float = 0.0):
        if self.worker:
            stopped = self.worker.stop(timeout=timeout)
            if stopped:
                self.worker = None
                self._worker_stop_pending = False
                self._set_camera_status("Камера выключена", MUTED)
                self.camera_meta_label.configure(text="Поток остановлен")
                return True
            self._worker_stop_pending = True
            self._set_camera_status("Камера останавливается…", AMBER)
            self.camera_meta_label.configure(text="Освобождаем камеру…")
            self.start_btn.configure(state="disabled")
            return False
        self._worker_stop_pending = False
        self._set_camera_status("Камера выключена", MUTED)
        self.camera_meta_label.configure(text="Поток остановлен")
        return True

    def _poll_messages(self):
        while True:
            try:
                message = self.messages.get_nowait()
            except queue.Empty:
                break
            try:
                self._handle_message(message)
            except Exception as exc:
                self.state_label.configure(text="Ошибка", fg=RED)
                self.state_hint.configure(text=str(exc))
        self._refresh_live_preview()
        if self.worker and not self.worker.running:
            self.worker = None
            if self._worker_stop_pending:
                self._worker_stop_pending = False
                self._set_camera_status("Камера выключена", MUTED)
                self.camera_meta_label.configure(text="Поток остановлен")
                if not self.tracking and not self.calibrating:
                    self.start_btn.configure(state="normal" if self.profile else "disabled")
                    if not self.task_busy:
                        self.calibrate_btn.configure(state="normal")
            if self.calibrating:
                self._start_worker()
            elif self.profile and not self.tracking:
                self.start_btn.configure(state="normal")
                self.calibrate_btn.configure(state="normal")
        self._handle_task_messages()
        if self.tracking and self.latest_observation is not None:
            fresh = self._observation_is_fresh(self.latest_observation)
            usable = self.latest_observation.present and self.latest_observation.quality >= 0.45
            self.save_sample_btn.configure(state="normal" if fresh and usable else "disabled")
        self._expire_stale_frame()
        self.root.after(80, self._poll_messages)

    def _handle_message(self, message: WorkerMessage):
        if message.generation != self._worker_generation:
            return
        if message.kind == "ready":
            self._set_camera_status("Камера активна", GREEN)
            self.camera_meta_label.configure(text="Поток камеры подключён")
            return
        if message.kind == "camera_ready":
            self._set_camera_status("Камера открыта · загружаю анализ…", GREEN)
            self.camera_meta_label.configure(text="Ожидаем первый кадр")
            return
        if message.kind == "error":
            self._set_landmark_status()
            was_calibrating = self.calibrating
            self.calibrating = False
            self.calibration_progress.grid_remove()
            if was_calibrating:
                self.calibration_label.configure(text="Камера остановилась. Проверь подключение и попробуй ещё раз.")
                self.calibration_label.grid()
                self.root.after(6500, self.calibration_label.grid_remove)
            else:
                self.calibration_label.grid_remove()
            self.tracking = False
            self._set_rating_buttons("disabled")
            self.current_index = None
            self.latest_score = None
            self._draw_gauge(None)
            self.quality_label.configure(text="Нет активного кадра", fg=MUTED)
            self._draw_quality_bars(None)
            self._record_trend_sample(None)
            self._refresh_blind_rating_save_state()
            if self.session:
                self.session.finish(datetime.now().timestamp(), interrupted=True)
                self.session = None
            self._set_camera_status("Ошибка камеры или модели", RED)
            self.state_label.configure(text="Пауза", fg=AMBER)
            self.state_hint.configure(text=message.error or "Не удалось запустить анализ")
            self._clear_expression(message.error or "Нет активного предсказания")
            self.calibrate_btn.configure(state="normal")
            self.start_btn.configure(state="normal" if self.profile else "disabled")
            self.pause_btn.configure(state="disabled")
            self.finish_btn.configure(state="disabled")
            self.calibrate_btn.configure(state="normal")
            return
        if message.kind == "stopped":
            return
        if message.kind != "frame" or message.frame is None or message.observation is None:
            return
        raw_observation = message.observation
        observation = raw_observation
        if self.personal_adapter and raw_observation.expressions:
            try:
                observation = replace(raw_observation, expressions=self.personal_adapter.apply(raw_observation.expressions))
            except (TypeError, ValueError):
                observation = raw_observation
        if self.profile is not None and not self.calibrating:
            observation = apply_personal_sharpness_gate(observation, self.profile)
        if not self._observation_is_fresh(observation):
            return
        self._set_landmark_status(observation)
        au_status = (observation.action_units_status if observation.present
                     else "Нет лица для AU-оценки")
        self._show_action_units(observation.action_units, au_status)
        analysis_rate = self._record_analysis_rate(observation.monotonic_timestamp)
        frame_monotonic = (observation.monotonic_timestamp if observation.monotonic_timestamp is not None
                           else time.monotonic())
        self.latest_frame = message.frame
        self.latest_observation = observation
        frame_height, frame_width = message.frame.shape[:2]
        analysis_rate_text = f" · {analysis_rate:.0f} к/с" if analysis_rate is not None else ""
        self.camera_meta_label.configure(text=f"{frame_width}×{frame_height}{analysis_rate_text}")
        # Анализируемый кадр может уже устареть относительно предпросмотра.
        # Не перерисовываем экран назад, если поток камеры успел уйти вперёд.
        if frame_monotonic >= self._last_preview_monotonic:
            self._last_preview_monotonic = frame_monotonic
            self._draw_preview(message.frame, observation.face_box, observation.landmarks)
        processing_time = (observation.monotonic_timestamp if observation.monotonic_timestamp is not None
                           else time.monotonic())
        display_expressions = consensus_expression_probabilities(observation)
        stable = self.expression_smoother.update(processing_time, observation.quality, display_expressions)
        self._set_stable_label(
            f"{EXPRESSION_RU.get(stable.label, stable.label)}"
            if stable.ready and stable.label else (stable.reason or "Недостаточно данных")
        )
        if self.pattern_tracker is not None:
            pattern_summary = self.pattern_tracker.update(processing_time, observation.quality,
                                                          observation.patterns)
            self._show_patterns(pattern_summary)
        self.save_sample_btn.configure(state="normal" if self.tracking and observation.present and
                                       observation.quality >= 0.45 and observation.face_box and
                                       self._observation_is_fresh(observation)
                                       else "disabled")
        ranked = top_expressions(display_expressions) if display_expressions and observation.quality >= 0.45 else []
        if ranked:
            top_label, _top_probability = ranked[0]
            # Вероятность классификатора не является уверенностью модели,
            # поэтому на главном экране показываем только название мимики.
            self.expression_label.configure(text=f"Мимика: {EXPRESSION_RU.get(top_label, top_label)}")
            self.expression_alternatives.configure(
                text="Другие варианты: " + "   ·   ".join(
                    f"{EXPRESSION_RU.get(label, label)} {probability:.0%}" for label, probability in ranked[1:]
                )
            )
        else:
            self.expression_label.configure(text=observation.quality_reason or "Нет надёжного предсказания")
            self.expression_alternatives.configure(text="Альтернативы: —")
        lighting = ""
        if observation.mean_face_brightness is not None:
            brightness = observation.mean_face_brightness
            baseline_brightness = (self.profile.mean_brightness_median if self.profile is not None else None)
            if baseline_brightness is None:
                lighting = f" · яркость лица {brightness:.0f}/255"
            else:
                lighting_delta = brightness - baseline_brightness
                lighting = (f" · свет {brightness:.0f}/255 · Δ {lighting_delta:+.0f} к калибровке")
        exposure = (f" · пиксели на пределе {observation.saturated_pixel_fraction:.1%}"
                    if observation.saturated_pixel_fraction is not None else "")
        rate = f" · в анализе {analysis_rate:.1f} кадр/с" if analysis_rate is not None else ""
        self.quality_diagnostics_label.configure(
            text=f"Техническое качество кадра: {observation.quality:.0%}{lighting}{exposure}{rate}. "
                 "Это не точность оценки состояния.")
        if observation.quality >= 0.75:
            self.quality_label.configure(text="Высокое", fg=TEAL)
        elif observation.quality >= 0.45:
            self.quality_label.configure(text="Среднее", fg=AMBER)
        else:
            reason = observation.quality_reason or "Качество низкое"
            self.quality_label.configure(text=reason, fg=AMBER)
        self._draw_quality_bars(observation.quality)
        if self.calibrating:
            if observation.present and observation.quality >= 0.45 and observation.features:
                accepted = self.calibration_sampler.accept(processing_time)
                if self.calibration_sampler.continuity_broken:
                    self.calibration_observations.clear()
                if accepted:
                    self.calibration_observations.append(observation)
                count = min(50, self.calibration_sampler.sample_count)
                self.calibration_progress.configure(value=count)
                first_sample = self.calibration_sampler.first_sample_at
                elapsed = max(0.0, processing_time - first_sample) if first_sample is not None else 0.0
                self.calibration_label.configure(
                    text=f"Собираем личную базу… {count}/50 · {elapsed:.1f}/5 сек"
                )
                if self.calibration_sampler.ready(processing_time, 50):
                    self._finish_calibration()
            else:
                self.calibration_observations.clear()
                self.calibration_sampler.start(processing_time)
                self.calibration_progress.configure(value=0)
                reason = observation.quality_reason or "Поставь лицо по центру камеры"
                self.calibration_label.configure(text=f"{reason} · база сброшена, начни снова")
            return
        if not self.tracking or self.scorer is None or self.session is None:
            return
        score = self.scorer.update(observation, processing_time)
        self.latest_score = score
        self.session.update(score, processing_time, observation.timestamp)
        if not score.usable:
            self.current_index = None
            self._draw_gauge(None)
            self._record_trend_sample(None)
            self._set_rating_buttons("disabled")
            self._refresh_blind_rating_save_state()
            self.state_label.configure(text="Оценка\nприостановлена", fg=AMBER)
            self.state_hint.configure(text=score.reason or "Плохое качество наблюдения")
            self.quality_label.configure(text=f"{score.reason or 'Качество низкое'}{lighting}{exposure}{rate}")
            return
        self.current_index = score.index
        self._draw_gauge(score.index)
        self._record_trend_sample(score.index)
        self._set_rating_buttons("normal")
        self._refresh_blind_rating_save_state()
        self.current_expression = score.expression_score
        self.current_deviation = score.deviation_score
        self.expr_value.configure(text=f"{score.expression_score:.0f}")
        self.dev_value.configure(text=f"{score.deviation_score:.0f}")
        state = self.event_tracker.state
        tense = state == TrackerState.TENSION
        self.state_label.configure(text="Устойчивое изменение лица" if tense else "Близко к личной базе",
                                   fg=AMBER if tense else GREEN)
        self.observation_message.configure(
            text="Изменение лица сохраняется." if tense else "Сигнал меняется плавно.")
        self.state_hint.configure(text="Изменение держится дольше заданного времени." if tense else
                                   "Сравнение идёт с короткой базой, которую ты только что создал.")
        if self.episode_open != tense:
            self.episode_open = tense
            self.refresh_journal()

    def _finish_calibration(self):
        self.calibrating = False
        try:
            self.profile = CalibrationProfile.from_observations(self.calibration_observations)
        except ValueError as exc:
            self.calibration_label.configure(text=str(exc))
            self.calibration_progress.grid_remove()
            self.calibration_label.grid()
            self.root.after(6500, lambda: self.calibration_label.grid_remove()
                            if not self.calibrating else None)
            self.state_hint.configure(text="Базу не удалось создать. Проверь кадр и попробуй снова.")
            self.calibrate_btn.configure(state="normal")
            return
        self.calibration_progress.configure(value=50)
        first_sample = self.calibration_sampler.first_sample_at
        last_sample = self.calibration_sampler.last_sample_at
        observed_span = max(0.0, last_sample - first_sample) if first_sample is not None and last_sample is not None else 0.0
        median_quality = median(float(item.quality) for item in self.calibration_observations)
        self.calibration_label.configure(
            text=(f"База готова · {len(self.calibration_observations)} кадров · "
                  f"{observed_span:.1f} с · медианное качество кадров {median_quality:.0%}. "
                  "Это короткий ориентир личной мимики и условий съёмки, не норма стресса.")
        )
        self.state_label.configure(text="Готово", fg=GREEN)
        self.state_hint.configure(
            text="Личная мимическая база. Качество кадров не означает точность оценки стресса."
        )
        self.calibration_progress.grid_remove()
        self.root.after(6500, lambda: self.calibration_label.grid_remove()
                        if not self.calibrating else None)
        self._show_patterns(None)
        self.calibrate_btn.configure(state="normal")
        self.start_btn.configure(state="normal")
        self.calibrate_btn.configure(state="normal")
        self._worker_generation += 1
        stopped = self._stop_worker()
        self._clear_expression("Калибровка завершена; камера остановлена")
        if not stopped:
            self.start_btn.configure(state="disabled")

    def _expire_stale_frame(self):
        observation = self.latest_observation
        if observation is None or self._observation_is_fresh(observation):
            return
        if self.tracking and self.session is not None:
            self.session.interrupt_observation_gap()
            self.episode_open = False
            self.refresh_journal()
        self.expression_smoother.reset()
        if self.pattern_tracker:
            self.pattern_tracker.reset()
        self._show_patterns(None)
        if self.scorer:
            self.scorer.reset()
        self.latest_frame = None
        self.latest_observation = None
        self._set_landmark_status()
        self.preview.configure(image="", text="НЕТ СВЕЖЕГО КАДРА\n\nКлючевые точки временно скрыты")
        self.photo = None
        self.current_index = self.current_expression = self.current_deviation = None
        self.latest_score = None
        self.save_sample_btn.configure(state="disabled")
        self._set_rating_buttons("disabled")
        self._refresh_blind_rating_save_state()
        self.expression_label.configure(text="Кадр устарел · анализ приостановлен")
        self.expression_alternatives.configure(text="Альтернативы: —")
        self._set_stable_label("Нет свежих кадров")
        self.quality_label.configure(text="Ожидание свежего кадра")
        self._draw_quality_bars(None)
        if self.tracking:
            self._draw_gauge(None)
            self._record_trend_sample(None)
            self.expr_value.configure(text="—")
            self.dev_value.configure(text="—")
            self.state_label.configure(text="Кадр устарел", fg=AMBER)
            self.state_hint.configure(text="Индекс приостановлен до получения нового качественного кадра")
        elif self.calibrating:
            self.calibration_label.configure(text="Кадр устарел; жду новый кадр камеры…")

    def _refresh_live_preview(self):
        """Обновляет изображение прямо из потока камеры, не ожидая AU-инференс."""
        # Во время короткого завершения потока worker ещё может хранить последний
        # кадр. Не рисуем его после остановки, иначе заглушка снова станет стоп-кадром.
        if self.worker is None or not (self.calibrating or self.tracking):
            return
        frame_result = self.worker.latest_preview
        if frame_result is None:
            return
        captured = frame_result.captured_monotonic
        if captured is None:
            captured = time.monotonic()
        if captured <= self._last_preview_monotonic:
            return
        self._last_preview_monotonic = captured
        self.latest_frame = frame_result.frame
        face_box = None
        landmarks = ()
        observation = self.latest_observation
        if (observation is not None and observation.present and observation.face_box and
                self._observation_is_fresh(observation, max_age=0.4)):
            face_box = observation.face_box
            observed_at = observation.monotonic_timestamp
            if observed_at is not None and 0 <= captured - observed_at <= 0.25:
                landmarks = observation.landmarks
        self._set_landmark_status(observation if landmarks else None)
        self._draw_preview(frame_result.frame, face_box, landmarks)

    @staticmethod
    def _observation_is_fresh(observation, max_age: float = LIVE_MAX_AGE_SECONDS) -> bool:
        return observation_is_fresh(
            observation.timestamp, time.time(), max_age,
            monotonic_timestamp=observation.monotonic_timestamp,
            monotonic_now=time.monotonic(),
        )

    def _draw_preview(self, frame, face_box, landmarks=()):
        display = prepare_preview_frame(frame, face_box, landmarks)
        width = max(320, self.preview.winfo_width() - 20)
        height = max(240, self.preview.winfo_height() - 20)
        source_height, source_width = display.shape[:2]
        scale = min(width / source_width, height / source_height, 1.0)
        target_width = max(1, round(source_width * scale))
        target_height = max(1, round(source_height * scale))
        if (target_width, target_height) != (source_width, source_height):
            # Сначала уменьшаем BGR кадр: это снижает объём данных для RGB-конверсии
            # и создания Tk-изображения, особенно для камер 1080p и выше.
            display = cv2.resize(display, (target_width, target_height),
                                 interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        # Маска сглаживает именно края видеокадра, а не только рамку внешней
        # карточки. Прозрачные углы показывают совпадающий фон панели.
        corner_radius = max(12, min(22, min(image.size) // 18))
        image = apply_rounded_mask(image, corner_radius)
        self.photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.photo, text="")

    def refresh_journal(self):
        if not hasattr(self, "journal"):
            return
        for item in self.journal.get_children():
            self.journal.delete(item)
        for session in self.store.list_sessions(100):
            duration = max(0, (session.ended_at or datetime.now().timestamp()) - session.started_at)
            state = "Прервана" if session.interrupted else ("Идёт" if session.ended_at is None else "Завершена")
            self.journal.insert("", "end", iid=str(session.session_id), values=(
                datetime.fromtimestamp(session.started_at).strftime("%d.%m.%Y · %H:%M"),
                self._format_duration(duration),
                ("Собирается" if session.ended_at is None else
                 (f"{min(100.0, 100.0 * session.usable_seconds / duration):.0f}% · "
                  f"{session.usable_seconds:.0f} с" if session.usable_seconds is not None and duration > 0
                  else ("0% · 0 с" if session.usable_seconds is not None else "—"))),
                session.episode_count, session.rating_count, state,
            ))
        self.refresh_self_rating_summary()

    def _show_selected_session(self, _event=None):
        selection = self.journal.selection()
        if not selection:
            return
        session_id = int(selection[0])
        session = next((item for item in self.store.list_sessions(100)
                        if item.session_id == session_id), None)
        episodes = self.store.list_episodes(session_id)
        ratings = self.store.list_self_ratings(session_id)
        lines = []
        if session is not None:
            duration = max(0.0, (session.ended_at or datetime.now().timestamp()) - session.started_at)
            if session.ended_at is None:
                lines.append("Техническое покрытие собирается и появится после завершения сессии.")
            elif session.usable_seconds is None:
                lines.append("Покрытие: не записывалось в этой версии журнала")
            else:
                coverage = min(100.0, 100.0 * session.usable_seconds / duration) if duration > 0 else 0.0
                lines.append(f"Техническое покрытие: {coverage:.0f}% времени · "
                             f"{session.usable_seconds:.1f} с пригодных наблюдений · "
                             f"{session.usable_observations or 0} кадров")
            lines.append("Это доля времени с непрерывными качественными кадрами, не точность индекса.")
        for episode in episodes:
            start = datetime.fromtimestamp(episode["started_at"]).strftime("%H:%M:%S")
            if episode["ended_at"] is not None:
                end = datetime.fromtimestamp(episode["ended_at"]).strftime("%H:%M:%S")
                duration_seconds = episode["duration"] or 0
            elif episode["interrupted"]:
                end = "прервана"
                duration_seconds = 0
            else:
                end = "идёт"
                duration_seconds = datetime.now().timestamp() - episode["started_at"]
            duration = self._format_duration(duration_seconds)
            entry_count = int(episode.get("entry_observations", 0) or 0)
            entry_span = float(episode.get("entry_observed_span", 0) or 0)
            entry_interval = episode.get("entry_mean_interval")
            entry_max_interval = episode.get("entry_max_interval")
            if entry_count:
                interval_text = (f"{float(entry_interval):.2f} с" if entry_interval is not None else "—")
                max_interval_text = (f"{float(entry_max_interval):.2f} с"
                                     if entry_max_interval is not None else "нет в старой записи")
                entry_evidence = (f"  Подтверждение входа: {entry_count} наблюдений за {entry_span:.1f} с · "
                                  f"средний интервал {interval_text} · макс. {max_interval_text}")
            else:
                entry_evidence = "  Подтверждение входа: данные для старых эпизодов недоступны"
            exit_count = int(episode.get("exit_observations", 0) or 0)
            exit_span = float(episode.get("exit_observed_span", 0) or 0)
            exit_interval = episode.get("exit_mean_interval")
            exit_max_interval = episode.get("exit_max_interval")
            lines.append(f"{start} — {end}   ·   {duration}   ·   пик {episode['peak_index']:.0f}   ·   "
                         f"качество {episode['average_quality']:.0%}")
            lines.append(entry_evidence)
            if exit_count:
                interval_text = f"{float(exit_interval):.2f} с" if exit_interval is not None else "—"
                max_interval_text = (f"{float(exit_max_interval):.2f} с"
                                     if exit_max_interval is not None else "нет в старой записи")
                lines.append(f"  Подтверждение возврата: {exit_count} наблюдений за {exit_span:.1f} с · "
                             f"средний интервал {interval_text} · макс. {max_interval_text}")
            elif episode["ended_at"] is not None and not episode["interrupted"]:
                lines.append("  Подтверждение возврата: данные старого эпизода недоступны")
        if ratings:
            lines.append("МОИ ОТМЕТКИ · ЛИЦЕВОЙ СИГНАЛ В ТОТ ЖЕ МОМЕНТ")
            for item in ratings:
                at = datetime.fromtimestamp(item["created_at"]).strftime("%H:%M:%S")
                mode = " · слепая" if item.get("rating_mode") == "blind" else ""
                lines.append(f"{at} · напряжение {item['rating']}/10 · мимический сигнал "
                             f"{item['face_index']:.0f}/100 · качество {item['face_quality']:.0%}{mode}")
        if not lines:
            self.episode_details.configure(text="В этой сессии пока нет данных.")
        else:
            self.episode_details.configure(text="ЭПИЗОДЫ И ЛИЧНЫЕ ОТМЕТКИ\n" + "\n".join(lines))

    def _rating_snapshot(self):
        score = self.latest_score
        observation = self.latest_observation
        if (not self.tracking or self.session is None or score is None or not score.usable or
                score.index is None or observation is None or not observation.present or
                observation.quality < 0.45 or not self._observation_is_fresh(observation)):
            return None
        return score, observation

    def _store_current_self_rating(self, rating: int, rating_mode: str = "visible") -> None:
        snapshot = self._rating_snapshot()
        if snapshot is None:
            raise ValueError("Нужен свежий качественный лицевой сигнал в активной сессии.")
        score, observation = snapshot
        self.store.record_self_rating(self.session.session_id, datetime.now().timestamp(),
                                      observation.timestamp, rating, score.index, observation.quality,
                                      rating_mode=rating_mode)
        self.self_rating_status.configure(
            text=(f"Отметка {rating}/10 сохранена рядом со свежим лицевым сигналом. "
                  "Несовпадение тоже можно анализировать."))
        self.refresh_journal()
        self.refresh_self_rating_summary()

    def save_self_rating(self):
        try:
            rating = int(self.self_rating_var.get())
        except (TypeError, ValueError):
            self.self_rating_status.configure(text="Выбери личную оценку от 0 до 10.")
            return
        try:
            self._store_current_self_rating(rating)
        except ValueError as exc:
            self.self_rating_status.configure(text=str(exc))

    def start_blind_rating(self):
        if self._blind_rating_dialog is not None:
            self._blind_rating_dialog.lift()
            return
        if self._rating_snapshot() is None:
            self.self_rating_status.configure(text="Нужен свежий качественный лицевой сигнал в активной сессии.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.withdraw()
        dialog.title("Личная оценка без подсказки")
        dialog.configure(bg=BG)
        dialog.resizable(False, False)
        dialog.geometry("430x250")
        ttk.Label(dialog, text="СНАЧАЛА ТВОЯ ОЦЕНКА", font=("Segoe UI Semibold", 12)).pack(
            anchor="w", padx=24, pady=(22, 6))
        ttk.Label(dialog,
                  text="Какое напряжение ты чувствуешь сейчас? Монитор и его показатели скрыты до сохранения.",
                  style="Muted.TLabel", wraplength=380, justify="left").pack(
                      anchor="w", padx=24, pady=(0, 16))
        ttk.Label(dialog, text="0 — совсем нет    ·    10 — максимум").pack(anchor="w", padx=24)
        rating_var = tk.StringVar(value="")
        self._blind_rating_var = rating_var
        rating_combo = ttk.Combobox(dialog, state="readonly", width=6,
                                     values=[str(value) for value in range(11)],
                                     textvariable=rating_var)
        rating_combo.pack(anchor="w", padx=24, pady=(6, 0))
        status = ttk.Label(dialog, text="", style="Muted.TLabel", wraplength=380)
        status.pack(anchor="w", padx=24, pady=(6, 0))
        buttons = ttk.Frame(dialog)
        buttons.pack(fill="x", side="bottom", padx=24, pady=18)

        def close_dialog():
            if not dialog.winfo_exists():
                return
            try:
                dialog.grab_release()
            except tk.TclError:
                pass
            dialog.destroy()
            self._blind_rating_dialog = None
            self._blind_rating_save_button = None
            self._blind_rating_var = None
            try:
                self.root.deiconify()
                self.root.lift()
            except tk.TclError:
                pass

        def save_blind_rating():
            try:
                rating = int(rating_var.get())
                self._store_current_self_rating(rating, rating_mode="blind")
            except (TypeError, ValueError) as exc:
                status.configure(text=str(exc) if str(exc) else "Выбери оценку от 0 до 10.")
                return
            close_dialog()

        cancel_button = ttk.Button(buttons, text="Отмена", command=close_dialog)
        cancel_button.pack(side="right")
        save_button = ttk.Button(buttons, text="Сохранить и показать монитор",
                                 command=save_blind_rating, state="disabled")
        save_button.pack(side="right", padx=(0, 8))
        self._blind_rating_save_button = save_button
        rating_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_blind_rating_save_state())
        dialog.bind("<Return>", lambda _event: save_blind_rating())
        dialog.protocol("WM_DELETE_WINDOW", close_dialog)
        dialog.bind("<Escape>", lambda _event: close_dialog())
        self.root.update_idletasks()
        x = max(0, (dialog.winfo_screenwidth() - 430) // 2)
        y = max(0, (dialog.winfo_screenheight() - 250) // 2)
        dialog.geometry(f"430x250+{x}+{y}")
        self._blind_rating_dialog = dialog
        self.root.withdraw()
        dialog.deiconify()
        dialog.lift()
        dialog.grab_set()
        rating_combo.focus_set()

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    def _show_patterns(self, summary):
        for key, label in self.pattern_rows.items():
            if self.profile is None or key not in self.profile.pattern_medians:
                text, color = "Нужна калибровка", MUTED
            elif summary is None or not summary.ready:
                text, color = "Ожидание качественных кадров", MUTED
            else:
                signal = summary.signals.get(key)
                if signal is None or signal.delta is None:
                    text, color = "Недостаточно данных", MUTED
                elif signal.above_baseline:
                    text, color = f"Устойчиво выше · +{signal.delta:.2f}", AMBER
                else:
                    text, color = f"Около личной базы · {signal.delta:+.2f}", MUTED
            label.configure(text=text, foreground=color)

    def _show_action_units(self, values: dict[str, float] | None, status: str | None = None):
        """Показывает FACS AU отдельно от личного индекса мимических изменений."""
        if status:
            self.au_status_label.configure(text=status)
        for key, label in self.au_rows.items():
            value = values.get(key) if values else None
            if value is None:
                label.configure(text="—", foreground=MUTED)
            else:
                label.configure(text=f"{value:.2f}", foreground=AMBER if value >= 0.5 else TEXT)

    def delete_history(self):
        if self.tracking or self.calibrating:
            messagebox.showinfo("Журнал занят", "Сначала заверши сессию или калибровку, затем удаляй историю.")
            return
        if messagebox.askyesno("Удалить журнал", "Удалить все сессии и эпизоды? Это действие нельзя отменить."):
            self.store.delete_all()
            self.refresh_journal()
            self.episode_details.configure(text="История удалена.")

    def show_privacy(self):
        messagebox.showinfo(
            "Данные и приватность",
            f"Кадры камеры анализируются локально и не сохраняются автоматически. Журнал хранится на этом компьютере:\n{self.store.path}\n\n"
            f"Отдельно сохраняются только вручную помеченные обрезки лица: {self.sample_store.root}\n"
            "Оценку напряжения 0–10 можно сохранить только вручную вместе с текущим лицевым индексом и качеством; видеокадр не записывается. "
            "Эти отметки удаляются вместе с историей сессий.\n"
            f"Сравнение и адаптер сохраняются в {self.report_directory}; ничего не отправляется наружу. "
            "Примеры и отчёты можно удалить кнопкой ниже. Модели загружаются из сети один раз. "
            "Калибровочная база хранится только в памяти до закрытия приложения. "
            "Индекс — экспериментальная оценка мимики, не медицинское измерение стресса.",
        )

    def refresh_sample_counts(self):
        counts = self.sample_store.counts()
        summary = "Примеров: " + " · ".join(
            f"{EXPRESSION_RU[label]} {count}" for label, count in counts.items() if count
        )
        self.sample_counts_label.configure(text=summary if any(counts.values()) else "Примеров пока нет")

    def save_labeled_sample(self):
        observation = self.latest_observation
        if (not self.tracking or self.latest_frame is None or observation is None or
                not observation.present or observation.quality < 0.45 or not observation.face_box or
                not self._observation_is_fresh(observation)):
            self.save_sample_btn.configure(state="disabled")
            messagebox.showinfo("Нужен качественный кадр", "Запусти сессию и дождись кадра с лицом хорошего качества.")
            return
        selected = self.sample_label_var.get()
        label = selected.rsplit("(", 1)[-1].rstrip(")")
        try:
            sample = self.sample_store.save_crop(
                self.latest_frame, observation.face_box, label, observation.quality,
                EMOTION_MODEL_VERSION, "mediapipe-mesh-bbox / BGR-JPEG92",
                self.session.capture_id if self.session and self.session.capture_id else "unknown-session",
            )
        except (ValueError, OSError, RuntimeError) as exc:
            messagebox.showerror("Не удалось сохранить пример", str(exc))
            return
        self.refresh_sample_counts()
        self.refresh_sample_table()
        messagebox.showinfo("Пример сохранён", f"{EXPRESSION_RU[label]} · качество {sample.quality:.0%}\n{sample.path}")

    def delete_labeled_samples(self):
        if self.task_busy:
            messagebox.showinfo("Идёт обработка", "Дождись завершения сравнения, затем удаляй локальные примеры.")
            return
        if self.tracking or self.calibrating:
            messagebox.showinfo("Сессия занята", "Приостанови камеру перед удалением образцов.")
            return
        if messagebox.askyesno("Удалить размеченные лица?", "Удалить все вручную сохранённые обрезки лица и их метки? Это нельзя отменить."):
            amount = self.sample_store.delete_all()
            self.personal_adapter = None
            for path in (self.adapter_path, self.benchmark_report_path, self.benchmark_report_path.with_suffix(".csv"),
                         self.personalization_report_path):
                path.unlink(missing_ok=True)
            self.refresh_sample_counts()
            self.refresh_accuracy()
            self._refresh_model_status()
            messagebox.showinfo("Локальные примеры удалены", f"Удалено примеров: {amount}")

    def close(self):
        self.calibrating = False
        self.tracking = False
        if self.session:
            self.session.finish(datetime.now().timestamp(), interrupted=True)
        self.task_cancel.set()
        if self.task_process is not None and self.task_process.poll() is None:
            self.task_process.terminate()
        self._stop_worker(timeout=3.0)
        self.sample_store.close()
        self.store.close()
        self.root.destroy()

    def _clear_expression(self, message: str = "Нет активного предсказания"):
        self.expression_smoother.reset()
        if self.pattern_tracker:
            self.pattern_tracker.reset()
        self._show_patterns(None)
        self._show_action_units(None, "Ожидание нового кадра")
        self.latest_frame = None
        self.latest_observation = None
        self._set_landmark_status()
        self.latest_score = None
        self.current_index = None
        self.current_expression = None
        self.current_deviation = None
        self._draw_gauge(None)
        if hasattr(self, "expr_value"):
            self.expr_value.configure(text="—")
        if hasattr(self, "dev_value"):
            self.dev_value.configure(text="—")
        self._draw_quality_bars(None)
        # Не оставляем последний кадр на экране после остановки: неподвижное
        # лицо выглядело как зависшее видео, хотя камера уже была выключена.
        prompt = ("КАМЕРА ОСТАНОВЛЕНА\n\nНажми «Начать», чтобы запустить отслеживание"
                  if self.profile is not None else
                  "КАМЕРА ОСТАНОВЛЕНА\n\nНажми «Создать базу», чтобы начать")
        self.preview.configure(image="", text=prompt)
        self.photo = None
        if hasattr(self, "save_sample_btn"):
            self.save_sample_btn.configure(state="disabled")
        self.expression_label.configure(text=message)
        self._set_stable_label("Недостаточно данных")
        self.expression_alternatives.configure(text="Альтернативы: —")
