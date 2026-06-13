"""Shared UI theme constants and ttk style configuration."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
C_SIDEBAR       = "#1e2130"
C_SIDEBAR_ITEM  = "#252840"
C_ACCENT        = "#4f8ef7"
C_BTN_FG_OFF    = "#a0a4c0"

C_BG            = "#13152a"
C_BG_MID        = "#1a1d30"
C_BORDER        = "#2e3148"

C_FG            = "#c8cce8"
C_FG_DIM        = "#6b7299"
C_FG_PLACEHOLDER= "#4a4e6a"

C_BTN_BG        = "#252840"
C_BTN_HOVER     = "#2e3250"

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------
FONT_UI    = ("Segoe UI", 10)
FONT_BOLD  = ("Segoe UI", 10, "bold")
FONT_SMALL = ("Segoe UI", 9)
FONT_MONO  = ("Consolas", 9)


# ---------------------------------------------------------------------------
# ttk style application
# ---------------------------------------------------------------------------

def apply_main_theme(root: tk.Tk) -> None:
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TFrame", background=C_BG)
    style.configure("TLabel", background=C_BG, font=FONT_UI)
    _apply_notebook_style(style, "TNotebook")
    _apply_scrollbar_style(style, "TScrollbar")


def apply_plots_theme(window: tk.Toplevel) -> None:
    style = ttk.Style(window)
    style.configure("Plots.TFrame", background=C_BG)
    style.configure("Plots.TLabel", background=C_BG, foreground=C_FG)
    _apply_notebook_style(style, "Plots.TNotebook")
    _apply_scrollbar_style(style, "TScrollbar")


def _apply_notebook_style(style: ttk.Style, name: str) -> None:
    tab = name + ".Tab"
    style.configure(name,
        background=C_BG, borderwidth=0, tabmargins=[0, 0, 0, 0], relief="flat",
        bordercolor=C_BG, lightcolor=C_BG, darkcolor=C_BG,
    )
    style.configure(tab,
        background=C_BG_MID, foreground=C_FG_DIM,
        font=FONT_SMALL, padding=(10, 4), borderwidth=0, relief="flat",
        bordercolor=C_BG, lightcolor=C_BG_MID, darkcolor=C_BG_MID,
    )
    style.map(tab,
        background=[("selected", C_BG), ("active", "#222540")],
        foreground=[("selected", C_ACCENT)],
        borderwidth=[("selected", 0)],
        lightcolor=[("selected", C_BG)],
        darkcolor=[("selected", C_BG)],
        bordercolor=[("selected", C_BG)],
    )
    style.layout(name, [("TNotebook.client", {"sticky": "nswe"})])
    style.layout(tab, [
        ("TNotebook.tab", {"sticky": "nswe", "children": [
            ("TNotebook.padding", {"side": "top", "sticky": "nswe", "children": [
                ("TNotebook.label", {"side": "top", "sticky": ""}),
            ]}),
        ]}),
    ])


def _apply_scrollbar_style(style: ttk.Style, name: str) -> None:
    style.configure(name,
        background="#1e2235", troughcolor=C_BG,
        borderwidth=0, arrowsize=12, relief="flat",
    )
    style.map(name, background=[("active", C_BORDER)])


# ---------------------------------------------------------------------------
# Widget factories
# ---------------------------------------------------------------------------

def make_sidebar_button(parent: tk.Widget, text: str, command) -> tk.Button:
    return tk.Button(
        parent, text=text, anchor="w",
        relief="flat", bd=0, padx=10, pady=6,
        font=FONT_SMALL, cursor="hand2",
        bg=C_SIDEBAR_ITEM, fg=C_BTN_FG_OFF,
        activebackground=C_SIDEBAR, activeforeground="white",
        command=command,
    )


def make_toolbar_button(parent: tk.Widget, text: str, command) -> tk.Button:
    return tk.Button(
        parent, text=text, font=FONT_SMALL,
        relief="flat", bd=0,
        bg=C_BTN_BG, fg=C_FG,
        activebackground=C_BTN_HOVER, activeforeground="white",
        padx=14, pady=5, cursor="hand2",
        command=command,
    )


def make_filter_button(parent: tk.Widget, text: str, command) -> tk.Button:
    return tk.Button(
        parent, text=text,
        padx=8, pady=3, relief="flat", bd=0,
        bg=C_BTN_BG, fg=C_FG_DIM,
        activebackground=C_BTN_HOVER, activeforeground="white",
        font=FONT_SMALL, cursor="hand2",
        command=command,
    )


def set_filter_active(btn: tk.Button, active: bool) -> None:
    if active:
        btn.config(bg=C_ACCENT, fg="white", font=("Segoe UI", 9, "bold"))
    else:
        btn.config(bg=C_BTN_BG, fg=C_FG_DIM, font=FONT_SMALL)


def make_result_text(parent: tk.Widget, scrollbar: tk.Widget) -> tk.Text:
    return tk.Text(
        parent,
        wrap="word", padx=10, pady=10,
        relief="flat", borderwidth=0,
        bg=C_BG_MID, fg=C_FG,
        font=FONT_MONO,
        yscrollcommand=scrollbar.set,
        selectbackground=C_ACCENT, selectforeground="white",
        insertbackground=C_FG,
        spacing1=2, spacing3=2,
    )
