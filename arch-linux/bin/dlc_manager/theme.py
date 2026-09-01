#!/usr/bin/env python3
"""Design tokens and the single application stylesheet.

State is never communicated by colour alone: every coloured element in the UI
also carries a word (``LIVE``, ``STALLED``, ``NOT ASSIGNED``) and a leading
glyph, so the interface stays readable with a colour-vision deficiency or on a
badly calibrated panel.
"""

from __future__ import annotations


BACKGROUND = "#0b0f14"
SURFACE = "#111821"
SURFACE_RAISED = "#0e141c"
SURFACE_SUNKEN = "#070b10"
BORDER = "#263241"
BORDER_STRONG = "#354557"
TEXT = "#d7e0ea"
TEXT_MUTED = "#8b9aaa"
TEXT_FAINT = "#66757f"
ACCENT = "#7ee787"
LINK = "#79c0ff"


def readable_size(size_bytes: int) -> str:
    """Format byte count as human-readable size."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


# state -> (background, border, foreground, glyph)
STATE_TOKENS: dict[str, tuple[str, str, str, str]] = {
    "running": ("#12281b", "#3fb950", "#7ee787", "●"),
    "active": ("#12301e", "#3fb950", "#b7f5c4", "●"),
    "selected": ("#172719", "#7ee787", "#d7ffe0", "◆"),
    "working": ("#2c2412", "#d29922", "#e3b341", "◐"),
    "switching": ("#2c2412", "#d29922", "#e3b341", "◐"),
    "stale": ("#2c2412", "#d29922", "#e3b341", "▲"),
    "stopped": ("#1a212a", "#455464", "#aab8c5", "■"),
    "ready": ("#131d29", "#355777", "#9ecbff", "○"),
    "failed": ("#32191c", "#f85149", "#ff7b72", "▲"),
    "error": ("#32191c", "#f85149", "#ff7b72", "▲"),
    "unavailable": ("#0b0f14", "#2c3440", "#586573", "—"),
    "off": ("#0b0f14", "#2c3440", "#586573", "—"),
    "unknown": ("#1a212a", "#455464", "#aab8c5", "?"),
    "waiting": ("#131d29", "#355777", "#9ecbff", "○"),
    "live": ("#12281b", "#3fb950", "#7ee787", "●"),
    "stalled": ("#2c2412", "#d29922", "#e3b341", "▲"),
    "critical": ("#32191c", "#f85149", "#ff7b72", "▲"),
    "warning": ("#2c2412", "#d29922", "#e3b341", "▲"),
    "info": ("#131d29", "#355777", "#9ecbff", "○"),
}


def state_tokens(state: str) -> tuple[str, str, str, str]:
    return STATE_TOKENS.get(state, STATE_TOKENS["unknown"])


def state_glyph(state: str) -> str:
    return state_tokens(state)[3]


def stylesheet() -> str:
    """Return the application stylesheet.

    Focus rings are explicit on every interactive class: keyboard navigation
    has to stay visible, and Qt's default focus rectangle disappears against a
    dark palette.
    """
    return f"""
    QMainWindow, QWidget {{ background: {BACKGROUND}; color: {TEXT}; }}
    QToolTip {{ background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER_STRONG};
        padding: 5px; }}

    QFrame#appHeader {{ background: #0d141c; border-bottom: 1px solid {BORDER}; }}
    QLabel#appTitle {{ color: {ACCENT}; font-size: 16px; font-weight: 700;
        letter-spacing: 1px; }}
    QLabel#appSubtitle {{ color: {TEXT_FAINT}; font: 9px monospace; letter-spacing: 1px; }}
    QFrame#routeRibbon {{ background: {SURFACE_RAISED}; border: 1px solid {BORDER};
        border-radius: 6px; }}
    QLabel#ribbonLabel {{ color: {TEXT_FAINT}; font: bold 8px monospace;
        letter-spacing: 1px; }}
    QLabel#ribbonValue {{ color: {TEXT}; font: 11px monospace; }}
    QLabel#ribbonArrow {{ color: {LINK}; font: bold 13px monospace; }}

    QFrame#navigationRail {{ background: #090d12; border-right: 1px solid {BORDER}; }}
    QLabel#navigationTitle {{ color: {TEXT_FAINT}; font: bold 9px monospace;
        letter-spacing: 1px; padding: 2px 7px 6px 7px; }}
    QLabel#navigationNote {{ color: #5e6b78; font: 9px monospace; padding: 8px 7px; }}
    QPushButton#navButton {{ background: transparent; border: 1px solid transparent;
        border-radius: 6px; color: {TEXT_MUTED}; padding: 8px 10px; text-align: left;
        font: bold 10px monospace; }}
    QPushButton#navButton:hover {{ background: #121b25; color: {TEXT};
        border-color: {BORDER}; }}
    QPushButton#navButton:checked {{ background: #173522; color: {ACCENT};
        border: 1px solid #2f7542; }}
    QPushButton#navButton:focus {{ border: 1px solid {LINK}; }}

    QTabWidget#primaryTabs {{ background: {BACKGROUND}; border: 0; }}
    QTabWidget#primaryTabs::pane {{ border: 0; border-top: 1px solid {BORDER}; }}
    QTabBar::tab {{ background: #0d141c; color: {TEXT_MUTED};
        border: 1px solid {BORDER}; border-bottom: 0; padding: 10px 28px;
        min-width: 130px; font: bold 11px monospace; }}
    QTabBar::tab:hover {{ background: #14202b; color: {TEXT}; }}
    QTabBar::tab:selected {{ background: #173522; color: {ACCENT};
        border-color: #2f7542; }}
    QTabBar::tab:focus {{ border: 2px solid {LINK}; }}
    QScrollArea#workspaceScroll {{ border: 0; background: {BACKGROUND}; }}
    QLabel#pageTitle {{ color: {TEXT}; font-size: 20px; font-weight: 700; }}
    QLabel#pageSubtitle {{ color: #7d8b99; font: 10px monospace; }}

    QFrame#card {{ background: {SURFACE_RAISED}; border: 1px solid {BORDER};
        border-radius: 8px; }}
    QLabel#cardTitle {{ color: {LINK}; font-size: 12px; font-weight: 700;
        letter-spacing: 0.5px; }}
    QLabel#cardDetail {{ color: #7d8b99; font: 10px monospace; }}
    QFrame#cardSeparator {{ background: {BORDER}; max-height: 1px; min-height: 1px; }}

    QLabel#statusPill {{ border-radius: 4px; padding: 3px 8px;
        font: bold 9px monospace; }}
    QLabel#scopeBadge {{ background: #16202b; border: 1px solid {BORDER_STRONG};
        border-radius: 3px; color: #9ecbff; font: bold 8px monospace;
        padding: 2px 5px; }}
    QLabel#metricLabel {{ color: {TEXT_FAINT}; font: 9px monospace;
        letter-spacing: 0.5px; }}
    QLabel#metricValue {{ color: {TEXT}; font: bold 12px monospace; }}
    QLabel#metricHint {{ color: #6f7d8b; font: 9px monospace; }}
    QLabel#hintText {{ color: #7d8b99; font-weight: 400; font-family: monospace; }}
    QLabel#sectionLabel {{ color: #aab8c5; font: bold 9px monospace;
        letter-spacing: 1px; padding-top: 2px; }}
    QLabel#noteBox {{ background: #17130a; border: 1px solid #5b4a19;
        border-radius: 5px; color: #d9b84e; font: 10px monospace; padding: 7px; }}
    QLabel#infoBox {{ background: #0d1a26; border: 1px solid #1f4b73;
        border-radius: 5px; color: #9ecbff; font: 10px monospace; padding: 7px; }}
    QLabel#readout {{ background: {SURFACE_SUNKEN}; border: 1px solid {BORDER};
        border-radius: 5px; color: #b8c7d4; font: 10px monospace; padding: 9px; }}

    QFrame#videoPane {{ background: {SURFACE}; border: 1px solid {BORDER};
        border-radius: 8px; }}
    QLabel#paneTitle {{ color: {LINK}; font-size: 13px; font-weight: 700; }}
    QLabel#paneSubtitle {{ color: #7d8b99; font: 9px monospace; }}
    QLabel#videoSurface {{ background: #020406; border: 1px solid #202b36;
        border-radius: 4px; color: #536273; font-family: monospace; font-size: 14px; }}

    QToolButton#slotCard {{ background: {SURFACE_RAISED}; border: 1px solid {BORDER_STRONG};
        border-radius: 7px; color: {TEXT_MUTED}; padding: 10px 12px; text-align: left;
        font: 10px monospace; }}
    QToolButton#slotCard:hover {{ background: #162231; border-color: {LINK};
        color: {TEXT}; }}
    QToolButton#slotCard:focus {{ border: 2px solid {LINK}; }}
    QToolButton#slotCard[routeState="active"] {{ background: #12301e;
        border: 2px solid #3fb950; color: #b7f5c4; }}
    QToolButton#slotCard[routeState="selected"] {{ background: #172719;
        border: 2px solid #7ee787; color: #d7ffe0; }}
    QToolButton#slotCard[routeState="working"],
    QToolButton#slotCard[routeState="switching"],
    QToolButton#slotCard[routeState="stale"] {{ background: #2c2412;
        border: 2px solid #d29922; color: #e3b341; }}
    QToolButton#slotCard[routeState="error"] {{ background: #32191c;
        border: 2px solid #f85149; color: #ff7b72; }}
    QToolButton#slotCard[routeState="ready"] {{ border-color: #355777; color: #9ecbff; }}
    QToolButton#slotCard[routeState="unavailable"] {{ background: {BACKGROUND};
        border: 1px dashed #2c3440; color: #586573; }}
    QToolButton#slotCard:disabled {{ color: #66727d; }}

    QToolButton#historyItem {{ background: {SURFACE}; border: 1px solid {BORDER_STRONG};
        border-radius: 5px; color: {TEXT_MUTED}; padding: 3px; font: 9px monospace; }}
    QToolButton#historyItem:hover {{ background: #263647; border-color: {LINK}; }}
    QToolButton#historyItem:focus {{ border: 2px solid {LINK}; }}
    QToolButton#historyItem:checked {{ background: #173522; border: 2px solid {ACCENT};
        color: #d7ffe0; }}
    QLabel#sourcePreview {{ background: #05080c; border: 1px solid {BORDER_STRONG};
        border-radius: 6px; color: #617181; padding: 6px; }}
    QLabel#sourceName {{ color: {TEXT}; font-weight: 700; }}

    QFrame#topologyNode {{ background: {SURFACE_RAISED}; border: 1px solid {BORDER_STRONG};
        border-radius: 6px; }}
    QLabel#topologyLink {{ color: {LINK}; font: bold 12px monospace; }}
    QLabel#nodeTitle {{ color: {TEXT}; font: bold 10px monospace; }}

    QFrame#alertRow {{ background: {SURFACE_RAISED}; border: 1px solid {BORDER};
        border-left: 3px solid {BORDER_STRONG}; border-radius: 5px; }}
    QFrame#alertRow[severity="critical"] {{ border-left-color: #f85149; }}
    QFrame#alertRow[severity="warning"] {{ border-left-color: #d29922; }}
    QFrame#alertRow[severity="info"] {{ border-left-color: #355777; }}
    QLabel#alertComponent {{ color: {TEXT}; font: bold 10px monospace; }}
    QLabel#alertMessage {{ color: #c3ced9; font: 10px monospace; }}
    QLabel#alertAction {{ color: #9ecbff; font: 10px monospace; }}

    QGroupBox {{ border: 1px solid {BORDER}; border-radius: 6px; margin-top: 10px;
        padding-top: 8px; color: {LINK}; font-weight: 700; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 9px; padding: 0 4px; }}

    QComboBox {{ background: {SURFACE}; border: 1px solid {BORDER_STRONG};
        border-radius: 4px; color: {TEXT}; padding: 5px 6px; }}
    QComboBox:focus {{ border: 1px solid {LINK}; }}
    QComboBox:disabled {{ color: #66727d; background: #0d1218; }}
    QComboBox QAbstractItemView {{ background: {SURFACE}; color: {TEXT};
        selection-background-color: #1f6feb; border: 1px solid {BORDER_STRONG}; }}
    QCheckBox {{ color: {TEXT}; spacing: 7px; }}
    QCheckBox:disabled {{ color: #66727d; }}
    QCheckBox:focus {{ color: {LINK}; }}
    QRadioButton {{ color: {TEXT}; spacing: 7px; }}
    QRadioButton:focus {{ color: {LINK}; }}

    QLabel#rangeEndpoint {{ color: #667788; font: 9px monospace; }}
    QLabel#rangeValue {{ color: {ACCENT}; font: bold 10px monospace; }}
    QSlider::groove:horizontal {{ height: 5px; background: {BORDER}; border-radius: 2px; }}
    QSlider::sub-page:horizontal {{ background: #238636; border-radius: 2px; }}
    QSlider::handle:horizontal {{ background: {ACCENT}; border: 1px solid {BACKGROUND};
        width: 14px; margin: -5px 0; border-radius: 7px; }}
    QSlider:focus::handle:horizontal {{ border: 2px solid {LINK}; }}
    QSlider::handle:horizontal:disabled {{ background: #4a5561; }}

    QTextEdit#diagnostics {{ background: {SURFACE_SUNKEN}; border: 1px solid {BORDER};
        border-radius: 6px; color: #b8c7d4; selection-background-color: #1f6feb; }}

    QPushButton {{ background: #1c2733; border: 1px solid {BORDER_STRONG};
        border-radius: 5px; color: {TEXT}; padding: 7px 12px; }}
    QPushButton[compact="true"] {{ padding: 5px 10px; min-width: 52px; }}
    QPushButton[emphasis="primary"] {{ background: #1b3b25; border-color: #2f7542;
        color: #b7f5c4; font-weight: 700; }}
    QPushButton[emphasis="danger"] {{ background: #2a171a; border-color: #7a2b28;
        color: #ff9a94; }}
    QPushButton:hover {{ background: #263647; border-color: {LINK}; }}
    QPushButton:focus {{ border: 2px solid {LINK}; }}
    QPushButton:disabled {{ color: #66727d; background: #141a21;
        border-color: #232c36; }}

    QSplitter::handle {{ background: #18212b; }}
    QScrollBar:vertical {{ background: {BACKGROUND}; width: 11px; margin: 0; }}
    QScrollBar::handle:vertical {{ background: #2b3745; border-radius: 5px;
        min-height: 28px; }}
    QScrollBar::handle:vertical:hover {{ background: #3a4a5c; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar:horizontal {{ background: {BACKGROUND}; height: 11px; margin: 0; }}
    QScrollBar::handle:horizontal {{ background: #2b3745; border-radius: 5px;
        min-width: 28px; }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
    """
