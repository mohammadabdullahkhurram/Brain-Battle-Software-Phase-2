# labels_config.py  --  Brain Battle label directory
# Default labels are defined here.
# Any changes made in-app are saved to labels_data.json in this folder.
# Delete labels_data.json to reset to these defaults.

import os, json, sys

_DEFAULT_LABELS = [
    ('ROUND_START',      '#22c55e'),
    ('ROUND_END',        '#ef4444'),
    ('PUNCH_JAB',        '#3b82f6'),
    ('PUNCH_CROSS',      '#60a5fa'),
    ('PUNCH_HOOK',       '#818cf8'),
    ('PUNCH_UPPERCUT',   '#a78bfa'),
    ('CLINCH',           '#f59e0b'),
    ('KNOCKDOWN',        '#ef4444'),
    ('CORNER_CUT',       '#6b7280'),
    ('REF_STOPPAGE',     '#f97316'),
    ('DOMINANT_BOXER_A', '#06b6d4'),
    ('DOMINANT_BOXER_B', '#ec4899'),
]

_COLOURS = [
    '#e8ff00', '#06b6d4', '#34d399', '#f97316',
    '#a78bfa', '#f43f5e', '#38bdf8', '#fb923c',
    '#4ade80', '#c084fc', '#fbbf24', '#67e8f9',
]


def _data_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'labels_data.json')


def _load():
    p = _data_path()
    if os.path.exists(p):
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [tuple(x) for x in data]
    return list(_DEFAULT_LABELS)


# LABELS is the live list — loaded from JSON if it exists, else defaults
LABELS = _load()


def next_colour(existing):
    used = {c for _, c in existing}
    for c in _COLOURS:
        if c not in used:
            return c
    return _COLOURS[len(existing) % len(_COLOURS)]


def save(labels):
    mod = sys.modules.get('labels_config')
    if mod:
        mod.LABELS = list(labels)
    with open(_data_path(), 'w', encoding='utf-8') as f:
        json.dump(list(labels), f, indent=2)
