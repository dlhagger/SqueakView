"""Compatibility imports for the operator session dialogs.

New code should import focused dialog classes from experiment_dialog,
subject_dialog, and session_launcher. This module preserves the historical
import surface used by the operator window and third-party tools.
"""

from squeakview.common.profiles import ProfileStore

from .dialog_style import (
    COMBO_POPUP_STYLE,
    DARK_DIALOG_STYLE,
    _dark_item_dialog,
    _meta_label,
    _size_button,
    apply_dark_combo_popups,
    center_window,
)
from .experiment_dialog import CreateExperimentDialog
from .session_launcher import SessionLauncherDialog
from .subject_dialog import CreateSubjectDialog

__all__ = [
    "COMBO_POPUP_STYLE",
    "DARK_DIALOG_STYLE",
    "CreateExperimentDialog",
    "CreateSubjectDialog",
    "ProfileStore",
    "SessionLauncherDialog",
    "_dark_item_dialog",
    "_meta_label",
    "_size_button",
    "apply_dark_combo_popups",
    "center_window",
]

