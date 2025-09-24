from __future__ import annotations


class TemplateError(RuntimeError):
    """Base exception for the MuJoCo template."""


class NameLookupError(TemplateError):
    """Raised when a named entity cannot be resolved inside a model."""


class CompatibilityError(TemplateError):
    """Raised when controller/model compatibility checks fail."""


class LinearizationError(TemplateError):
    """Raised when linearization cannot be performed."""


class ConfigError(TemplateError):
    """Raised when template configuration is invalid."""


__all__ = [
    "TemplateError",
    "NameLookupError",
    "CompatibilityError",
    "LinearizationError",
    "ConfigError",
]
