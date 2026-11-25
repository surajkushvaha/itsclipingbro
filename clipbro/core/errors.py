"""Exception hierarchy shared by all clipbro layers."""


class ClipbroError(Exception):
    """Base class for every clipbro error."""


class InvalidRange(ClipbroError, ValueError):
    """Raised when a time range is malformed: negative, empty or non-finite."""


class MediaError(ClipbroError):
    """Raised when media cannot be read, probed or decoded."""
