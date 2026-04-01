"""Build definitions for MUSA compilation."""

def if_musa(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with MUSA."""
    return select({
        "@local_config_musa//:using_musa": if_true,
        "//conditions:default": if_false,
    })

def musa_default_copts():
    return if_musa(["-x", "musa"])
