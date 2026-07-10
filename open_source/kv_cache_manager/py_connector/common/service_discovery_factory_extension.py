# -*- coding: utf-8 -*-
"""Provider registration hook for the open-source build."""


def register_service_discovery_providers(register_provider) -> None:
    """The open-source build does not install additional providers."""
    return None
