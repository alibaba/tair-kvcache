import pytest

from subscriber.kvcm.base import AbstractKvCacheManagerClient


def test_kvcm_manager_client_contract_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        AbstractKvCacheManagerClient()
