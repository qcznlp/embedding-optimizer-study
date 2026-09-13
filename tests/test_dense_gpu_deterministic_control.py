from types import SimpleNamespace

import pytest

from scripts.audit_dense_gpu_deterministic_control import change_owned_model


def fixture(count=22, original=False):
    cls = type("ModernBertAttention", (), {})
    modules = [cls() for _ in range(count)]
    for module in modules:
        module.deterministic_flash_attn = original
    config = SimpleNamespace(deterministic_flash_attn=original)
    model = [SimpleNamespace(auto_model=SimpleNamespace(config=config, modules=lambda: modules))]
    return model, modules


def test_only_declared_owned_attention_flags_change():
    model, modules = fixture()
    result = change_owned_model(model)
    assert result == {"attention_modules": 22, "before": False, "after": True}
    assert all(m.deterministic_flash_attn for m in modules)


def test_reject_changed_default():
    model, _ = fixture(original=True)
    with pytest.raises(ValueError):
        change_owned_model(model)


@pytest.mark.parametrize("count", [0, 21, 23])
def test_reject_unreviewed_topology(count):
    model, _ = fixture(count=count)
    with pytest.raises(ValueError):
        change_owned_model(model)
