"""Immutable first-backward reference bytes, with no producer-directory fallback."""

import hashlib
import json
from importlib.resources import files

CASES = {
    "adamw": ("a", "00e0dab0a4aaf52a5783ffb6ae2d5399952d55f2055b65b033fb5c4a0baa400d"),
    "muon": ("b", "dda70c52285f3eb1025b990bd00a46dedf89e54c9a9e65d8fce29aa9f1aa7795"),
}
WARM = {
    "a": (
        "20beef1b0b017b962bea72864b492c6eae8d92817808a8db6b2a3e05656157db",
        "48a715ea3b4b2101db0f8f1a90eda3c15122df9ec42e007abc836ced6630e411",
        "e917995154ac012f3f1137318881cbdba82bd4e435fecb7f528da27513b2cd82",
        "1c3cbfba725ffe9fe7928c613a878e59769aae73c94c5d8503f0939d8a935ad3",
    ),
    "b": (
        "a552328ced4cb5a180b2658ea62c34d0889f8d62efd32c930e589379fdb47a5a",
        "4baa714c7388c6cf0eb72b1f14ff038715cad7d63a5b6e6bf2de993714886e78",
        "93b90e593dcc5c8fd7c6e99a29738ec1eeb30669a35c756a1d06c89bc24ee731",
        "e756a500e396ff843e4bc2affa2a82b9cedef046a4f7274776cce8c3a8b9dd59",
    ),
}
COLD = {
    "a": (
        "05716b51d04380d155b5996ad70c7b115613aa7b768946f63c83bf5035294a7c",
        "835dba07d74fb5545ab770434f1ec4c292a67a7535bb98b81766f81f2071ccb5",
        "06913ec3f000da27934a22ec220e488cbb86d146c514bd817d33858ffec3843c",
        "cd0f7add568ec14e4e0a760150ac3b23f07d61fc43b8fd0d47d4939a2ce50665",
    ),
    "b": (
        "9eb4dfc9ee6b26b2d95cbfcb805760a46f581c2666033c7e0d98ab8b75a33cc5",
        "10d586122e28b75b51f8e73de6c2374e818adbac902f0fa2ba2f601450beb2cb",
        "7633cdb3437eb5d5ef86f98eaef32d9448b1aa6ea92ae2c07ff1df31b4a4ce27",
        "57380fc6be3529fd76b47db49d5e2cc975e9040892af11292ddd8753fc094019",
    ),
}


def _read(name, expected):
    raw = files("embed_optim_restore").joinpath("references", name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Packaged restoration reference bytes differ")
    return json.loads(raw)


def load_reference(case, rank):
    if (
        not isinstance(case, str)
        or case not in CASES
        or type(rank) is not int
        or rank not in range(4)
    ):
        raise ValueError("Only the two declared cases and four integer ranks are supported")
    pool, component = CASES[case]
    cold = _read(f"{pool}-{rank}-cold.json", COLD[pool][rank])
    warm = _read(f"{pool}-{rank}-warm.json", WARM[pool][rank])
    if cold["step"] != 313 or warm["step"] != 313 or warm["optimizer_updates"] != 0:
        raise ValueError("Reference is not the declared zero-update boundary")
    return {
        "case": case,
        "rank": rank,
        "run_id": f"factorial-v3-adamw_state-{case}-seed314159",
        "component_sha256": component,
        "cold_sha256": COLD[pool][rank],
        "warm_sha256": WARM[pool][rank],
        "cold": cold,
        "warm": warm,
    }
