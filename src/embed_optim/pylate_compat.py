"""Narrow PyLate 1.6 compatibility shims for SentenceTransformers 5.x."""

from __future__ import annotations

import inspect
import json


def configure_pylate_compatibility():
    from pylate import models
    from sentence_transformers.util import load_file_path

    # ST 5 dispatches subclass loaders by this class marker.
    models.ColBERT.model_type = "ColBERT"
    # PyLate's encode path still calls the ST 4 private spelling.
    if not hasattr(models.ColBERT, "_text_length"):
        models.ColBERT._text_length = staticmethod(models.ColBERT._input_length)
    if not getattr(models.ColBERT, "_embed_optim_st5_compat", False):
        original_init = models.ColBERT.__init__
        signature = inspect.signature(original_init)
        pylate_keys = (
            "query_prefix",
            "document_prefix",
            "query_length",
            "document_length",
            "attend_to_expansion_tokens",
            "skiplist_words",
            "do_query_expansion",
        )

        def compatible_init(self, *args, **kwargs):
            bound = signature.bind_partial(self, *args, **kwargs)
            values = dict(bound.arguments)
            values.pop("self", None)
            model_name = values.get("model_name_or_path")
            saved_config = None
            if model_name:
                config_path = load_file_path(
                    model_name,
                    "config_sentence_transformers.json",
                    token=values.get("token"),
                    cache_folder=values.get("cache_folder"),
                    revision=values.get("revision"),
                    local_files_only=values.get("local_files_only", False),
                )
                if config_path:
                    with open(config_path) as handle:
                        saved_config = json.load(handle)
                    for key in pylate_keys:
                        if values.get(key) is None and key in saved_config:
                            values[key] = saved_config[key]
            original_init(self, **values)
            if saved_config is not None:
                self._model_config = saved_config

        models.ColBERT.__init__ = compatible_init
        models.ColBERT._embed_optim_st5_compat = True
    return models
