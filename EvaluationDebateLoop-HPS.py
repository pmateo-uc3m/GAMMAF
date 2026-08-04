"""
EvaluationDebateLoop-HPS.py  --  Hyperparameter-search variant of EvaluationDebateLoop.

This module provides:

* ``LiveDebateOrchestrationHPS`` -- a subclass of ``LiveDebateOrchestration`` that
  accepts an externally-built questions dataloader (the HPS pool loader) and an
  optional shared text-processor, so that the expensive dataset / model loading is
  performed only once and reused across every hyperparameter configuration.

* ``build_hps_pool_loader`` -- builds the fixed HPS evaluation-pool dataloader.
  The pool is a deterministic selection of ``hps_total_samples`` questions that
  are NOT part of the training data.  The selected dataset indices are persisted
  to ``index_pkl`` and reused on subsequent (interrupted) runs.

The original ``EvaluationDebateLoop.py`` is intentionally left untouched.
"""

import os
import pickle
import numpy as np
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from pydantic import SecretStr

from EvaluationDebateLoop import (
    LiveDebateOrchestration,
    load_class_from_path,
    load_class_by_tag_from_path,
)
from LoggingUtils import log_info, log_warn, log_error


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


# ---------------------------------------------------------------------------
#  A reproducible "identity" RNG used to build the full question list without
#  any random subsampling, so that previously-stored pool indices can be mapped
#  back to their exact questions.
# ---------------------------------------------------------------------------

class _IdentityRNG:
    """Mimics ``np.random.default_rng`` but returns the population untouched."""

    def choice(self, a, size=None, replace=False, axis=None, **kwargs):
        arr = np.asarray(list(a))
        if size is None:
            return arr[0] if len(arr) else arr
        n = int(size)
        if n <= len(arr):
            return arr[:n]
        return arr

    def __getattr__(self, _name):
        def _noop(*args, **kwargs):
            return None
        return _noop


def _build_full_question_list(loader_cls):
    """
    Instantiate *loader_cls* while patching ``np.random.default_rng`` so that
    ``load_questions`` returns every available question (in dataset order),
    without any random subsampling.
    """
    import numpy as _np
    _orig = _np.random.default_rng
    _np.random.default_rng = lambda seed=None: _IdentityRNG()
    try:
        loader = loader_cls(num_questions=10**12, random_seed=0, indexes=[])
    finally:
        _np.random.default_rng = _orig
    return loader


def _resolve_questions_loader_cls(live_cfg):
    dataset_tag = getattr(
        live_cfg,
        "questions_dataset_tag",
        getattr(live_cfg, "dataset_tag", None),
    )
    if dataset_tag:
        loader_cls = load_class_by_tag_from_path(live_cfg.questions_path, dataset_tag)
        log_info(
            f"Selected questions loader by dataset tag '{dataset_tag}': "
            f"{loader_cls.__name__}"
        )
    else:
        loader_cls = load_class_from_path(
            live_cfg.questions_path, live_cfg.questions_class_name
        )
        log_info(
            f"Selected questions loader by class name: {loader_cls.__name__}"
        )
    return loader_cls


def build_hps_pool_loader(
    live_cfg,
    train_indexes,
    hps_total_samples,
    hps_split_seed,
    index_pkl,
):
    """
    Build (or reload) the HPS pool dataloader.

    Parameters
    ----------
    live_cfg : object
        ``live_evaluation_config`` namespace.
    train_indexes : list[int]
        Dataset indices that belong to the model training data; never selected.
    hps_total_samples : int
        Size of the fixed HPS evaluation pool.
    hps_split_seed : int
        Seed controlling the one-time pool selection.
    index_pkl : str | Path
        Pickle file used to persist / reuse the selected pool indices.

    Returns
    -------
    pool_loader
        A fully initialized questions loader whose ``get_formatted_questions``
        returns the pool (of size ``hps_total_samples``) in deterministic order,
        and whose ``indexes`` attribute holds the selected dataset positions.
    pool_indices : list[int]
        The persisted dataset indices.
    """
    loader_cls = _resolve_questions_loader_cls(live_cfg)
    index_pkl_path = Path(index_pkl)
    train_set = set(train_indexes or [])

    if index_pkl_path.exists():
        with open(index_pkl_path, "rb") as f:
            stored = pickle.load(f)
        stored_indices = list(stored.get("indices", []))
        stored_params = stored.get("params", {})
        log_info(
            f"Reusing stored HPS pool indices from {index_pkl_path} "
            f"({len(stored_indices)} indices)"
        )
        if stored_params:
            log_info(f"Stored pool params: {stored_params}")

        if len(stored_indices) != hps_total_samples:
            raise ValueError(
                f"index_pkl contains {len(stored_indices)} indices but "
                f"hps_total_samples={hps_total_samples}. Delete {index_pkl_path} "
                f"or align the configuration."
            )

        # Reconstruct the pool from the stored dataset indices.
        full_loader = _build_full_question_list(loader_cls)
        pool_raw = [full_loader.questions[i] for i in stored_indices]
        full_loader.questions = pool_raw
        full_loader.indexes = list(stored_indices)
        full_loader.formatted_questions = full_loader.format_questions()

        leaked = train_set.intersection(stored_indices)
        if leaked:
            log_warn(
                f"Stored pool contains {len(leaked)} training indices -- "
                f"they will still be excluded from evaluation by the loader."
            )
        return full_loader, list(stored_indices)

    # First time: perform the deterministic pool selection.
    log_info(
        f"Selecting HPS pool of {hps_total_samples} questions "
        f"(seed={hps_split_seed}, excluding {len(train_set)} training indices)"
    )
    pool_loader = loader_cls(
        num_questions=hps_total_samples,
        random_seed=hps_split_seed,
        indexes=list(train_set),
    )
    pool_indices = [int(i) for i in list(pool_loader.indexes)]

    leaked = train_set.intersection(pool_indices)
    if leaked:
        raise RuntimeError(
            f"Pool selection leaked {len(leaked)} training indices. Aborting."
        )

    index_pkl_path.parent.mkdir(parents=True, exist_ok=True)
    params = {
        "hps_total_samples": hps_total_samples,
        "hps_split_seed": hps_split_seed,
        "train_indexes_count": len(train_set),
    }
    with open(index_pkl_path, "wb") as f:
        pickle.dump({"indices": pool_indices, "params": params}, f)
    log_info(f"Persisted pool indices to {index_pkl_path}")
    return pool_loader, pool_indices


# ---------------------------------------------------------------------------
#  HPS orchestration
# ---------------------------------------------------------------------------

class LiveDebateOrchestrationHPS(LiveDebateOrchestration):
    """
    Live-debate orchestration variant for hyperparameter search.

    Differences with respect to ``LiveDebateOrchestration``:

    * an externally-built dataloader (the HPS pool loader) is injected, so the
      dataset is loaded only once and the exact evaluation pool is fixed;
    * an optional shared ``text_processor`` may be injected to avoid reloading
      the embedding model for every configuration;
    * ``train_indexes`` are stored for downstream bookkeeping only (the pool
      loader already excludes them).
    """

    def __init__(
        self,
        config,
        dataloader,
        text_processor=None,
        train_indexes=None,
    ):
        # Replicate the relevant parent setup, injecting the dataloader.
        import random
        import threading
        import time as _time
        from datetime import datetime

        self.config = config
        self.python_seed = getattr(
            config, "python_seed", getattr(config, "questions_random_seed", 0)
        )
        self.numpy_seed = getattr(config, "numpy_seed", self.python_seed)
        self.answer_seed = getattr(config, "answer_seed", self.python_seed)
        random.seed(self.python_seed)
        np.random.seed(self.numpy_seed)
        self.timestamp = datetime.fromtimestamp(_time.time()).strftime("%Y%m%d%H%M%S")
        self._current_threshold = None
        self._model_predict_lock = threading.Lock()
        self.train_indexes = list(train_indexes or [])

        self.dataloader = dataloader
        self.prompts = self.dataloader.get_prompts()

        if text_processor is not None:
            self.text_processor = text_processor
        else:
            textProcessor = load_class_from_path(
                config.text_processor_path, config.text_processor_class_name
            )
            self.text_processor = textProcessor(device="cpu")

        self._model_name = _require_env("MODEL_NAME")
        self._base_url = _require_env("BASE_URL")
        self._api_key = SecretStr(_require_env("API_KEY"))

        self.llm_max_retries = getattr(config, "llm_max_retries", 3)
        self._llm_timeout = config.timeout

    # The inherited ``run_debate_with_defense`` / ``parse_stats_single_model``
    # methods are reused unchanged: they read ``self.config`` and
    # ``self.dataloader`` at call time, which is exactly what we need.