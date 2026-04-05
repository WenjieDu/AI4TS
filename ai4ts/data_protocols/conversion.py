"""
Format conversion utilities for time series data.

Implements the full pipeline:
  Raw Input (CSV/numpy/DataFrame)
    → Validated DataFrame
    → Scaled numeric features
    → Windowed samples
    → Filled & masked
    → Anchor points inserted
    → Padded to fixed length
    → Output as numpy arrays (ready for tensor conversion)

And the reverse:
  Model output arrays → inverse-scaled → reconstructed DataFrame/numpy/CSV
"""

from __future__ import annotations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

from .constants import COL_NAME_SAMPLE_ID, COL_NAME_TIMESTAMP
from .enums import ScalerType, InputFormat, TaskType, MissingValueStrategy
from .schemas import (
    ProcessingConfig,
    TimeSeriesMetadata,
    ScalerParams,
    TensorDescriptor,
    ResultDescriptor,
)
from .validation import validate_csv_file, validate_numpy_array, validate_dataframe
from .utils import (
    separate_features,
    compute_sliding_stride,
    compute_n_samples,
    forward_fill,
    create_missing_mask,
    fill_and_get_mask,
    insert_anchor_points,
    pad_sequence,
    create_padding_mask,
    inject_artificial_missing,
    ensure_sample_id,
    ensure_timestamp,
)


class FeatureSeparator:
    """
    Identifies and categorizes columns in a time series DataFrame.

    This is a class-based wrapper around the separate_features utility
    that caches results and provides convenient accessors.
    """

    def __init__(self, df: pd.DataFrame):
        self.numeric, self.string, self.integer, self.classification_targets = (
            separate_features(df)
        )

    @property
    def all_features(self) -> list[str]:
        """All feature column names (numeric + string)."""
        return self.numeric + self.string

    @property
    def n_numeric_features(self) -> int:
        return len(self.numeric)


class TimeSeriesConverter:
    """
    Converts time series data between user-friendly formats and model-ready arrays.

    Handles the full lifecycle:
    1. Input: CSV file, numpy array, or pandas DataFrame
    2. Validation and metadata extraction
    3. Transformation to model-ready format (windowed, scaled, masked, padded)
    4. Reconstruction from model output back to user-friendly format

    Attributes
    ----------
    metadata : TimeSeriesMetadata
        Metadata from the validated input.
    config : ProcessingConfig
        Processing configuration.
    scaler : StandardScaler | MinMaxScaler | None
        Fitted scaler for numeric features.
    label_encoder : LabelEncoder | None
        Fitted encoder for classification targets.
    """

    def __init__(self, config: ProcessingConfig | None = None):
        self.config = config or ProcessingConfig()
        self.metadata: TimeSeriesMetadata | None = None
        self.scaler: StandardScaler | MinMaxScaler | None = None
        self.label_encoder: LabelEncoder | None = None
        self._df: pd.DataFrame | None = None

    def from_csv(self, path: str | Path) -> tuple[pd.DataFrame, TimeSeriesMetadata]:
        """Load, validate, and prepare a CSV file."""
        df = validate_csv_file(path)
        return self._prepare(df)

    def from_numpy(
        self,
        arr: np.ndarray,
        feature_names: list[str] | None = None,
        sample_ids: list | None = None,
    ) -> tuple[pd.DataFrame, TimeSeriesMetadata]:
        """Convert, validate, and prepare a numpy array.

        Accepts 2D (timesteps, features) or 3D (samples, timesteps, features).
        NaN values in the input are preserved throughout the pipeline when
        missing_value_strategy is NAN_KEEP.
        """
        df, _ = validate_numpy_array(arr, feature_names, sample_ids)
        return self._prepare(df)

    def from_dataframe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, TimeSeriesMetadata]:
        """Validate and prepare an existing DataFrame."""
        return self._prepare(df.copy())

    def _prepare(self, df: pd.DataFrame) -> tuple[pd.DataFrame, TimeSeriesMetadata]:
        """Common preparation: ensure columns, sort, validate, scale."""
        df = ensure_sample_id(df)
        df = ensure_timestamp(df)

        # Sort by SAMPLE_ID then TIMESTAMP
        if COL_NAME_TIMESTAMP in df.columns:
            try:
                df[COL_NAME_TIMESTAMP] = pd.to_datetime(df[COL_NAME_TIMESTAMP])
            except Exception:
                pass  # keep as-is if not parseable
        df = df.sort_values([COL_NAME_SAMPLE_ID, COL_NAME_TIMESTAMP]).reset_index(drop=True)

        # Validate
        self.metadata = validate_dataframe(df, self.config)
        self._df = df

        # Scale numeric features
        self._fit_scaler(df)

        # Encode classification targets
        if self.metadata.classification_target_names:
            self._fit_label_encoder(df)

        return df, self.metadata

    def _fit_scaler(self, df: pd.DataFrame) -> None:
        """Fit and apply scaler to numeric features.

        NaN values are excluded during fitting and preserved during transformation.
        The scaler is fitted only on observed (non-NaN) values, so missing data
        does not corrupt the learned statistics (mean, std, min, max).
        After transformation, positions that were NaN remain NaN.
        """
        if self.config.scaler_type == ScalerType.NONE:
            self.scaler = None
            return

        numeric_cols = self.metadata.numeric_feature_names
        if self.config.scaler_type == ScalerType.STANDARD:
            self.scaler = StandardScaler()
        elif self.config.scaler_type == ScalerType.MINMAX:
            self.scaler = MinMaxScaler()

        raw = df[numeric_cols].values.astype(np.float64)
        nan_mask = np.isnan(raw)
        has_nan = nan_mask.any()

        if has_nan:
            # Fit on non-NaN values only: fill NaN temporarily for fitting,
            # then restore NaN positions after transform
            col_means = np.nanmean(raw, axis=0)
            # Replace NaN with column means for fitting
            filled = raw.copy()
            for j in range(raw.shape[1]):
                col_nans = nan_mask[:, j]
                filled[col_nans, j] = col_means[j]
            self.scaler.fit(filled)
            transformed = self.scaler.transform(filled)
            # Restore NaN positions
            transformed[nan_mask] = np.nan
            df[numeric_cols] = transformed
        else:
            self.scaler.fit(raw)
            df[numeric_cols] = self.scaler.transform(raw)

    def _fit_label_encoder(self, df: pd.DataFrame) -> None:
        """Fit label encoder for classification targets."""
        target_col = self.metadata.classification_target_names[0]
        self.label_encoder = LabelEncoder()
        df[target_col] = self.label_encoder.fit_transform(df[target_col])

    def get_scaler_params(self) -> ScalerParams | None:
        """Extract serializable scaler parameters."""
        if self.scaler is None:
            return None

        params = ScalerParams(
            scaler_type=self.config.scaler_type,
            feature_names=self.metadata.numeric_feature_names,
        )
        if isinstance(self.scaler, StandardScaler):
            params.mean = self.scaler.mean_.tolist()
            params.std = self.scaler.scale_.tolist()
        elif isinstance(self.scaler, MinMaxScaler):
            params.data_min = self.scaler.data_min_.tolist()
            params.data_max = self.scaler.data_max_.tolist()
            params.scale = self.scaler.scale_.tolist()
        return params

    def to_tensor_samples(
        self,
        include_forecasting: bool = False,
        include_classification: bool = False,
        training_mode: bool = True,
    ) -> dict[str, Any]:
        """
        Convert the prepared DataFrame into model-ready numpy arrays.

        Applies windowing, missing value handling, anchor insertion, and padding.

        Parameters
        ----------
        include_forecasting : bool
            Whether to extract forecasting targets.
        include_classification : bool
            Whether to extract classification targets.
        training_mode : bool
            If True, inject artificial missing values and create indicating masks.

        Returns
        -------
        dict with keys:
            - "X": np.ndarray, shape (n_samples, max_input_steps, n_features)
            - "missing_mask": np.ndarray, same shape as X
            - "padding_mask": np.ndarray, shape (n_samples, max_input_steps, max_input_steps)
            - "indicating_mask": np.ndarray (only if training_mode=True)
            - "X_original": np.ndarray (only if training_mode=True)
            - "fore_target": np.ndarray (only if include_forecasting=True)
            - "fore_mask": np.ndarray (only if include_forecasting=True)
            - "clas_target": np.ndarray (only if include_classification=True)
            - "sample_actual_lengths": list[int]
            - "tensor_descriptor": TensorDescriptor
        """
        if self._df is None or self.metadata is None:
            raise RuntimeError("No data loaded. Call from_csv/from_numpy/from_dataframe first.")

        df = self._df
        config = self.config
        meta = self.metadata
        numeric_cols = meta.numeric_feature_names
        n_features = len(numeric_cols)
        max_steps = config.max_input_steps  # ts_context_len + n_anchor_points

        # Determine forecasting features
        fore_features = config.forecasting_features or numeric_cols
        n_fore = len(fore_features) if include_forecasting else 0

        # Group samples
        sample_groups = df.groupby(COL_NAME_SAMPLE_ID, sort=False)
        is_single = meta.is_single_sample

        # Compute windowing for single-sample
        if is_single:
            total_len = len(df)
            stride = config.sliding_stride or compute_sliding_stride(
                total_len, config.ts_context_len
            )
            n_samples = compute_n_samples(total_len, config.ts_context_len, stride)
        else:
            n_samples = meta.n_samples
            stride = 1  # not used for multi-sample

        # Pre-allocate arrays
        X_all = np.zeros((n_samples, max_steps, n_features), dtype=np.float32)
        mask_all = np.zeros_like(X_all)
        pad_mask_all = np.zeros((n_samples, max_steps, max_steps), dtype=bool)
        sample_lengths = []

        X_ori_all = None
        ind_mask_all = None
        if training_mode:
            X_ori_all = np.zeros_like(X_all)
            ind_mask_all = np.zeros_like(X_all)

        fore_target_all = None
        fore_mask_all = None
        if include_forecasting:
            fore_target_all = np.zeros(
                (n_samples, config.forecasting_horizon, n_fore), dtype=np.float32
            )
            fore_mask_all = np.zeros_like(fore_target_all)

        clas_target_all = None
        if include_classification and meta.classification_target_names:
            clas_target_all = np.zeros(n_samples, dtype=np.int64)

        # Process each sample
        rng = np.random.default_rng(42) if training_mode else None

        if is_single:
            data_array = df[numeric_cols].values.astype(np.float32)
            for i in range(n_samples):
                start = i * stride
                end = start + config.ts_context_len
                segment = data_array[start:min(end, len(data_array))]
                self._process_sample(
                    i, segment, n_features, max_steps, config, rng,
                    X_all, mask_all, pad_mask_all, sample_lengths,
                    X_ori_all, ind_mask_all, training_mode,
                )
                if include_forecasting:
                    fore_start = min(end, len(data_array))
                    fore_end = min(fore_start + config.forecasting_horizon, len(data_array))
                    fore_cols_idx = [numeric_cols.index(f) for f in fore_features]
                    if fore_start < len(data_array):
                        fore_seg = data_array[fore_start:fore_end, :][:, fore_cols_idx]
                        fore_len = fore_seg.shape[0]
                        fore_target_all[i, :fore_len] = fore_seg
                        fore_mask_all[i, :fore_len] = ~np.isnan(fore_seg)

                if include_classification and clas_target_all is not None:
                    target_col = meta.classification_target_names[0]
                    clas_target_all[i] = int(df[target_col].iloc[start])
        else:
            for i, (sample_id, group) in enumerate(sample_groups):
                segment = group[numeric_cols].values.astype(np.float32)
                self._process_sample(
                    i, segment, n_features, max_steps, config, rng,
                    X_all, mask_all, pad_mask_all, sample_lengths,
                    X_ori_all, ind_mask_all, training_mode,
                )
                if include_classification and clas_target_all is not None:
                    target_col = meta.classification_target_names[0]
                    clas_target_all[i] = int(group[target_col].iloc[0])

        # Build result
        result: dict[str, Any] = {
            "X": X_all,
            "missing_mask": mask_all,
            "padding_mask": pad_mask_all,
            "sample_actual_lengths": sample_lengths,
        }
        if training_mode:
            result["X_original"] = X_ori_all
            result["indicating_mask"] = ind_mask_all
        if include_forecasting:
            result["fore_target"] = fore_target_all
            result["fore_mask"] = fore_mask_all
        if include_classification and clas_target_all is not None:
            result["clas_target"] = clas_target_all

        result["tensor_descriptor"] = TensorDescriptor(
            shape=X_all.shape,
            dtype=config.cast_dtype,
            n_anchor_points=config.n_anchor_points,
            padding_lengths=[max_steps - l for l in sample_lengths],
            scaler_params=self.get_scaler_params(),
            masks=list(result.keys()),
            metadata=meta,
            config=config,
        )

        return result

    def _process_sample(
        self,
        idx: int,
        segment: np.ndarray,
        n_features: int,
        max_steps: int,
        config: ProcessingConfig,
        rng: np.random.Generator | None,
        X_all: np.ndarray,
        mask_all: np.ndarray,
        pad_mask_all: np.ndarray,
        sample_lengths: list,
        X_ori_all: np.ndarray | None,
        ind_mask_all: np.ndarray | None,
        training_mode: bool,
    ) -> None:
        """Process a single sample: fill/keep NaN, mask, anchor, pad.

        When missing_value_strategy is NAN_KEEP, NaN values are preserved in
        the output arrays so downstream models (e.g. PyPOTS) can handle them
        natively. The missing_mask still records which positions are NaN.
        """
        X_ori = segment.copy()
        keep_nan = config.missing_value_strategy == MissingValueStrategy.NAN_KEEP

        if training_mode and config.artificial_missing_rate > 0:
            X_with_missing = inject_artificial_missing(X_ori, config.artificial_missing_rate, rng)
        else:
            X_with_missing = X_ori.copy()

        # Create masks before any filling
        missing_mask = create_missing_mask(X_with_missing)
        ori_mask = create_missing_mask(X_ori)

        if keep_nan:
            # Preserve NaN in data; only fill anchor/padding positions
            X_filled = X_with_missing.copy()
        else:
            X_filled = forward_fill(X_with_missing)

        # Indicating mask (train only): marks artificially injected missing values
        indicating_mask = None
        if training_mode:
            indicating_mask = ori_mask - missing_mask
            indicating_mask = np.clip(indicating_mask, 0, 1)

        # Insert anchor points
        X_anchored = insert_anchor_points(X_filled, config.n_anchor_points)
        mask_anchored = insert_anchor_points(missing_mask, config.n_anchor_points)
        # Anchors are observed (mask=1)
        mask_anchored[:config.n_anchor_points] = 1.0

        actual_len = X_anchored.shape[0]

        # Pad to fixed length
        X_padded, valid = pad_sequence(X_anchored, max_steps)
        mask_padded, _ = pad_sequence(mask_anchored, max_steps, pad_value=0.0)

        # Padding mask (attention mask)
        pad_mask = create_padding_mask(min(actual_len, max_steps), max_steps)

        X_all[idx] = X_padded
        mask_all[idx] = mask_padded
        pad_mask_all[idx] = pad_mask
        sample_lengths.append(min(actual_len, max_steps))

        if training_mode and X_ori_all is not None and ind_mask_all is not None:
            if keep_nan:
                X_ori_processed = X_ori.copy()
            else:
                X_ori_processed = forward_fill(X_ori)
            X_ori_anchored = insert_anchor_points(X_ori_processed, config.n_anchor_points)
            X_ori_padded, _ = pad_sequence(X_ori_anchored, max_steps)
            X_ori_all[idx] = X_ori_padded

            ind_anchored = insert_anchor_points(
                indicating_mask if training_mode else np.zeros_like(missing_mask),
                config.n_anchor_points,
            )
            ind_anchored[:config.n_anchor_points] = 0.0
            ind_padded, _ = pad_sequence(ind_anchored, max_steps, pad_value=0.0)
            ind_mask_all[idx] = ind_padded

    # ──────────────────────────────────────────────
    # Reconstruction: model output → user format
    # ──────────────────────────────────────────────

    def tensor_to_dataframe(
        self,
        tensor: np.ndarray,
        strip_anchors: bool = True,
        strip_padding: bool = True,
        sample_lengths: list[int] | None = None,
        inverse_scale: bool = True,
        column_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Reconstruct a DataFrame from model output tensor.

        Parameters
        ----------
        tensor : np.ndarray
            3D array of shape (n_samples, max_steps, n_features) or
            2D array of shape (max_steps, n_features) for single sample.
        strip_anchors : bool
            Whether to remove anchor points from the beginning.
        strip_padding : bool
            Whether to remove padding from the end.
        sample_lengths : list[int] or None
            Actual length per sample (including anchors). Required if strip_padding=True.
        inverse_scale : bool
            Whether to apply inverse scaling.
        column_names : list[str] or None
            Column names for the output. Defaults to numeric feature names from metadata.

        Returns
        -------
        pd.DataFrame
            Reconstructed DataFrame.
        """
        if tensor.ndim == 2:
            tensor = tensor[np.newaxis, :]  # (1, steps, features)

        n_samples = tensor.shape[0]
        n_anchors = self.config.n_anchor_points if strip_anchors else 0
        names = column_names or (self.metadata.numeric_feature_names if self.metadata else None)
        if names is None:
            names = [f"feat_{i}" for i in range(tensor.shape[2])]

        frames = []
        for i in range(n_samples):
            sample = tensor[i]
            start = n_anchors
            end = sample_lengths[i] if (strip_padding and sample_lengths) else sample.shape[0]
            segment = sample[start:end]

            if inverse_scale and self.scaler is not None:
                # Preserve NaN positions through inverse transform
                seg_float = segment.astype(np.float64)
                nan_mask = np.isnan(seg_float)
                if nan_mask.any():
                    # Fill NaN temporarily for inverse_transform, then restore
                    seg_filled = seg_float.copy()
                    seg_filled[nan_mask] = 0.0
                    seg_inv = self.scaler.inverse_transform(seg_filled)
                    seg_inv[nan_mask] = np.nan
                    segment = seg_inv
                else:
                    segment = self.scaler.inverse_transform(seg_float)

            sample_df = pd.DataFrame(segment, columns=names)
            if n_samples > 1:
                sample_df[COL_NAME_SAMPLE_ID] = i
            frames.append(sample_df)

        return pd.concat(frames, ignore_index=True)

    def tensor_to_numpy(
        self,
        tensor: np.ndarray,
        strip_anchors: bool = True,
        strip_padding: bool = True,
        sample_lengths: list[int] | None = None,
        inverse_scale: bool = True,
    ) -> np.ndarray:
        """Reconstruct a numpy array from model output tensor."""
        df = self.tensor_to_dataframe(
            tensor, strip_anchors, strip_padding, sample_lengths, inverse_scale
        )
        numeric_cols = [c for c in df.columns if c != COL_NAME_SAMPLE_ID]
        return df[numeric_cols].values

    def tensor_to_csv(
        self,
        tensor: np.ndarray,
        path: str | Path,
        strip_anchors: bool = True,
        strip_padding: bool = True,
        sample_lengths: list[int] | None = None,
        inverse_scale: bool = True,
    ) -> str:
        """Reconstruct and save as CSV."""
        df = self.tensor_to_dataframe(
            tensor, strip_anchors, strip_padding, sample_lengths, inverse_scale
        )
        path = Path(path)
        df.to_csv(path, index=False)
        return str(path)
