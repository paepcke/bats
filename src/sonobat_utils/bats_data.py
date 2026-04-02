 # **********************************************************
 #
 # @Author: Andreas Paepcke
 # @Date:   2026-04-02 11:50:18
 # @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/sonobat_utils/bats_data.py
 # @Last Modified by:   Andreas Paepcke
 # @Last Modified time: 2026-04-02 11:54:22
 #
 # **********************************************************

# ---------------------------- Class BatsData -------------

import json
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from sklearn.preprocessing import RobustScaler

from logging_service import LoggingService


class BatsData:
    """
    Envelope that bundles the final merged DataFrame with the ancillary
    metadata needed to interpret and inverse-transform it.

    The main DataFrame (``self.df``) is a plain pandas DataFrame and should
    be used as such for all filtering, analysis, and ML work.  The envelope
    is only needed at the boundary — when reading from or writing to disk.

    The 'rec_site' column in ``df`` is a pandas Categorical whose categories
    are exactly the site names passed to the pipeline.  No integer site codes
    are ever exposed to the caller.

    Parquet is the sole serialization format.  All metadata (file_map,
    normalizer state) is stored as JSON in the Parquet schema metadata, so
    no sidecar files are required.

    Typical usage::

        # --- Produced by the pipeline ---
        bats = BatsData(df=df_final, file_map=encoder.id_to_path,
                        normalizer=normalizer, timestamp=ts)
        bats.to_parquet(dest_dir / 'measures.parquet')

        # --- Consumed downstream ---
        bats = BatsData.read_parquet('measures.parquet')
        df = bats.df                          # plain DataFrame from here on
        barn = df[df.rec_site == 'barn']
        bats.to_parquet(barn, 'barn_only.parquet')  # save subset

    :param df: Final merged, normalized DataFrame.
    :param file_map: Mapping from integer file_id to original file path string.
    :param normalizer: Fitted MeasureNormalizer instance.
    :param timestamp: ISO timestamp string from the processing run.
    """

    _META_KEY = b'bats_metadata'

    def __init__(
        self,
        df: pd.DataFrame,
        file_map: dict[int, str],
        normalizer: 'MeasureNormalizer',
        timestamp: str,
    ):
        self.df         = df
        self.file_map   = file_map
        self.normalizer = normalizer
        self.timestamp  = timestamp

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_parquet(self, df_or_path: 'pd.DataFrame | str | Path',
                   path: 'str | Path | None' = None) -> None:
        """
        Write a DataFrame (plus this envelope's metadata) to a Parquet file.

        Two call signatures are supported::

            bats.to_parquet('out.parquet')           # saves bats.df
            bats.to_parquet(modified_df, 'out.parquet')  # saves modified_df

        The 'rec_site' Categorical is preserved natively by PyArrow.
        All other metadata (file_map, normalizer state) is serialized as
        JSON in the Parquet schema metadata.

        :param df_or_path: Either a DataFrame to save, or the output path
                           (in which case ``self.df`` is saved).
        :param path: Output path; required when ``df_or_path`` is a DataFrame.
        :return: None
        """
        if isinstance(df_or_path, pd.DataFrame):
            df   = df_or_path
            dest = Path(path)
        else:
            df   = self.df
            dest = Path(df_or_path)

        meta_dict = {
            'timestamp' : self.timestamp,
            'file_map'  : {str(k): v for k, v in self.file_map.items()},
            'normalizer': self.normalizer.to_dict(),
        }
        meta_json = json.dumps(meta_dict)

        table    = pa.Table.from_pandas(df)
        existing = table.schema.metadata or {}
        new_meta = {**existing, self._META_KEY: meta_json.encode()}
        table    = table.replace_schema_metadata(new_meta)

        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dest)

    @classmethod
    def read_parquet(cls, path: 'str | Path') -> 'BatsData':
        """
        Load a Parquet file written by :meth:`to_parquet` and return a
        fully reconstructed BatsData envelope.

        The 'rec_site' column is restored as a pandas Categorical
        automatically by PyArrow.

        :param path: Path to the .parquet file.
        :return: BatsData with ``df``, ``file_map``, ``normalizer``, and
                 ``timestamp`` populated.
        :raises KeyError: If the file is missing the expected metadata key.
        """
        table    = pq.read_table(path)
        raw_meta = table.schema.metadata or {}

        if cls._META_KEY not in raw_meta:
            raise KeyError(
                f"Parquet file {path!r} has no '{cls._META_KEY.decode()}' "
                f"metadata key — was it written by BatsData.to_parquet()?"
            )

        meta_dict  = json.loads(raw_meta[cls._META_KEY].decode())
        file_map   = {int(k): v for k, v in meta_dict['file_map'].items()}
        normalizer = MeasureNormalizer.from_dict(meta_dict['normalizer'])
        timestamp  = meta_dict['timestamp']
        df         = table.to_pandas()

        return cls(df=df, file_map=file_map,
                   normalizer=normalizer, timestamp=timestamp)

# ----------------------------- Class MeasureNormalizer -----------------

NON_FEATURE_COLS = {'file_id', 'chirp_idx', 'cluster', 'TimeInFile', 'rec_site'}


class MeasureNormalizer:
    """
    Clean and normalize a SonoBat chirp measures DataFrame.

    Pipeline stages:
      1. Separate numeric feature columns from all others (non-numeric,
         and known non-feature columns such as identifiers and targets).
      2. Identify Tier 1 columns (outlier_factor > outlier_factor_thresh)
         and drop rows whose value in any Tier 1 column exceeds
         Q3 + fence_iqr_mult * IQR.
      3. Log-transform (log1p) columns with outlier_factor > log_transform_thresh.
      4. Apply RobustScaler to all numeric feature columns.
      5. Rejoin all non-feature columns (non-numeric, identifiers, targets)
         to the normalized result, aligned by index, before returning.

    The returned DataFrame from fit_transform() and transform() therefore
    contains both the normalized feature columns and all original non-feature
    columns, with outlier rows absent.

    The fitted normalizer can be saved to disk and reloaded, enabling:
      - Exact inverse-transform back to approximately original scale.
      - Applying the same scaling to new data without refitting.

    :param outlier_factor_thresh: outlier_factor above which a column is Tier 1.
    :param fence_iqr_mult: IQR multiplier for the per-row outlier fence.
    :param log_transform_thresh: outlier_factor above which log1p is applied.
    """

    def __init__(
        self,
        outlier_factor_thresh: float = 1000.0,
        fence_iqr_mult: float = 5.0,
        log_transform_thresh: float = 10.0,
    ):
        self.outlier_factor_thresh = outlier_factor_thresh
        self.fence_iqr_mult = fence_iqr_mult
        self.log_transform_thresh = log_transform_thresh

        self.log = LoggingService()

        self.numeric_cols_: list[str] = []
        self.feature_cols_: list[str] = []
        self.tier1_cols_: list[str] = []
        self.log_cols_: list[str] = []
        self.diagnostic_df_: pd.DataFrame | None = None
        self.scaler_: RobustScaler | None = None
        self.n_rows_before_: int = 0
        self.n_rows_after_: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df_measures: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full normalization pipeline on the concatenated measures DataFrame.

        Returns a DataFrame containing both the normalized numeric feature columns
        and all original non-feature columns (non-numeric, identifiers, targets),
        with outlier rows removed. The caller needs no post-processing to recover
        non-feature columns.

        :param df_measures: Raw concatenated SonoBat measures DataFrame.
        :return: Normalized feature columns joined with non-feature columns,
                 index-aligned, with outlier rows absent.
        """
        self.n_rows_before_ = len(df_measures)

        numeric_df = self._extract_numeric_features(df_measures)
        self.diagnostic_df_ = self._compute_diagnostics(numeric_df)

        self.tier1_cols_ = self._identify_tier(self.outlier_factor_thresh)
        self.log_cols_   = self._identify_tier(self.log_transform_thresh)

        self.log.info(
            f"Tier 1 columns (outlier_factor > {self.outlier_factor_thresh}): "
            f"{self.tier1_cols_}"
        )
        self.log.info(
            f"Log-transform columns (outlier_factor > {self.log_transform_thresh}): "
            f"{self.log_cols_}"
        )

        filtered_df = self._filter_outlier_rows(numeric_df)
        self.n_rows_after_ = len(filtered_df)
        self.log.info(
            f"Row filtering: {self.n_rows_before_} → {self.n_rows_after_} rows "
            f"({self.n_rows_before_ - self.n_rows_after_} removed, "
            f"{100*(self.n_rows_before_ - self.n_rows_after_)/self.n_rows_before_:.1f}%)"
        )

        log_transformed_df = self._apply_log_transform(filtered_df)
        normalized_df      = self._apply_robust_scaling(log_transformed_df)

        return self._rejoin_non_feature_cols(df_measures, normalized_df)

    def transform(self, df_new: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the already-fitted normalizer to new data without refitting.

        Intended for new SonoBat recordings that arrive after the initial fit,
        when preserving the original normalized space is desired. No row
        filtering is applied — the caller is responsible for pre-filtering.

        Returns a DataFrame containing both normalized feature columns and
        all original non-feature columns from df_new, index-aligned.

        :param df_new: Raw measures DataFrame with the same columns as
                       the original fit data.
        :return: Normalized feature columns joined with non-feature columns.
        :raises RuntimeError: If called before fit_transform.
        """
        self._assert_fitted()

        numeric_df = df_new.select_dtypes(include=['number'])
        numeric_df = numeric_df.drop(
            columns=[c for c in NON_FEATURE_COLS if c in numeric_df.columns]
        )
        numeric_df = numeric_df.reindex(columns=self.feature_cols_, fill_value=np.nan)

        log_transformed_df = self._apply_log_transform(numeric_df)
        scaled_array = self.scaler_.transform(log_transformed_df)
        normalized_df = pd.DataFrame(
            scaled_array, columns=self.feature_cols_, index=numeric_df.index
        )

        return self._rejoin_non_feature_cols(df_new, normalized_df)

    def inverse_transform(self, df_normalized: pd.DataFrame) -> pd.DataFrame:
        """
        Recover approximately original-scale values from a normalized DataFrame.

        Operates only on the numeric feature columns present in df_normalized,
        leaving any non-feature columns (if the full rejoined DataFrame is passed)
        untouched and passed through as-is.

        Applies inverse operations in reverse pipeline order:
          1. RobustScaler inverse_transform  (undo median/IQR scaling)
          2. expm1 on log-transformed columns (undo log1p)

        Note: rows dropped during outlier filtering are not recoverable.
        Recovered values may differ slightly from originals due to
        floating-point rounding.

        :param df_normalized: DataFrame in the normalized feature space,
                              as produced by fit_transform or transform.
                              May include non-feature columns.
        :return: DataFrame with feature columns in approximately original
                 measurement units, non-feature columns passed through unchanged.
        :raises RuntimeError: If called before fit_transform.
        """
        self._assert_fitted()

        feature_cols_present = [c for c in self.feature_cols_ if c in df_normalized.columns]
        non_feature_cols_present = [
            c for c in df_normalized.columns if c not in self.feature_cols_
        ]

        unscaled_array = self.scaler_.inverse_transform(
            df_normalized[feature_cols_present]
        )
        df_unscaled = pd.DataFrame(
            unscaled_array,
            columns=feature_cols_present,
            index=df_normalized.index
        )

        cols_to_exp = [c for c in self.log_cols_ if c in df_unscaled.columns]
        if cols_to_exp:
            df_unscaled[cols_to_exp] = np.expm1(df_unscaled[cols_to_exp])

        if non_feature_cols_present:
            df_unscaled = df_unscaled.join(df_normalized[non_feature_cols_present])

        return df_unscaled

    def save(self, path: str | Path) -> None:
        """
        Persist the fitted normalizer to disk as a standalone joblib file.

        This is an escape hatch for cases where the normalizer needs to be
        shared or inspected outside of a BatsData Parquet file.  For normal
        pipeline use, the normalizer travels inside the Parquet metadata via
        :meth:`to_dict` / :meth:`from_dict` and this method is not needed.

        :param path: Destination file path (conventionally .joblib).
        :return: None
        :raises RuntimeError: If called before fit_transform.
        """
        self._assert_fitted()
        joblib.dump(self, path)
        self.log.info(f"Normalizer saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> 'MeasureNormalizer':
        """
        Reload a previously saved MeasureNormalizer from disk.

        :param path: Path to a file saved by MeasureNormalizer.save().
        :return: Fully restored MeasureNormalizer instance.
        """
        log = LoggingService()
        normalizer = joblib.load(path)
        log.info(f"Normalizer loaded from {path}")
        return normalizer

    def report(self) -> None:
        """
        Print a human-readable summary of the normalization decisions.

        :return: None
        """
        if self.diagnostic_df_ is None:
            print("Call fit_transform() first.")
            return

        print(f"\n{'='*60}")
        print(f"MeasureNormalizer Report")
        print(f"{'='*60}")
        print(f"Rows before filtering : {self.n_rows_before_}")
        print(f"Rows after filtering  : {self.n_rows_after_}")
        print(f"Rows dropped          : {self.n_rows_before_ - self.n_rows_after_}")
        print(f"\nTier 1 (row fence applied, outlier_factor > "
              f"{self.outlier_factor_thresh}):")
        for col in self.tier1_cols_:
            of = self.diagnostic_df_.loc[col, 'outlier_factor']
            print(f"  {col:<30s}  outlier_factor={of:.1f}")
        print(f"\nLog-transformed (outlier_factor > {self.log_transform_thresh}):")
        for col in self.log_cols_:
            of = self.diagnostic_df_.loc[col, 'outlier_factor']
            print(f"  {col:<30s}  outlier_factor={of:.1f}")
        print(f"\nFeature columns after normalization ({len(self.feature_cols_)}):")
        print(f"  {self.feature_cols_}")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rejoin_non_feature_cols(
        self,
        df_original: pd.DataFrame,
        df_normalized: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Join non-feature columns from df_original back onto df_normalized.

        Uses df_normalized's index (the survivor set after outlier filtering)
        to slice df_original, so dropped rows are automatically absent.
        The join is left on df_normalized, guaranteeing no phantom rows appear.

        :param df_original: The raw input DataFrame passed to fit_transform
                            or transform, from which non-feature columns are drawn.
        :param df_normalized: The normalized feature-only DataFrame to join onto.
        :return: df_normalized with non-feature columns appended, index-aligned.
        """
        non_feature_cols = [
            c for c in df_original.columns if c not in self.feature_cols_
        ]
        if not non_feature_cols:
            return df_normalized

        return df_normalized.join(
            df_original.loc[df_normalized.index, non_feature_cols],
            how='left'
        )

    def to_dict(self) -> dict:
        """
        Serialize the fitted normalizer state to a JSON-safe dictionary.

        Stores the RobustScaler's center and scale arrays as plain lists so
        the result can be embedded directly in Parquet schema metadata
        without requiring pickle.

        :return: Dictionary suitable for ``json.dumps()``.
        :raises RuntimeError: If called before fit_transform.
        """
        self._assert_fitted()
        return {
            'outlier_factor_thresh': self.outlier_factor_thresh,
            'fence_iqr_mult'       : self.fence_iqr_mult,
            'log_transform_thresh' : self.log_transform_thresh,
            'feature_cols'         : self.feature_cols_,
            'numeric_cols'         : self.numeric_cols_,
            'tier1_cols'           : self.tier1_cols_,
            'log_cols'             : self.log_cols_,
            'scaler_center'        : self.scaler_.center_.tolist(),
            'scaler_scale'         : self.scaler_.scale_.tolist(),
            'n_rows_before'        : self.n_rows_before_,
            'n_rows_after'         : self.n_rows_after_,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'MeasureNormalizer':
        """
        Reconstruct a fitted MeasureNormalizer from a dictionary produced
        by :meth:`to_dict`.

        The restored instance is fully functional for :meth:`transform` and
        :meth:`inverse_transform`.  :meth:`report` will print a "not fitted"
        message because the diagnostic DataFrame is not serialized (it is a
        development aid, not needed for inference).

        :param d: Dictionary as produced by ``to_dict()``.
        :return: Fully restored MeasureNormalizer ready for ``transform()``
                 and ``inverse_transform()``.
        """
        obj = cls(
            outlier_factor_thresh = d['outlier_factor_thresh'],
            fence_iqr_mult        = d['fence_iqr_mult'],
            log_transform_thresh  = d['log_transform_thresh'],
        )
        obj.feature_cols_  = d['feature_cols']
        obj.numeric_cols_  = d['numeric_cols']
        obj.tier1_cols_    = d['tier1_cols']
        obj.log_cols_      = d['log_cols']
        obj.n_rows_before_ = d['n_rows_before']
        obj.n_rows_after_  = d['n_rows_after']

        scaler = RobustScaler()
        scaler.center_ = np.array(d['scaler_center'])
        scaler.scale_  = np.array(d['scaler_scale'])
        obj.scaler_    = scaler
        return obj

    def _assert_fitted(self) -> None:
        """
        Raise RuntimeError if the normalizer has not yet been fitted.

        :return: None
        :raises RuntimeError: If scaler_ is None.
        """
        if self.scaler_ is None:
            raise RuntimeError(
                "Normalizer has not been fitted. Call fit_transform() first."
            )

    def _extract_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep only numeric columns, then drop known non-feature columns.

        :param df: Raw measures DataFrame.
        :return: Numeric feature-only DataFrame.
        """
        numeric_df = df.select_dtypes(include=['number'])
        self.numeric_cols_ = list(numeric_df.columns)
        feature_df = numeric_df.drop(
            columns=[c for c in NON_FEATURE_COLS if c in numeric_df.columns]
        )
        self.feature_cols_ = list(feature_df.columns)
        return feature_df

    def _compute_diagnostics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-column diagnostics including outlier_factor.

        :param df: Numeric feature DataFrame.
        :return: Diagnostic DataFrame indexed by column name.
        """
        diag = df.agg(['mean', 'median', 'std', 'skew', 'kurt', 'min', 'max']).T
        diag['iqr'] = df.quantile(0.75) - df.quantile(0.25)
        diag['outlier_factor'] = (diag['max'] - diag['min']) / diag['iqr']
        return diag

    def _identify_tier(self, threshold: float) -> list[str]:
        """
        Return column names whose outlier_factor exceeds threshold.

        :param threshold: Minimum outlier_factor to include.
        :return: List of column names.
        """
        return list(
            self.diagnostic_df_[
                self.diagnostic_df_['outlier_factor'] > threshold
            ].index
        )

    def _filter_outlier_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows where any Tier 1 column value exceeds Q3 + fence_iqr_mult * IQR.
        Logs per-column drop counts before combining masks.

        :param df: Numeric feature DataFrame.
        :return: Filtered DataFrame with original index preserved.
        """
        if not self.tier1_cols_:
            return df

        mask = pd.Series(True, index=df.index)
        for col in self.tier1_cols_:
            q3  = df[col].quantile(0.75)
            iqr = q3 - df[col].quantile(0.25)
            fence = q3 + self.fence_iqr_mult * iqr
            col_mask = df[col] <= fence
            n_flagged = (~col_mask).sum()
            self.log.info(f"  {col}: {n_flagged} rows above fence ({fence:.3f})")
            mask &= col_mask

        return df[mask].copy()

    def _apply_log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply log1p to columns whose outlier_factor exceeded log_transform_thresh.

        :param df: Filtered numeric feature DataFrame.
        :return: DataFrame with log-transformed columns, index preserved.
        """
        df = df.copy()
        cols_present = [c for c in self.log_cols_ if c in df.columns]
        if cols_present:
            df[cols_present] = np.log1p(df[cols_present])
        return df

    def _apply_robust_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply RobustScaler (median/IQR) to all feature columns.

        :param df: Log-transformed numeric feature DataFrame.
        :return: Scaled DataFrame with same columns and index preserved.
        """
        self.scaler_ = RobustScaler()
        scaled_array = self.scaler_.fit_transform(df)
        return pd.DataFrame(scaled_array, columns=df.columns, index=df.index)
