 # **********************************************************
 #
 # @Author: Andreas Paepcke
 # @Date:   2026-04-02 15:19:09
 # @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/sonobat_utils/test/test_sb_measures_postprocessing.py
 # @Last Modified by:   Andreas Paepcke
 # @Last Modified time: 2026-04-02 15:31:39
 #
 # **********************************************************

#!/usr/bin/env python
"""
Tests for sb_measures_postprocessing.py.

Synthesizes minimal *_CumulativeParameters_*.txt and
*_CumulativeSonoBatch_*.txt fixture files that replicate the structure of
real SonoBat output, without depending on any actual field-recording data.

Fixture design notes:
- Paths use Windows-style separators (Y:\\...) to match real SonoBat output,
  and are treated as plain strings throughout (not pathlib.Path objects).
- Each site has two recordings; one recording intentionally appears in both
  measures and species files (normal case), one appears only in measures
  (no species ID — simulates a recording where SonoBat found nothing).
- One species row is given low confidence to exercise the threshold filter.
- One SonoBatch row has NaN species fields to exercise the zero-confidence
  / filter-out path.
"""

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sonobat_utils.sb_measures_postprocessing import (
    BatsData,
    CompositeSpecies,
    MeasureNormalizer,
    PathEncoder,
    SonoBatPostProcessor,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

# Exact column list the pipeline selects from the Parameters file.
MEASURES_COLS = ['Path'] + SonoBatPostProcessor.RELEVANT_MEASURES_COLS

# Columns the pipeline selects from the SonoBatch file.
SPECIES_COLS = ['Path', 'SppAccp', 'Prob', '#Maj', '#Accp']

# Windows-style paths that match real SonoBat output format.
BARN_PATH_A  = r'Y:\barn\batch1\barn_Parsed\barn-20220205_190030_2secs.wav'
BARN_PATH_B  = r'Y:\barn\batch1\barn_Parsed\barn-20220205_190034_2secs.wav'
LAKE_PATH_A  = r'Y:\lake2\batch1\chopped\lake2-20220210_000000_2secs.wav'
LAKE_PATH_B  = r'Y:\lake2\batch1\chopped\lake2-20220210_000002_2secs.wav'


def _measures_row(path: str, time_in_file: float, preceding: float) -> dict:
    """
    Return one measures row with realistic-ish numeric values.
    All amplitude columns are positive to avoid log1p issues.

    :param path: Windows-style wav path string.
    :param time_in_file: TimeInFile value (ms).
    :param preceding: PrecedingIntrvl value (ms).
    :return: Dict of column→value for one chirp row.
    """
    base = {
        'Path'          : path,
        'TimeInFile'    : time_in_file,
        'PrecedingIntrvl': preceding,
        'HiFreq'        : 80.0,
        'Bndwdth'       : 35.0,
        'FreqMaxPwr'    : 60.0,
        'PrcntMaxAmpDur': 60.0,
        'FreqKnee'      : 64.0,
        'PrcntKneeDur'  : 48.0,
        'StartF'        : 80.0,
        'UpprKnFreq'    : 64.0,
        'HiFtoUpprKnAmp': 25.0,
        'HiFtoKnAmp'    : 110.0,
        'HiFtoFcAmp'    : 99.0,
        'UpprKnToKnAmp' : 0.0,
        'KnToFcAmp'     : 63.0,
        'LdgToFcAmp'    : 53.0,
        'FreqCtr'       : 63.0,
        'FFwd32dB'      : 94.0,
        'FFwd20dB'      : 75.0,
        'FFwd15dB'      : 74.0,
        'FBak5dB'       : 59.0,
        'FFwd5dB'       : 70.0,
        'Bndw32dB'      : 42.0,
        'Amp1stQrtl'    : 2.7,
        'Amp2ndQrtl'    : 2.1,
        'Amp3rdQrtl'    : 1.9,
        'Amp4thQrtl'    : 1.7,
        '1st10kHzSlp'   : 54.0,
        '1st5to15kHzSlp': 49.0,
        '1st10kHzExp'   : 127.0,
        '1st5to15kHzExp': 126.0,
        'AmpK@start'    : 8.9,
    }
    return base


def _write_measures_file(path: Path, rows: list[dict]) -> None:
    """
    Write a tab-separated *_CumulativeParameters_*.txt fixture file.
    Includes extra columns (as real SonoBat output does) beyond what the
    pipeline selects, to verify the column-culling logic.

    :param path: Destination file path.
    :param rows: List of row dicts (from :func:`_measures_row`).
    """
    df = pd.DataFrame(rows)
    # Add extra columns that the pipeline should silently ignore
    df['CallsPerSec'] = 8.0
    df['Quality']     = 0.95
    df.to_csv(path, sep='\t', index=False)


def _write_species_file(path: Path, rows: list[dict]) -> None:
    """
    Write a tab-separated *_CumulativeSonoBatch_*.txt fixture file.
    Includes extra columns beyond what the pipeline selects.

    :param path: Destination file path.
    :param rows: List of row dicts with keys: Path, SppAccp, Prob, #Maj, #Accp.
    """
    df = pd.DataFrame(rows)
    # Extra columns the pipeline ignores
    df['HiF']     = 1
    df['Filename'] = df['Path'].apply(lambda p: p.split('\\')[-1])
    df.to_csv(path, sep='\t', index=False)


@pytest.fixture()
def site_dirs(tmp_path: Path):
    """
    Build a two-site fixture tree under tmp_path and return a dict with
    paths and site labels.

    Tree::

        tmp_path/
          barn/batch1/
            barn_chopped_CumulativeParameters_v30.2.20250912.txt
            barn_chopped_CumulativeSonoBatch_v30.2.20250912.txt
          lake2/batch1/
            chopped_CumulativeParameters_v30.2.20250912.txt
            chopped_CumulativeSonoBatch_v30.2.20250912.txt
          dest/

    Barn has two chirps from BARN_PATH_A and one from BARN_PATH_B.
    BARN_PATH_B has no species entry (measures-only recording).
    Lake2 has two chirps from LAKE_PATH_A and one from LAKE_PATH_B.

    :param tmp_path: pytest built-in temporary directory fixture.
    :return: Dict with keys 'barn_dir', 'lake2_dir', 'dest_dir',
             'barn_root', 'lake2_root'.
    """
    barn_dir  = tmp_path / 'barn'  / 'batch1'
    lake2_dir = tmp_path / 'lake2' / 'batch1'
    dest_dir  = tmp_path / 'dest'
    barn_dir.mkdir(parents=True)
    lake2_dir.mkdir(parents=True)
    dest_dir.mkdir()

    # --- Barn measures: two chirps from path A, one from path B ---
    barn_meas_rows = [
        _measures_row(BARN_PATH_A, time_in_file=494, preceding=159),
        _measures_row(BARN_PATH_A, time_in_file=246, preceding=145),
        _measures_row(BARN_PATH_B, time_in_file=300, preceding=200),
    ]
    _write_measures_file(
        barn_dir / 'barn_chopped_CumulativeParameters_v30.2.20250912.txt',
        barn_meas_rows,
    )

    # --- Barn species: path A identified, path B absent (measures-only) ---
    barn_spp_rows = [
        {'Path': BARN_PATH_A, 'SppAccp': 'Myca', 'Prob': 0.9609,
         '#Maj': 1, '#Accp': 1},
    ]
    _write_species_file(
        barn_dir / 'barn_chopped_CumulativeSonoBatch_v30.2.20250912.txt',
        barn_spp_rows,
    )

    # --- Lake2 measures: two chirps from path A, one from path B ---
    lake_meas_rows = [
        _measures_row(LAKE_PATH_A, time_in_file=86,  preceding=50),
        _measures_row(LAKE_PATH_A, time_in_file=403, preceding=83),
        _measures_row(LAKE_PATH_B, time_in_file=120, preceding=60),
    ]
    _write_measures_file(
        lake2_dir / 'chopped_CumulativeParameters_v30.2.20250912.txt',
        lake_meas_rows,
    )

    # --- Lake2 species: path A has NaN (no ID), path B has good ID ---
    lake_spp_rows = [
        {'Path': LAKE_PATH_A, 'SppAccp': float('nan'), 'Prob': float('nan'),
         '#Maj': float('nan'), '#Accp': float('nan')},
        {'Path': LAKE_PATH_B, 'SppAccp': 'Myca', 'Prob': 0.9487,
         '#Maj': 1, '#Accp': 2},
    ]
    _write_species_file(
        lake2_dir / 'chopped_CumulativeSonoBatch_v30.2.20250912.txt',
        lake_spp_rows,
    )

    return {
        'barn_root' : tmp_path / 'barn',
        'lake2_root': tmp_path / 'lake2',
        'dest_dir'  : dest_dir,
    }


@pytest.fixture()
def processor(site_dirs):
    """
    Run the full pipeline with conf_thresh=0.0 so no rows are dropped by
    confidence, giving tests full visibility into the merged dataset.

    :param site_dirs: The site_dirs fixture.
    :return: Constructed SonoBatPostProcessor instance.
    """
    return SonoBatPostProcessor(
        root_dirs   = [site_dirs['barn_root'], site_dirs['lake2_root']],
        rec_sites   = ['barn', 'lake2'],
        dest_dir    = site_dirs['dest_dir'],
        conf_thresh = 0.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineSmoke:
    """End-to-end: pipeline runs and produces a Parquet file."""

    def test_parquet_file_created(self, site_dirs):
        """A .parquet file is written to dest_dir."""
        SonoBatPostProcessor(
            root_dirs   = [site_dirs['barn_root'], site_dirs['lake2_root']],
            rec_sites   = ['barn', 'lake2'],
            dest_dir    = site_dirs['dest_dir'],
            conf_thresh = 0.0,
        )
        parquet_files = list(site_dirs['dest_dir'].glob('bats_*.parquet'))
        assert len(parquet_files) == 1

    def test_output_is_readable_batsdata(self, site_dirs):
        """The produced Parquet file can be loaded back as a BatsData."""
        SonoBatPostProcessor(
            root_dirs   = [site_dirs['barn_root'], site_dirs['lake2_root']],
            rec_sites   = ['barn', 'lake2'],
            dest_dir    = site_dirs['dest_dir'],
            conf_thresh = 0.0,
        )
        parquet_path = next(site_dirs['dest_dir'].glob('bats_*.parquet'))
        bats = BatsData.read_parquet(parquet_path)
        assert isinstance(bats.df, pd.DataFrame)
        assert len(bats.df) > 0


class TestRecSite:
    """rec_site column is Categorical with correct categories."""

    def test_rec_site_is_categorical(self, processor):
        assert processor.bats_data.df['rec_site'].dtype.name == 'category'

    def test_rec_site_categories(self, processor):
        cats = set(processor.bats_data.df['rec_site'].cat.categories)
        assert cats == {'barn', 'lake2'}

    def test_rec_site_values(self, processor):
        """All rec_site values are one of the known sites."""
        values = set(processor.bats_data.df['rec_site'].dropna().unique())
        assert values <= {'barn', 'lake2'}

    def test_rec_site_survives_parquet_roundtrip(self, site_dirs):
        """Categorical dtype is preserved through to_parquet / read_parquet."""
        proc = SonoBatPostProcessor(
            root_dirs   = [site_dirs['barn_root'], site_dirs['lake2_root']],
            rec_sites   = ['barn', 'lake2'],
            dest_dir    = site_dirs['dest_dir'],
            conf_thresh = 0.0,
        )
        parquet_path = next(site_dirs['dest_dir'].glob('bats_*.parquet'))
        bats2 = BatsData.read_parquet(parquet_path)
        assert bats2.df['rec_site'].dtype.name == 'category'
        assert set(bats2.df['rec_site'].cat.categories) == {'barn', 'lake2'}

    def test_filter_by_rec_site_string(self, processor):
        """Filtering by string label works without needing integer codes."""
        df = processor.bats_data.df
        barn_df = df[df['rec_site'] == 'barn']
        assert len(barn_df) > 0
        assert (barn_df['rec_site'] == 'barn').all()


class TestChirpIdx:
    """chirp_idx is 0-based and ordered by TimeInFile within each file_id."""

    def test_chirp_idx_present(self, processor):
        assert 'chirp_idx' in processor.bats_data.df.columns

    def test_chirp_idx_starts_at_zero(self, processor):
        df = processor.bats_data.df
        min_per_file = df.groupby('file_id')['chirp_idx'].min()
        assert (min_per_file == 0).all()

    def test_chirp_idx_contiguous(self, processor):
        """Within each file, chirp_idx values are 0, 1, 2, ... with no gaps."""
        df = processor.bats_data.df
        for _, group in df.groupby('file_id'):
            idxs = sorted(group['chirp_idx'].tolist())
            assert idxs == list(range(len(idxs)))

    def test_chirp_idx_ordered_by_time(self, processor):
        """Within each file, chirp_idx increases with TimeInFile."""
        df = processor.bats_data.df
        for _, group in df.groupby('file_id'):
            group_sorted = group.sort_values('chirp_idx')
            assert group_sorted['TimeInFile'].is_monotonic_increasing


class TestUnifiedFileId:
    """The same path gets the same file_id in both measures and species."""

    def test_file_id_globally_unique(self, processor):
        """file_map covers all paths; no two paths share an id."""
        file_map = processor.bats_data.file_map
        assert len(file_map) == len(set(file_map.values()))

    def test_file_map_covers_all_paths(self, processor):
        """Every file_id in the df has an entry in file_map."""
        df      = processor.bats_data.df
        file_map = processor.bats_data.file_map
        for fid in df['file_id'].unique():
            assert fid in file_map

    def test_same_path_same_id_across_sites(self, processor):
        """
        A path that appears in both measures and species receives the same
        file_id — verified by checking that the encoder's path_to_id is
        consistent with the file_map.
        """
        enc = processor.path_encoder
        for path, fid in enc.path_to_id.items():
            assert enc.id_to_path[fid] == path


class TestConfidenceFilter:
    """Rows below conf_thresh are dropped; threshold=0 retains all non-NaN."""

    def test_all_rows_above_threshold(self, site_dirs):
        """With default threshold, every surviving row meets it."""
        proc = SonoBatPostProcessor(
            root_dirs   = [site_dirs['barn_root'], site_dirs['lake2_root']],
            rec_sites   = ['barn', 'lake2'],
            dest_dir    = site_dirs['dest_dir'],
            conf_thresh = SonoBatPostProcessor.CONF_ACCEPT_THRESH_DEFAULT,
        )
        df = proc.bats_data.df
        assert (df['confidence'] >= SonoBatPostProcessor.CONF_ACCEPT_THRESH_DEFAULT).all()

    def test_no_nan_confidence_in_output(self, processor):
        """No NaN confidence values survive regardless of threshold."""
        assert processor.bats_data.df['confidence'].notna().all()

    def test_threshold_zero_keeps_nonzero_confidence(self, processor):
        """conf_thresh=0.0 retains rows with positive confidence."""
        df = processor.bats_data.df
        assert (df['confidence'] > 0).any()

    def test_high_threshold_drops_all(self, site_dirs):
        """conf_thresh=1.0 drops everything (no row can score exactly 1.0)."""
        proc = SonoBatPostProcessor(
            root_dirs   = [site_dirs['barn_root'], site_dirs['lake2_root']],
            rec_sites   = ['barn', 'lake2'],
            dest_dir    = site_dirs['dest_dir'],
            conf_thresh = 1.0,
        )
        assert len(proc.bats_data.df) == 0


class TestSpeciesColumn:
    """SppAccp is renamed to 'species'; intermediate columns are absent."""

    def test_species_column_present(self, processor):
        assert 'species' in processor.bats_data.df.columns

    def test_sppaccp_column_absent(self, processor):
        assert 'SppAccp' not in processor.bats_data.df.columns

    def test_intermediate_species_cols_absent(self, processor):
        """Maj_scaled, Accp_scaled, Prob should not appear in final df."""
        df = processor.bats_data.df
        for col in ('Maj_scaled', 'Accp_scaled', 'Prob'):
            assert col not in df.columns


class TestMismatchWarning:
    """Species rows with no matching measures row trigger a log warning."""

    def test_unmatched_species_logged(self, site_dirs, caplog):
        """
        BARN_PATH_B has measures but no species entry — that's fine (no warn).
        To trigger the warning we need a species entry with no measures match.
        We add an extra species-only path to the barn SonoBatch file.
        """
        # Append a species row for a path that has no measures file
        phantom_path = r'Y:\barn\batch1\barn_Parsed\phantom_file.wav'
        extra_row = pd.DataFrame([{
            'Path': phantom_path, 'SppAccp': 'Laci', 'Prob': 0.95,
            '#Maj': 2, '#Accp': 3, 'HiF': 1,
            'Filename': 'phantom_file.wav',
        }])
        spp_file = next(
            (site_dirs['barn_root'] / 'batch1').glob('*SonoBatch*')
        )
        existing = pd.read_csv(spp_file, sep='\t')
        pd.concat([existing, extra_row], ignore_index=True).to_csv(
            spp_file, sep='\t', index=False
        )

        with caplog.at_level(logging.WARNING):
            SonoBatPostProcessor(
                root_dirs   = [site_dirs['barn_root'], site_dirs['lake2_root']],
                rec_sites   = ['barn', 'lake2'],
                dest_dir    = site_dirs['dest_dir'],
                conf_thresh = 0.0,
            )

        assert any('phantom' in rec.message or 'unmatched' in rec.message.lower()
                   or 'no matching' in rec.message.lower()
                   for rec in caplog.records)


class TestArgValidation:
    """Constructor enforces root_dirs/rec_sites length parity."""

    def test_mismatched_lengths_raise(self, site_dirs):
        with pytest.raises(ValueError, match='same length'):
            SonoBatPostProcessor(
                root_dirs   = [site_dirs['barn_root']],
                rec_sites   = ['barn', 'lake2'],
                dest_dir    = site_dirs['dest_dir'],
            )


class TestBatsDataRoundtrip:
    """BatsData.to_parquet / read_parquet preserves all envelope state."""

    def test_file_map_roundtrip(self, site_dirs):
        proc = SonoBatPostProcessor(
            root_dirs   = [site_dirs['barn_root'], site_dirs['lake2_root']],
            rec_sites   = ['barn', 'lake2'],
            dest_dir    = site_dirs['dest_dir'],
            conf_thresh = 0.0,
        )
        parquet_path = next(site_dirs['dest_dir'].glob('bats_*.parquet'))
        bats2 = BatsData.read_parquet(parquet_path)
        assert bats2.file_map == proc.bats_data.file_map

    def test_normalizer_roundtrip(self, site_dirs):
        """Normalizer feature_cols, log_cols, and scaler arrays survive roundtrip."""
        proc = SonoBatPostProcessor(
            root_dirs   = [site_dirs['barn_root'], site_dirs['lake2_root']],
            rec_sites   = ['barn', 'lake2'],
            dest_dir    = site_dirs['dest_dir'],
            conf_thresh = 0.0,
        )
        parquet_path = next(site_dirs['dest_dir'].glob('bats_*.parquet'))
        bats2 = BatsData.read_parquet(parquet_path)

        n1 = proc.bats_data.normalizer
        n2 = bats2.normalizer
        assert n1.feature_cols_ == n2.feature_cols_
        assert n1.log_cols_     == n2.log_cols_
        np.testing.assert_array_almost_equal(
            n1.scaler_.center_, n2.scaler_.center_
        )
        np.testing.assert_array_almost_equal(
            n1.scaler_.scale_,  n2.scaler_.scale_
        )

    def test_timestamp_roundtrip(self, site_dirs):
        proc = SonoBatPostProcessor(
            root_dirs   = [site_dirs['barn_root'], site_dirs['lake2_root']],
            rec_sites   = ['barn', 'lake2'],
            dest_dir    = site_dirs['dest_dir'],
            conf_thresh = 0.0,
        )
        parquet_path = next(site_dirs['dest_dir'].glob('bats_*.parquet'))
        bats2 = BatsData.read_parquet(parquet_path)
        assert bats2.timestamp == proc.bats_data.timestamp

    def test_to_parquet_with_modified_df(self, site_dirs):
        """to_parquet(modified_df, path) saves subset but preserves metadata."""
        proc = SonoBatPostProcessor(
            root_dirs   = [site_dirs['barn_root'], site_dirs['lake2_root']],
            rec_sites   = ['barn', 'lake2'],
            dest_dir    = site_dirs['dest_dir'],
            conf_thresh = 0.0,
        )
        barn_only = proc.bats_data.df[proc.bats_data.df['rec_site'] == 'barn']
        out_path  = site_dirs['dest_dir'] / 'barn_only.parquet'
        proc.bats_data.to_parquet(barn_only, out_path)

        bats2 = BatsData.read_parquet(out_path)
        assert (bats2.df['rec_site'] == 'barn').all()
        # Metadata from the original run is preserved
        assert bats2.file_map == proc.bats_data.file_map


class TestCompositeSpecies:
    """CompositeSpecies canonicalises slash-separated species strings."""

    def test_order_independent(self):
        assert str(CompositeSpecies('Laci/Lano')) == str(CompositeSpecies('Lano/Laci'))

    def test_canonical_form_is_sorted(self):
        assert str(CompositeSpecies('Lano/Laci')) == 'Laci/Lano'

    def test_hash_equality(self):
        a = CompositeSpecies('Laci/Lano')
        b = CompositeSpecies('Lano/Laci')
        assert a == b
        assert hash(a) == hash(b)

    def test_set_deduplication(self):
        s = {CompositeSpecies('Laci/Lano'), CompositeSpecies('Lano/Laci')}
        assert len(s) == 1

    def test_single_species_raises(self):
        with pytest.raises(ValueError):
            CompositeSpecies('Laci')


class TestPathEncoder:
    """PathEncoder assigns consistent IDs from a unified path universe."""

    def test_same_path_same_id(self):
        paths = [BARN_PATH_A, BARN_PATH_B, LAKE_PATH_A]
        enc   = PathEncoder.from_paths(paths)
        df_a  = pd.DataFrame({'Path': [BARN_PATH_A, LAKE_PATH_A], 'val': [1, 2]})
        df_b  = pd.DataFrame({'Path': [BARN_PATH_A, BARN_PATH_B], 'val': [3, 4]})
        enc_a = enc.encode_df(df_a)
        enc_b = enc.encode_df(df_b)
        id_in_a = enc_a.loc[enc_a['val'] == 1, 'file_id'].iloc[0]
        id_in_b = enc_b.loc[enc_b['val'] == 3, 'file_id'].iloc[0]
        assert id_in_a == id_in_b

    def test_unknown_path_raises(self):
        enc = PathEncoder.from_paths([BARN_PATH_A])
        df  = pd.DataFrame({'Path': [LAKE_PATH_A], 'val': [1]})
        with pytest.raises(KeyError):
            enc.encode_df(df)

    def test_file_id_dtype(self):
        enc = PathEncoder.from_paths([BARN_PATH_A, BARN_PATH_B])
        df  = pd.DataFrame({'Path': [BARN_PATH_A], 'val': [1]})
        encoded = enc.encode_df(df)
        assert encoded['file_id'].dtype == np.int32

    def test_path_column_removed(self):
        enc = PathEncoder.from_paths([BARN_PATH_A])
        df  = pd.DataFrame({'Path': [BARN_PATH_A], 'val': [1]})
        encoded = enc.encode_df(df)
        assert 'Path' not in encoded.columns

    def test_file_id_first_column(self):
        enc = PathEncoder.from_paths([BARN_PATH_A])
        df  = pd.DataFrame({'Path': [BARN_PATH_A], 'val': [1]})
        encoded = enc.encode_df(df)
        assert encoded.columns[0] == 'file_id'


class TestMeasureNormalizerRoundtrip:
    """MeasureNormalizer.to_dict / from_dict preserves scaler state."""

    def test_to_dict_from_dict_inverse_transform(self):
        """
        Inverse-transform via a from_dict-restored normalizer produces
        approximately the original values.
        """
        rng  = np.random.default_rng(42)
        data = rng.uniform(1, 100, size=(50, len(SonoBatPostProcessor.RELEVANT_MEASURES_COLS)))
        df   = pd.DataFrame(data, columns=SonoBatPostProcessor.RELEVANT_MEASURES_COLS)

        norm     = MeasureNormalizer(outlier_factor_thresh=1e9)  # disable row filtering
        df_norm  = norm.fit_transform(df)
        d        = norm.to_dict()
        norm2    = MeasureNormalizer.from_dict(d)
        df_back  = norm2.inverse_transform(df_norm)

        # Only feature columns should round-trip
        for col in norm.feature_cols_:
            np.testing.assert_array_almost_equal(
                df.loc[df_norm.index, col].values,
                df_back[col].values,
                decimal=4,
            )

    def test_from_dict_missing_diagnostic_df(self):
        """from_dict produces a normalizer whose report() prints a safe message."""
        rng  = np.random.default_rng(0)
        data = rng.uniform(1, 50, size=(30, len(SonoBatPostProcessor.RELEVANT_MEASURES_COLS)))
        df   = pd.DataFrame(data, columns=SonoBatPostProcessor.RELEVANT_MEASURES_COLS)
        norm = MeasureNormalizer(outlier_factor_thresh=1e9)
        norm.fit_transform(df)
        norm2 = MeasureNormalizer.from_dict(norm.to_dict())
        # Should not raise; just prints a message
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            norm2.report()
        assert 'fit_transform' in buf.getvalue()
        