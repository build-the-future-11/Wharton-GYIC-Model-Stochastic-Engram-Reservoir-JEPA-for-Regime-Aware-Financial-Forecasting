"""File-level admission regressions; no financial data or model fitting."""
import hashlib

import numpy as np
import pytest

from frem import benchmark


@pytest.mark.parametrize('source', [
    'timestamp,value,value\n2026-01-01,nan,1\n2026-01-02,inf,2\n',
    'timestamp,timestamp,value\n2099-01-01,2026-01-01,1\n1900-01-01,2026-01-02,2\n',
    'timestamp,value,note,note\n2026-01-01,1,a,b\n',
    'timestamp,value,\n2026-01-01,1,a\n',
    'timestamp,value,   \n2026-01-01,1,a\n',
    'timestamp,value\n2026-01-01,1,discarded\n2026-01-02,2,discarded\n',
    'timestamp,value,note\n2026-01-01,1\n2026-01-02,2,complete\n',
    'timestamp,value,note\n2026-01-01,1,"unterminated\n',
    'timestamp,value,note\n2026-01-01,1,"quoted"junk\n',
])
def test_malformed_csv_cannot_hide_or_discard_evidence(tmp_path, source):
    path = tmp_path / 'input.csv'
    path.write_text(source, encoding='utf-8')
    with pytest.raises(ValueError):
        benchmark.read_csv(path, 'timestamp', 'value')


@pytest.mark.parametrize('timestamp_col,value_col', [
    ('timestamp', 'timestamp'), ('', 'value'), ('timestamp', ''),
    ('missing', 'value'), ('timestamp', 'missing'),
])
def test_selected_columns_are_present_and_distinct(tmp_path, timestamp_col, value_col):
    path = tmp_path / 'input.csv'
    path.write_text('timestamp,value\n2026-01-01,1\n2026-01-02,2\n')
    with pytest.raises(ValueError):
        benchmark.read_csv(path, timestamp_col, value_col)


def test_valid_optional_quoted_columns_bom_and_raw_digest_are_preserved(tmp_path):
    raw = ('\ufefftimestamp,value,note\r\n'
           '2026-01-01T10:00:00+05:30,1.25,"line one\r\nline two, quoted"\r\n'
           '2026-01-01T05:00:00Z,-2.5,"second"\r\n').encode('utf-8')
    path = tmp_path / 'input.csv'
    path.write_bytes(raw)
    values, digest = benchmark.read_csv(path, 'timestamp', 'value')
    assert np.array_equal(values, [1.25, -2.5])
    assert digest == hashlib.sha256(raw).hexdigest()
    assert path.read_bytes() == raw


def test_oversized_input_is_rejected_after_one_bounded_read(monkeypatch):
    class BoundedFile:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self, size):
            assert size == 129
            return b'x' * size
    monkeypatch.setattr(benchmark, '_MAX_CSV_BYTES', 128, raising=False)
    monkeypatch.setattr(benchmark.Path, 'open', lambda self, *a, **kw: BoundedFile())
    with pytest.raises(ValueError, match='bounded'):
        benchmark.read_csv('bounded.csv', 'timestamp', 'value')
