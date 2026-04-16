"""
Tests for the preprocessing module.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from preprocessing.loader import _map_attack_category, _add_labels
from preprocessing.pipeline import make_sequences

def test_map_attack_category():
    assert _map_attack_category("normal") == "Normal"
    assert _map_attack_category("neptune") == "DoS"
    assert _map_attack_category("satan") == "Probe"
    assert _map_attack_category("warezclient") == "R2L"
    assert _map_attack_category("rootkit") == "U2R"
    assert _map_attack_category("unknown_attack") == "DoS" # Default

def test_add_labels():
    df = pd.DataFrame({'label': ['normal.', 'neptune.']})
    df = _add_labels(df)
    assert 'label_binary' in df.columns
    assert 'label_category' in df.columns
    assert df.loc[0, 'label_binary'] == 0
    assert df.loc[1, 'label_binary'] == 1
    assert df.loc[0, 'label_category'] == 'Normal'
    assert df.loc[1, 'label_category'] == 'DoS'

def test_make_sequences():
    X = np.random.rand(20, 5)
    y = np.arange(20)
    seq_len = 10
    X_seq, y_seq = make_sequences(X, y, seq_len=seq_len)
    
    # N - seq_len + 1 = 20 - 10 + 1 = 11
    assert X_seq.shape == (11, 10, 5)
    assert y_seq.shape == (11,)
    # Last element of first sequence should be y[9]
    assert y_seq[0] == 9
    # First element of first sequence should be X[0]
    assert np.array_equal(X_seq[0, 0], X[0])
    # Last element of last sequence should be y[19]
    assert y_seq[-1] == 19
