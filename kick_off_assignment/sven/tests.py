from param_types import (
    CategoricalParam,
    IntegerParam,
    OrdinalParam,
    PermutationParam,
    RealParam,
    SwitchParam,
)


def test_switch_param_get_n_samples():
    param = SwitchParam(name="switch")
    samples = list(param.get_n_samples(4))
    assert samples == [False, False, True, True]


def test_real_param_get_n_samples():
    param = RealParam(name="real", val_range=(0.0, 1.0))
    samples = list(param.get_n_samples(6))
    max_delta = 0.01
    for idx, wanted in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
        assert abs(samples[idx] - wanted) < max_delta


def test_integer_param_get_n_samples():
    param = IntegerParam(name="integer", val_range=(0, 10))
    samples = list(param.get_n_samples(6))
    assert samples == [0, 2, 4, 6, 8, 10]
    samples = list(param.get_n_samples(5))
    assert samples == [0, 2, 5, 7, 10]


def test_ordinal_param_get_n_samples():
    param = OrdinalParam(name="ordinal", val_range=(1, 16))
    samples = list(param.get_n_samples(4))
    assert samples == [1, 2, 4, 16]


def test_categorical_param_get_n_samples():
    param = CategoricalParam(name="categorical", val_range=("a", "b", "c", "d"))
    samples = list(param.get_n_samples(2))
    assert samples == ["a", "d"]
    samples = list(param.get_n_samples(3))
    assert samples == ["a", "b", "d"]
    samples = list(param.get_n_samples(8))


def test_permutation_param_get_n_samples():
    param = PermutationParam(name="permutation", val_range=([0, 0], [10, 10]))
    samples = list(param.get_n_samples(3))
    assert samples == [[0, 0], [5, 5], [10, 10]]
