import pytest

from data_creation import DataGeneration, Noise, Parameters
from models import FEInterceptOLS, FullFEOLS, NormalOLS, REFullOLS, REInterceptOLS


@pytest.fixture
def data_creation():
    data_creation = DataGeneration(4)
    return data_creation


@pytest.fixture
def normal_ols():
    return NormalOLS()


@pytest.fixture
def fe_intercept_ols():
    return FEInterceptOLS()


@pytest.fixture
def full_fe_ols():
    return FullFEOLS()


@pytest.fixture
def re_intercept_ols():
    return REInterceptOLS()


@pytest.fixture
def re_full_ols():
    return REFullOLS()


def test_simulate_data(
    data_creation,
    normal_ols,
    fe_intercept_ols,
    full_fe_ols,
    re_intercept_ols,
    re_full_ols,
) -> None:
    noise = Noise(individual_noise=1, school_noise=1)
    data, true_parameters = data_creation.simulate_data(1000000, noise)
    assert len(data) == 1000000
    assert len(true_parameters.school_intercepts) == 4
    assert len(true_parameters.school_air_quality) == 4
    print(true_parameters)
    values = normal_ols.fit(data)
    values = fe_intercept_ols.fit(data)
    values = full_fe_ols.fit(data)
    values = re_intercept_ols.fit(data)
    values = re_full_ols.fit(data)
    print(values)
