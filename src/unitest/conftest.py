
import py
import pytest


@pytest.fixture
def toolkit_dir():
    return py.path.local(__file__).dirpath().join('..', '..')


@pytest.fixture
def data_dir(toolkit_dir):
    return toolkit_dir.join('resource', 'unittest')


@pytest.fixture
def config_dir(tmpdir):
    return tmpdir.mkdir('config')


@pytest.fixture
def pipelines_test_resources(data_dir):
    return data_dir.join('pipelines_test_resources')


@pytest.fixture
def sim_input(pipelines_test_resources):
    return str(pipelines_test_resources.join('sim_input.txt'))

