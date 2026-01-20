import pytest

pytestmark = pytest.mark.pipeline


def pytest_addoption(parser):
    parser.addoption("--test-server-id", action="store", default=None, help="Discord Server ID for testing")
    parser.addoption("--test-channel-id", action="store", default=None, help="Discord Channel ID for testing")
