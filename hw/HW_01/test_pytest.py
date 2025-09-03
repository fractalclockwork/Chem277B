import time
import pytest
import concurrent.futures


def normal():
    time.sleep(1)


def abnormal():
    time.sleep(2)


# Positive test: should complete under 2 seconds
@pytest.mark.timeout(2)
def test_if_normal():
    normal()
    assert True  # confirms it completed


# Negative test: should timeout in 1 second
def test_if_abnormal():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(abnormal)
        try:
            future.result(timeout=1)
            assert False, "Function did not timeout as expected"
        except concurrent.futures.TimeoutError:
            assert True  # Timeout occurred, test passes
