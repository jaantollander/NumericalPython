from hypothesis import given
import hypothesis.strategies as st

from package.module import routine


@given(st.floats(), st.floats())
def test_routine():
    assert True
