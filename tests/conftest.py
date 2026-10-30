"""Shared pytest configuration for pyheatmy. pytest loads this file automatically."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "known_bug: documents a known, not yet fixed bug; used with xfail(strict=True)"
    )
