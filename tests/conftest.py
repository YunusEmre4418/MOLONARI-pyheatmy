"""Shared pytest configuration for pyheatmy. pytest loads this file automatically."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: runs an MCMC, takes about a minute (skip with -m 'not slow')"
    )
    config.addinivalue_line(
        "markers", "known_bug: documents a known, not yet fixed bug; used with xfail(strict=True)"
    )
