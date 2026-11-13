def test_imports():
    import coilgen
    import coilstellaration
    from coilstellaration.benchmark import scoring

    assert coilgen.__version__
    assert coilstellaration is not None
    assert scoring is not None
