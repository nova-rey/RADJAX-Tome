def test_package_imports() -> None:
    import radjax_tome

    assert radjax_tome.FakeTeacherBackend().vocab_size == 8
