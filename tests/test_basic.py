def test_import():
    from morpho_tokenizer import MorphoTokenizer
    tok = MorphoTokenizer()
    assert tok is not None
