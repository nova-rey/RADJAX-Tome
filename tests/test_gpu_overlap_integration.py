from pathlib import Path


def test_native_c6_compact_uses_buffer_pipeline():
    source = (Path(__file__).parents[1] / 'src/radjax_tome/builder/delivery/assembly.py').read_text()
    assert 'if _native_streamed_payloads(config) and config.representation_mode == COMPACT_K_MONOLITHIC' in source
    assert 'write_compact_body_store_pipelined_from_compact' in source


def test_pipeline_uses_public_buffer_contract_boundary():
    source = (Path(__file__).parents[1] / 'src/radjax_tome/builder/delivery/simple_compact_body.py').read_text()
    assert 'compact_body_from_buffers' in source
    assert 'encode_compact_body_packed_from_buffers' in source
    assert 'compact_body_from_logical_payload' not in source.split('def write_compact_body_store_pipelined_from_compact', 1)[1]
