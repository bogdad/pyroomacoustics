"""
Tests for streaming room simulation functionality.

These tests verify that the streaming simulation produces identical results
to the batch simulation mode.
"""

import numpy as np
import pytest
import pyroomacoustics as pra


def test_streaming_single_source():
    """Test streaming mode with a single source matches batch mode"""

    # Setup parameters
    fs = 16000
    block_size = 256
    duration = 0.5
    signal = np.random.randn(int(fs * duration))

    # Truncate to block boundary
    n_blocks = len(signal) // block_size
    signal = signal[: n_blocks * block_size]

    # Batch simulation
    room_batch = pra.ShoeBox([4, 6, 3], fs=fs)
    room_batch.add_source([1, 2, 1.5], signal=signal)
    mic_locs = pra.linear_2D_array([3, 3], 4, 0, 0.1)
    mic_locs = np.vstack([mic_locs, np.ones(4) * 1.2])  # Add z-coordinate
    room_batch.add_microphone_array(pra.MicrophoneArray(mic_locs, fs=fs))
    room_batch.compute_rir()
    room_batch.simulate()
    batch_output = room_batch.mic_array.signals

    # Streaming simulation
    room_stream = pra.ShoeBox([4, 6, 3], fs=fs)
    room_stream.add_source([1, 2, 1.5])
    mic_locs = pra.linear_2D_array([3, 3], 4, 0, 0.1)
    mic_locs = np.vstack([mic_locs, np.ones(4) * 1.2])  # Add z-coordinate
    room_stream.add_microphone_array(pra.MicrophoneArray(mic_locs, fs=fs))
    room_stream.compute_rir()
    room_stream.init_streaming(block_size)

    room_stream.mic_array.signals = None
    for i in range(n_blocks):
        block = signal[i * block_size : (i + 1) * block_size]
        mic_block = room_stream.simulate_block([block])
        room_stream.mic_array.record_block(mic_block, fs)

    streaming_output = room_stream.mic_array.signals

    # Compare (only the part without RIR tail)
    min_len = min(streaming_output.shape[1], batch_output.shape[1])
    max_error = np.max(np.abs(streaming_output[:, :min_len] - batch_output[:, :min_len]))

    assert max_error < 1e-10, f"Streaming mismatch: max error = {max_error}"


def test_streaming_multiple_sources():
    """Test streaming with multiple sources"""

    fs = 16000
    block_size = 128
    n_samples = 2048
    n_sources = 3

    signals = [np.random.randn(n_samples) for _ in range(n_sources)]

    # Batch
    room = pra.ShoeBox([5, 5, 3], fs=fs, max_order=2)
    for i, sig in enumerate(signals):
        room.add_source([1 + i, 2, 1.5], signal=sig)
    room.add_microphone([2.5, 2.5, 1.5])
    room.compute_rir()
    room.simulate()
    batch_out = room.mic_array.signals

    # Streaming
    room2 = pra.ShoeBox([5, 5, 3], fs=fs, max_order=2)
    for i in range(n_sources):
        room2.add_source([1 + i, 2, 1.5])
    room2.add_microphone([2.5, 2.5, 1.5])
    room2.compute_rir()
    room2.init_streaming(block_size)

    room2.mic_array.signals = None
    n_blocks = n_samples // block_size
    for i in range(n_blocks):
        blocks = [sig[i * block_size : (i + 1) * block_size] for sig in signals]
        mic_block = room2.simulate_block(blocks)
        room2.mic_array.record_block(mic_block, fs)

    streaming_out = room2.mic_array.signals

    min_len = min(streaming_out.shape[1], batch_out.shape[1])
    max_error = np.max(np.abs(streaming_out[:, :min_len] - batch_out[:, :min_len]))

    assert max_error < 1e-10


def test_streaming_different_block_sizes():
    """Test that different block sizes produce the same result"""

    fs = 16000
    signal = np.random.randn(4096)

    # Reference batch
    room_ref = pra.ShoeBox([4, 4, 3], fs=fs)
    room_ref.add_source([1, 2, 1.5], signal=signal)
    room_ref.add_microphone([2, 2, 1.5])
    room_ref.compute_rir()
    room_ref.simulate()
    ref_output = room_ref.mic_array.signals

    # Test different block sizes
    for block_size in [64, 128, 256, 512]:
        n_blocks = len(signal) // block_size
        truncated_signal = signal[: n_blocks * block_size]

        room_stream = pra.ShoeBox([4, 4, 3], fs=fs)
        room_stream.add_source([1, 2, 1.5])
        room_stream.add_microphone([2, 2, 1.5])
        room_stream.rir = room_ref.rir  # Reuse RIRs
        room_stream.init_streaming(block_size)

        room_stream.mic_array.signals = None
        for i in range(n_blocks):
            block = truncated_signal[i * block_size : (i + 1) * block_size]
            mic_block = room_stream.simulate_block([block])
            room_stream.mic_array.record_block(mic_block, fs)

        streaming_out = room_stream.mic_array.signals
        min_len = min(streaming_out.shape[1], ref_output.shape[1])

        max_error = np.max(np.abs(streaming_out[:, :min_len] - ref_output[:, :min_len]))
        assert (
            max_error < 1e-10
        ), f"Block size {block_size} failed: max error = {max_error}"


def test_streaming_with_absorption():
    """Test streaming works correctly with room absorption"""

    fs = 16000
    block_size = 256
    signal = np.random.randn(2048)

    # Batch with absorption
    room_batch = pra.ShoeBox(
        [4, 5, 3], fs=fs, materials=pra.Material(energy_absorption=0.3), max_order=5
    )
    room_batch.add_source([1, 2, 1.5], signal=signal)
    mic_locs = pra.circular_2D_array([2, 2.5], 3, 0, 0.05)
    mic_locs = np.vstack([mic_locs, np.ones(3) * 1.2])  # Add z-coordinate
    room_batch.add_microphone_array(pra.MicrophoneArray(mic_locs, fs=fs))
    room_batch.compute_rir()
    room_batch.simulate()
    batch_out = room_batch.mic_array.signals

    # Streaming with absorption
    room_stream = pra.ShoeBox(
        [4, 5, 3], fs=fs, materials=pra.Material(energy_absorption=0.3), max_order=5
    )
    room_stream.add_source([1, 2, 1.5])
    mic_locs = pra.circular_2D_array([2, 2.5], 3, 0, 0.05)
    mic_locs = np.vstack([mic_locs, np.ones(3) * 1.2])  # Add z-coordinate
    room_stream.add_microphone_array(pra.MicrophoneArray(mic_locs, fs=fs))
    room_stream.compute_rir()
    room_stream.init_streaming(block_size)

    room_stream.mic_array.signals = None
    n_blocks = len(signal) // block_size
    for i in range(n_blocks):
        block = signal[i * block_size : (i + 1) * block_size]
        mic_block = room_stream.simulate_block([block])
        room_stream.mic_array.record_block(mic_block, fs)

    streaming_out = room_stream.mic_array.signals
    min_len = min(streaming_out.shape[1], batch_out.shape[1])

    max_error = np.max(np.abs(streaming_out[:, :min_len] - batch_out[:, :min_len]))
    assert max_error < 1e-10


def test_streaming_errors():
    """Test that streaming mode raises appropriate errors"""

    fs = 16000
    block_size = 256

    room = pra.ShoeBox([4, 4, 3], fs=fs)
    room.add_source([1, 2, 1.5])
    room.add_microphone([2, 2, 1.5])

    # Error: simulate_block before init_streaming
    with pytest.raises(ValueError, match="Must call init_streaming"):
        room.simulate_block([np.zeros(block_size)])

    # Error: init_streaming before compute_rir
    with pytest.raises(ValueError, match="Must call compute_rir"):
        room.init_streaming(block_size)

    room.compute_rir()
    room.init_streaming(block_size)

    # Error: wrong number of sources
    with pytest.raises(ValueError, match="Expected 1 source blocks"):
        room.simulate_block([np.zeros(block_size), np.zeros(block_size)])

    # Error: wrong block size
    with pytest.raises(ValueError, match="Expected block size 256"):
        room.simulate_block([np.zeros(128)])


def test_streaming_reset_state():
    """Test that convolvers can be reset for independent streams"""

    fs = 16000
    block_size = 256

    room = pra.ShoeBox([4, 4, 3], fs=fs)
    room.add_source([1, 2, 1.5])
    room.add_microphone([2, 2, 1.5])
    room.compute_rir()
    room.init_streaming(block_size)

    # Process first stream
    signal1 = np.random.randn(1024)
    room.mic_array.signals = None
    for i in range(4):
        block = signal1[i * block_size : (i + 1) * block_size]
        mic_block = room.simulate_block([block])
        room.mic_array.record_block(mic_block, fs)
    output1 = room.mic_array.signals.copy()

    # Reset convolvers
    for convolver in room.convolvers.values():
        convolver.reset()

    # Process second stream (same signal)
    room.mic_array.signals = None
    for i in range(4):
        block = signal1[i * block_size : (i + 1) * block_size]
        mic_block = room.simulate_block([block])
        room.mic_array.record_block(mic_block, fs)
    output2 = room.mic_array.signals.copy()

    # Should be identical after reset
    assert np.allclose(output1, output2)


def test_record_block_errors():
    """Test that record_block raises appropriate errors"""

    fs = 16000
    mic_array = pra.MicrophoneArray(np.array([[0, 1], [0, 0]]), fs=fs)

    # Error: wrong number of channels
    with pytest.raises(ValueError, match="should have 2 rows"):
        mic_array.record_block(np.zeros((3, 128)), fs)

    # Error: not 2D array
    with pytest.raises(ValueError, match="must be a 2D array"):
        mic_array.record_block(np.zeros(128), fs)

    # Error: sampling frequency mismatch
    with pytest.raises(ValueError, match="does not support resampling"):
        mic_array.record_block(np.zeros((2, 128)), fs=8000)


def test_streaming_continuous_processing():
    """Test that streaming produces continuous output without discontinuities"""

    fs = 16000
    block_size = 128
    n_blocks = 20

    # Create a continuous sine wave
    t_total = np.arange(n_blocks * block_size) / fs
    signal = np.sin(2 * np.pi * 440 * t_total)

    # Process in streaming mode
    room = pra.ShoeBox([4, 4, 3], fs=fs, max_order=1)
    room.add_source([1, 2, 1.5])
    room.add_microphone([2, 2, 1.5])
    room.compute_rir()
    room.init_streaming(block_size)

    room.mic_array.signals = None
    for i in range(n_blocks):
        block = signal[i * block_size : (i + 1) * block_size]
        mic_block = room.simulate_block([block])
        room.mic_array.record_block(mic_block, fs)

    output = room.mic_array.signals[0]

    # Check for discontinuities at block boundaries
    # Compute differences at block boundaries
    block_boundaries = np.arange(1, n_blocks) * block_size
    for boundary in block_boundaries:
        # The difference across boundary should be similar to differences within blocks
        diff_at_boundary = abs(output[boundary] - output[boundary - 1])
        diff_before = abs(output[boundary - 1] - output[boundary - 2])
        diff_after = abs(output[boundary + 1] - output[boundary])

        # Discontinuity would show as much larger difference
        assert (
            diff_at_boundary < 10 * max(diff_before, diff_after)
        ), f"Discontinuity detected at boundary {boundary}"


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])
