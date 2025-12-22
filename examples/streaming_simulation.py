"""
Streaming Room Simulation Example
==================================

This example demonstrates how to use pyroomacoustics in streaming mode
for block-by-block audio processing. This is useful for:
- Real-time applications
- Processing long audio files with limited memory
- Integration with live audio input/output

The example shows:
1. Pre-computing RIRs once
2. Initializing streaming mode
3. Processing audio block-by-block
4. Verifying equivalence with batch processing
"""

import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra

# Uncomment to save/load audio files
# from scipy.io import wavfile


def main():
    # =================================================================
    # Parameters
    # =================================================================
    fs = 16000  # Sampling frequency
    block_size = 512  # Samples per block (affects latency)
    duration = 2.0  # seconds of audio to simulate

    # Room dimensions [width, length, height] in meters
    room_dim = [8, 9, 3]

    # Source positions (x, y, z) in meters
    source_positions = [
        [2, 3, 1.5],  # Source 1
        [6, 5, 1.5],  # Source 2
    ]

    # Microphone array center and parameters
    mic_center = [4, 4.5, 1.2]
    mic_array_radius = 0.05
    n_mics = 4

    print("=" * 70)
    print("STREAMING ROOM SIMULATION EXAMPLE")
    print("=" * 70)
    print(f"Room dimensions: {room_dim} m")
    print(f"Sampling frequency: {fs} Hz")
    print(f"Block size: {block_size} samples ({block_size/fs*1000:.1f} ms)")
    print(f"Duration: {duration} s")
    print(f"Number of sources: {len(source_positions)}")
    print(f"Number of microphones: {n_mics}")
    print()

    # =================================================================
    # Generate Test Signals
    # =================================================================
    print("Generating test signals...")
    n_samples = int(fs * duration)

    # Truncate to block boundary for fair comparison
    n_blocks = n_samples // block_size
    n_samples = n_blocks * block_size

    t = np.arange(n_samples) / fs

    # Source 1: 440 Hz tone (musical A)
    signal_1 = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Source 2: 880 Hz tone (musical A, one octave higher)
    signal_2 = 0.3 * np.sin(2 * np.pi * 880 * t)

    # Alternative: Load from wav files
    # fs_file, signal_1 = wavfile.read("source1.wav")
    # fs_file, signal_2 = wavfile.read("source2.wav")
    # # Resample if needed...

    source_signals = [signal_1, signal_2]

    print(f"  Generated {n_samples} samples ({n_samples/fs:.2f} s)")
    print(f"  Will process {n_blocks} blocks")
    print()

    # =================================================================
    # Setup Room (Batch Mode - for comparison)
    # =================================================================
    print("Setting up batch simulation...")

    # Create room with some absorption
    room_batch = pra.ShoeBox(
        room_dim, fs=fs, materials=pra.Material(energy_absorption=0.2), max_order=3
    )

    # Add sources with signals
    for i, (pos, sig) in enumerate(zip(source_positions, source_signals)):
        room_batch.add_source(pos, signal=sig)
        print(f"  Added source {i+1} at {pos}")

    # Add circular microphone array
    mic_array = pra.MicrophoneArray(
        pra.circular_2D_array(mic_center[:2], n_mics, 0, mic_array_radius),
        fs=fs,
    )
    mic_array.R = np.vstack([mic_array.R, np.ones(n_mics) * mic_center[2]])
    room_batch.add_microphone_array(mic_array)
    print(f"  Added {n_mics}-element circular array at {mic_center}")
    print()

    # Compute RIRs (done once, shared by both modes)
    print("Computing Room Impulse Responses...")
    room_batch.compute_rir()
    rir_length = len(room_batch.rir[0][0])
    print(f"  RIR length: {rir_length} samples ({rir_length/fs*1000:.1f} ms)")
    print()

    # Run batch simulation
    print("Running batch simulation...")
    room_batch.simulate()
    batch_output = room_batch.mic_array.signals
    print(f"  Output shape: {batch_output.shape}")
    print()

    # =================================================================
    # Setup Room (Streaming Mode)
    # =================================================================
    print("Setting up streaming simulation...")

    # Create identical room (without signals in sources)
    room_stream = pra.ShoeBox(
        room_dim, fs=fs, materials=pra.Material(energy_absorption=0.2), max_order=3
    )

    # Add sources WITHOUT signals (we'll provide them block-by-block)
    for i, pos in enumerate(source_positions):
        room_stream.add_source(pos)  # No signal argument

    # Add same microphone array
    mic_array_stream = pra.MicrophoneArray(
        pra.circular_2D_array(mic_center[:2], n_mics, 0, mic_array_radius),
        fs=fs,
    )
    mic_array_stream.R = np.vstack(
        [mic_array_stream.R, np.ones(n_mics) * mic_center[2]]
    )
    room_stream.add_microphone_array(mic_array_stream)

    # Reuse RIRs from batch simulation (they're identical)
    room_stream.rir = room_batch.rir

    # Initialize streaming mode
    print(f"Initializing streaming with block_size={block_size}...")
    room_stream.init_streaming(block_size)
    print()

    # =================================================================
    # Process Audio Block-by-Block
    # =================================================================
    print("Processing blocks...")
    room_stream.mic_array.signals = None  # Reset recording

    for block_idx in range(n_blocks):
        # Extract block from each source signal
        start = block_idx * block_size
        end = start + block_size

        source_blocks = np.array([sig[start:end] for sig in source_signals])

        # Process through room acoustics
        mic_block = room_stream.simulate_block(source_blocks)

        # Record output
        room_stream.mic_array.record_block(mic_block, fs)

        # Progress indicator
        if block_idx % 10 == 0 or block_idx == n_blocks - 1:
            progress = (block_idx + 1) / n_blocks * 100
            print(f"  Block {block_idx+1:3d}/{n_blocks} ({progress:5.1f}%)")

    streaming_output = room_stream.mic_array.signals
    print(f"\nStreaming output shape: {streaming_output.shape}")
    print()

    # =================================================================
    # Verify Results Match
    # =================================================================
    print("Comparing batch vs streaming results...")

    # Batch output includes the RIR tail, so truncate to match streaming length
    # (streaming only outputs what's fed in blocks)
    min_length = min(streaming_output.shape[1], batch_output.shape[1])
    batch_truncated = batch_output[:, :min_length]
    streaming_truncated = streaming_output[:, :min_length]

    print(f"  Comparing first {min_length} samples")
    print(f"  (Batch tail of {batch_output.shape[1] - min_length} samples ignored)")

    difference = streaming_truncated - batch_truncated
    max_error = np.max(np.abs(difference))
    mean_error = np.mean(np.abs(difference))
    rms_error = np.sqrt(np.mean(difference**2))

    print(f"  Max error:  {max_error:.2e}")
    print(f"  Mean error: {mean_error:.2e}")
    print(f"  RMS error:  {rms_error:.2e}")
    print()

    if max_error < 1e-9:
        print("✓ SUCCESS: Streaming output matches batch processing!")
    else:
        print("✗ WARNING: Outputs don't match exactly (numerical precision?)")
    print()

    # =================================================================
    # Save Output
    # =================================================================
    # Uncomment to save output files
    # print("Saving output files...")
    # room_stream.mic_array.to_wav("streaming_output.wav", norm=True, bitdepth=np.int16)
    # room_batch.mic_array.to_wav("batch_output.wav", norm=True, bitdepth=np.int16)
    # print("  Saved: streaming_output.wav, batch_output.wav")
    # print()

    # =================================================================
    # Visualization
    # =================================================================
    print("Creating visualization...")

    fig = plt.figure(figsize=(12, 10))

    # Create subplots: first two are 2D, third is 3D
    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2)
    ax3 = plt.subplot(3, 1, 3, projection='3d')
    axes = [ax1, ax2, ax3]

    # Plot 1: First microphone signal comparison
    ax = axes[0]
    time_axis = np.arange(min_length) / fs
    ax.plot(time_axis[:1000], batch_truncated[0, :1000], label="Batch", alpha=0.7)
    ax.plot(
        time_axis[:1000],
        streaming_truncated[0, :1000],
        "--",
        label="Streaming",
        alpha=0.7,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Microphone 1 Output (first 1000 samples)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Error over time
    ax = axes[1]
    ax.plot(time_axis, np.abs(difference[0]), label="Mic 1")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Absolute Error")
    ax.set_title("Difference Between Batch and Streaming")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Room layout
    ax = axes[2]
    room_batch.plot(img_order=0, ax=ax)
    ax.set_title("Room Layout (Top View)")

    plt.tight_layout()

    try:
        plt.savefig("streaming_simulation_results.png", dpi=150)
        print("  Saved: streaming_simulation_results.png")
    except Exception as e:
        print(f"  Could not save figure: {e}")

    plt.show()

    # =================================================================
    # Performance Statistics
    # =================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total samples processed: {n_samples:,}")
    print(f"Blocks processed: {n_blocks}")
    print(f"Latency per block: {block_size/fs*1000:.1f} ms")
    total_latency_ms = block_size / fs * 1000 * 2  # input buffer + output buffer
    print(f"Total algorithmic latency: {total_latency_ms:.1f} ms")
    print(f"RIR length: {rir_length} samples ({rir_length/fs*1000:.1f} ms)")
    print(f"Verification: Max error = {max_error:.2e}")
    print("=" * 70)


if __name__ == "__main__":
    main()
