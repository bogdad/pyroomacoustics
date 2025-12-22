"""
Streaming audio processing utilities for real-time room simulation.

This module provides block-based convolution for streaming RIR application.
"""

import numpy as np
from scipy.fftpack import fft, ifft


class PartitionedConvolver:
    """
    Streaming convolution using Overlap-Save (OLS) algorithm.

    This class performs block-based convolution of an input signal with
    a fixed impulse response (e.g., Room Impulse Response). It maintains
    internal state to ensure seamless processing across blocks.

    The Overlap-Save algorithm:
    1. Prepends previous samples (overlap region) to current block
    2. Performs FFT-based convolution
    3. Discards aliased samples, keeps valid output
    4. Stores samples for next block's overlap

    Attributes
    ----------
    block_size : int
        Number of samples to process per block
    rir_length : int
        Length of the impulse response in samples
    fft_size : int
        FFT size used for frequency-domain convolution (power of 2)
    H : ndarray
        Frequency-domain representation of impulse response
    state : ndarray
        Buffer storing last (rir_length - 1) samples for overlap

    Examples
    --------
    >>> # Create a simple impulse response
    >>> rir = np.array([1.0, 0.5, 0.25])
    >>> convolver = PartitionedConvolver(rir, block_size=128)
    >>>
    >>> # Process audio block-by-block
    >>> input_block = np.random.randn(128)
    >>> output_block = convolver.process_block(input_block)
    >>>
    >>> # Reset state for a new stream
    >>> convolver.reset()

    References
    ----------
    Overlap-Save method described in:
    Oppenheim, A. V., & Schafer, R. W. (2009). Discrete-time signal processing.
    """

    def __init__(self, impulse_response, block_size):
        """
        Initialize the partitioned convolver.

        Parameters
        ----------
        impulse_response : array_like
            The impulse response (e.g., RIR) to convolve with input signal.
            Can be 1D array of any length.
        block_size : int
            Number of samples to process per block. Must be positive.

        Raises
        ------
        ValueError
            If block_size is not positive or impulse_response is empty
        """
        impulse_response = np.asarray(impulse_response)

        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        if len(impulse_response) == 0:
            raise ValueError("impulse_response cannot be empty")

        self.block_size = block_size
        self.rir_length = len(impulse_response)

        # FFT size must be at least block_size + rir_length - 1 for linear convolution
        # Round up to next power of 2 for efficiency
        min_fft_size = block_size + self.rir_length - 1
        self.fft_size = 2 ** int(np.ceil(np.log2(min_fft_size)))

        # Pre-compute FFT of impulse response (done once for efficiency)
        self.H = fft(impulse_response, n=self.fft_size)

        # State buffer: stores last (rir_length - 1) samples for overlap
        # This handles the "tail" of the convolution that extends into the next block
        self.state = np.zeros(self.rir_length - 1)

    def process_block(self, input_block):
        """
        Process one block of audio through the convolution.

        Parameters
        ----------
        input_block : array_like
            Input audio block of length block_size

        Returns
        -------
        output_block : ndarray
            Convolved output of length block_size

        Raises
        ------
        ValueError
            If input_block length doesn't match block_size

        Notes
        -----
        This method maintains internal state, so consecutive calls will
        produce a continuous output stream equivalent to convolving the
        entire concatenated input with the impulse response.
        """
        input_block = np.asarray(input_block)

        if len(input_block) != self.block_size:
            raise ValueError(
                f"Expected input block of size {self.block_size}, "
                f"got {len(input_block)}"
            )

        # Overlap-Save algorithm:
        # 1. Prepend state from previous block (overlap region)
        x_padded = np.concatenate([self.state, input_block])

        # 2. Transform to frequency domain and apply filter
        X = fft(x_padded, n=self.fft_size)
        Y = X * self.H

        # 3. Transform back to time domain
        y_padded = np.real(ifft(Y))

        # 4. Extract valid output (discard first rir_length-1 aliased samples)
        # The aliased region corresponds to the circular convolution artifact
        output_start = self.rir_length - 1
        output_end = output_start + self.block_size
        output_block = y_padded[output_start:output_end]

        # 5. Update state for next block
        # Store the last (rir_length - 1) samples from the padded input
        # This maintains continuity for the next block's overlap region
        if self.rir_length > 1:
            self.state = x_padded[-(self.rir_length - 1):].copy()

        return output_block

    def reset(self):
        """
        Reset the internal state buffer to zeros.

        Call this when starting to process a new independent audio stream
        to avoid contamination from previous processing.
        """
        self.state.fill(0)

    def get_latency(self):
        """
        Get the algorithmic latency in samples.

        Returns
        -------
        latency : int
            Number of samples of delay introduced by block processing.
            This equals block_size for Overlap-Save method.

        Notes
        -----
        The total system latency includes:
        - Algorithmic latency (this value)
        - Input buffering (block_size samples)
        - Processing time (depends on hardware)
        """
        return self.block_size

    def __repr__(self):
        return (
            f"PartitionedConvolver(rir_length={self.rir_length}, "
            f"block_size={self.block_size}, fft_size={self.fft_size})"
        )
