"""Speech synthesis and speaker playback for the SyncAI robot.

Split out of ``syncai_backend``'s in-process ``TtsGateway`` so that exactly one
process owns the speaker. See README.md for why that ownership is the whole
point of the split.
"""

__version__ = "0.1.0"
