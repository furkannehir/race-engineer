# IR-001 blue-flag regression

This privacy-safe fixture is reduced from the M3 live validation transition. At source
sequence 3732, iRacing's `SessionFlags` gains the blue bit immediately after green while
the player is unclassified and no opponent has completed an additional lap. Sequence 3738
shows that the bit must remain suppressed after classification when every active car is
still on the same completed lap.

The final synthetic control frame retains the blue bit but places an active, non-pace-car
opponent one completed lap ahead. It proves the normalizer still exposes a genuine race
blue condition and allows the deterministic flag event to be derived.
