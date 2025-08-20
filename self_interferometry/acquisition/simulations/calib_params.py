from dataclasses import dataclass


@dataclass
class CalibrationParameters:
    """Calibration parameters for the coil driver.

    Attributes:
        f0: Resonant frequency in Hz
        Q: Quality factor
        k: Gain factor
        c: Phase offset
        speaker_part_number: Optional part number of the speaker
    """

    f0: float = 257.20857316296724
    Q: float = 15.804110908084784
    k: float = 33.42493417407945
    c: float = -3.208233068626455
    speaker_part_number: str | None = None
