from enum import IntEnum, auto


class ExitCode(IntEnum):
    ImagesNotFound = auto()
    PVPDBaseClassNotUnique = auto()
