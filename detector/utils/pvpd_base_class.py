import logging
from abc import ABC

from detector.utils.exit_code import ExitCode


class PVPDBaseClass(ABC):
    @classmethod
    def name(cls):
        return str(cls.__name__)

    @classmethod
    def get_all_subclass(cls):
        return [klass.name() for klass in cls.__subclasses__()]

    @classmethod
    def get_subclass_by_name(cls, name):
        subclasses = cls.__subclasses__()
        config = [x for x in subclasses if x.name() == name]
        if len(config) != 1:
            logging.fatal(
                f"{str(cls.__name__)} name needs to be unique. Wanted to use config with name '{name}'. But received {config}")
            exit(ExitCode.PVPDBaseClassNotUnique)
        return config[0]
