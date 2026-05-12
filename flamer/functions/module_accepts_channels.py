import inspect


def module_accepts_channels(module):
    arguments = inspect.signature(module).args
    return "channels" in arguments or "in_channels" in arguments
