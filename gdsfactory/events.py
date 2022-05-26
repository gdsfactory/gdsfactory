###################################################################################################################
# PROPRIETARY AND CONFIDENTIAL
# THIS SOFTWARE IS THE SOLE PROPERTY AND COPYRIGHT (c) 2021 OF ROCKLEY PHOTONICS LTD.
# USE OR REPRODUCTION IN PART OR AS A WHOLE WITHOUT THE WRITTEN AGREEMENT OF ROCKLEY PHOTONICS LTD IS PROHIBITED.
# RPLTD NOTICE VERSION: 1.1.1
###################################################################################################################
from typing import Callable


class Event(object):
    def __init__(self):
        self._handlers = []

    def __iadd__(self, handler: Callable):
        self.add_handler(handler)
        return self

    def __isub__(self, handler: Callable):
        self._handlers.remove(handler)
        return self

    def add_handler(self, handler: Callable):
        """
        Adds a handler that will be executed when this event is fired.

        Args:
            handler: a function which matches the signature fired by this event
        """
        self._handlers.append(handler)

    def clear_handlers(self):
        self._handlers.clear()

    def fire(self, *args, **kwargs):
        """
        Fires an event, calling all handlers with the passed arguments.
        """
        for eventhandler in self._handlers:
            eventhandler(*args, **kwargs)
