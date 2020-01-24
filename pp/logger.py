"""
This module provides a fairly sophisticated logging system as an alternative to ``print``. You can use ``LOGGER.info(text)`` just like ``print``. However, there are some real advantages.

Rationale
----------

Firstly, ``LOGGER.info()`` doesn't just print to the terminal. It also writes more detailed logs to a file, usually in ``build/logs/main.log``. This gives a permanent record of warnings and errors, which is also synced from Jenkins into Dropbox.  While output on the terminal is simple::

    >>> from pp.logger import LOGGER
    >>> LOGGER.info("Hello world")
    Hello world

The output in the file is more sophisticated:

.. code-block:: none

    2017-11-08 04:05:08,481 [INFO ] [<ipython-input-2-9487df49f71a>:<module>:1] Hello world

Let's break down the structure of this log entry:

=============================================  =================================================================================================================================================================
Part                                           Meaning
=============================================  =================================================================================================================================================================
2017-11-08 04:05:08,481                        A timestamp
[INFO]                                         The log-level, it can be any of ``CRITICAL``, ``ERROR``, ``WARNING``, ``INFO``, ``DEBUG``, ``NOTSET``.
<ipython-input-2-9487df49f71a>:<module>:1      The module, function and line number which emitted this log entry. Really useful when somebody has added annoying terminal output to a complicated codebase.
Hello world                                    The text message
=============================================  =================================================================================================================================================================

Everyday usage
---------------

In 99% of cases all you need to do is this::

    from pp import CONFIG, LOGGER

    ... code ...

    LOGGER.info("Hello world")
    LOGGER.warning("Something bad is happening")

If you want to emit an error or a warning, call ``LOGGER.error("Some text")`` or ``LOGGER.warning("Some text")``. If you want something to be logged but not printed to the terminal, use ``LOGGER.debug``.

gdsdactory stores separate log files for each mask. You can easily review the logs just by reading the files, but you can also use ``pf log show``::

    $ pf log show

It can also post logs to Slack, which can be useful when running on Jenkins or a remote server. Just add a block like this to the local ``~/.gdsfactory/config.yml``:

.. code-block:: yaml

    slack:
      enabled: true
      channel: '<channel>'
      hook: 'https://hooks.slack.com/services/<secret>'

"""


import traceback
import logging
import sys
from os.path import join
from pp.config import CONFIG

if CONFIG.get("slack"):
    import slack


ERROR_COLOR = "#ff0000"  # color name is built in to Slack API
WARNING_COLOR = "warning"  # color name is built in to Slack API
INFO_COLOR = "#439FE0"
DEFAULT_EMOJI = ":godmode:"

COLORS = {
    logging.CRITICAL: ERROR_COLOR,
    logging.FATAL: ERROR_COLOR,
    logging.ERROR: ERROR_COLOR,
    logging.WARNING: WARNING_COLOR,
    logging.INFO: INFO_COLOR,
    logging.DEBUG: INFO_COLOR,
    logging.NOTSET: INFO_COLOR,
}


class SlackHandler(logging.Handler):
    """ A handler which tries to post to Slack """

    def __init__(self):
        logging.Handler.__init__(self)

    def emit(self, record):
        """ Emit a message """
        message = str(record.getMessage())
        if hasattr(record, "raw"):  # The raw flag was set, we need to handle specially
            fallback = message[:10] + "..."
            slack.post_log(
                message="Trace:", data=message, fallback=fallback, level=record.levelno
            )
        elif record.exc_info:  # There is a stack trace, let's show it
            exc = "\n".join(traceback.format_exception(*record.exc_info))
            fallback = str(type(record.exc_info[0]))
            slack.post_log(
                message=message, data=exc, fallback=fallback, level=record.levelno
            )
        else:  # It's just an INFO or something
            slack.post_log(message=message, level=record.levelno)


def get_filehandler(name, level):
    """ Get a filehandler """
    logdir = CONFIG["log_directory"]
    fh = logging.FileHandler(join(logdir, name + ".log"))
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)-5.5s] [%(module)s:%(funcName)s:%(lineno)d] %(message)s"
        )
    )
    fh.setLevel(level)
    return fh


def get_streamhandler(level=logging.INFO):
    """ Get a streamhandler """
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(message)s"))
    sh.setLevel(level)
    return sh


def get_slackhandler(level=logging.INFO):
    """ Get a Slack handler """
    sh = SlackHandler()
    sh.setLevel(level)
    return sh


def get_logger(name):
    """ Get a complicated logger """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Add a file handler for all messages (including debug messages)
    logger.addHandler(get_filehandler("debug", logging.DEBUG))
    logger.addHandler(get_filehandler(name, logging.INFO))
    logger.addHandler(get_filehandler("error", logging.ERROR))
    logger.addHandler(get_filehandler("failed_to_pack", logging.WARNING))

    # Add a console handler
    logger.addHandler(get_streamhandler())

    # Try to add a Slack handler
    if CONFIG.get("slack"):
        logger.addHandler(get_slackhandler())
    return logger


def handle_exception(exc_type, exc_value, exc_traceback):
    """ This lets us catch errors just before Python dies """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    LOGGER.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


if not __name__ == "__main__":
    LOGGER = get_logger("main")
    sys.excepthook = handle_exception
else:
    LOGGER2 = get_logger("test")
    LOGGER2.debug("Nothing")
    LOGGER2.info("Something")
    LOGGER2.error("Simple error")

    LOGGER2.info("I'm about to show some code:")
    LOGGER2.info("here\nis\nsome\ncode", extra={"raw": True})

    LOGGER2.error("Zero div error", extra={"raw": True})

    try:
        1 / 0
    except ZeroDivisionError as e:
        LOGGER2.error("Zero div error", exc_info=e)

    LOGGER2.warning("Simple warning")
