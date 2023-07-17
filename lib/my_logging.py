import logging
import sys
import coloredlogs

########################################
#               LOGGING                #
########################################
def createLogger(logname):
    logger = logging.getLogger(logname)
    coloredlogs.install(logger=logger)
    logger.propagate = False

    coloredFormatter = coloredlogs.ColoredFormatter(
        fmt='%(asctime)s %(funcName)s L%(lineno)-3d %(message)s',
        level_styles=dict(
            debug=dict(color='white'),
            info=dict(color='blue'),
            warning=dict(color='yellow', bright=True),
            error=dict(color='red', bold=True, bright=True),
            critical=dict(color='black', bold=True, background='red'),
        ),
        field_styles=dict(
            name=dict(color='white'),
            asctime=dict(color='red'),
            funcName=dict(color='blue'),
            lineno=dict(color='white'),
        )
    )


    logger.setLevel(level=logging.DEBUG)
    logStreamFormatter = logging.Formatter(
      fmt=f"%(levelname)-8s %(asctime)s \t %(filename)s @function %(funcName)s line %(lineno)s - %(message)s",
      datefmt="%H:%M:%S"
    )
    consoleHandler = logging.StreamHandler(stream=sys.stdout)
    consoleHandler.setFormatter(fmt=coloredFormatter)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.addHandler(hdlr=consoleHandler)
    logger.setLevel(level=logging.DEBUG)

    logFileFormatter = logging.Formatter(
        fmt=f"%(levelname)s %(asctime)s \t %(funcName)s L%(lineno)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fileHandler = logging.FileHandler(filename=logname)
    fileHandler.setFormatter(logFileFormatter)
    fileHandler.setLevel(level=logging.INFO)

    logger.addHandler(fileHandler)

    return logger
