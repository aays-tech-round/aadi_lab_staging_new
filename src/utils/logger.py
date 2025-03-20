import logging

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def log(message: str, title: str = "Info"):
    """
    Logs an informational message.

    Args:
        message (str): The message to log.
        title (str, optional): A custom title for the log. Defaults to "Info".
    
    Example:
        log("Application started")
        log("Data processing completed", title="Processing")
    """
    logger.info(f"{title}: {message}")

def logError(message: str, title: str = "Error"):
    """
    Logs an error message.

    Args:
        message (str): The error message to log.
        title (str, optional): A custom title for the log. Defaults to "Error".
    
    Example:
        logError("File not found")
        logError("Database connection failed", title="DatabaseError")
    """
    logger.error(f"{title}: {message}")
