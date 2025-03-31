"""
Version information for Prepzo Bot
"""

VERSION = "1.0.0"
BUILD_DATE = "2023-10-10"
GIT_COMMIT = "unknown"  # Will be updated during deployment

def get_version_info():
    """Return version information as a dictionary"""
    return {
        "version": VERSION,
        "build_date": BUILD_DATE,
        "git_commit": GIT_COMMIT
    } 