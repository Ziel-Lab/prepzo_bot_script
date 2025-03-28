VERSION = "1.0.0"
BUILD_DATE = "2025-03-28"

def get_version_info():
    """Return version information as a dictionary"""
    import os
    return {
        "version": VERSION,
        "build_date": BUILD_DATE,
        "git_commit": os.environ.get("GIT_COMMIT", "unknown")
    } 