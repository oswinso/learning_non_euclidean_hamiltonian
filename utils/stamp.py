from datetime import datetime
from typing import Optional


def get_date_stamp(now: Optional[datetime] = None) -> str:
    if now is None:
        now = datetime.now()
    return now.strftime("%Y%m%d")


def get_time_stamp(now: Optional[datetime] = None) -> str:
    if now is None:
        now = datetime.now()
    return now.strftime("%H%M%S")


def get_datetime_stamp(now: Optional[datetime] = None) -> str:
    if now is None:
        now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")
