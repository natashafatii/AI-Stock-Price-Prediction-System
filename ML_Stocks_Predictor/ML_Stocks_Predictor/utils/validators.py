

import re

from datetime import datetime

from typing import Tuple, Optional

class ValidationError(Exception):

    pass

def validate_ticker(ticker: str) -> str:

    if not ticker:

        raise ValidationError("Ticker symbol is required")

    ticker = ticker.strip().upper()

    if not re.match(r'^[A-Z0-9.\-]{1,10}$', ticker):

        raise ValidationError(

            f"Invalid ticker format: '{ticker}'. "

            "Ticker should contain only letters, numbers, dots, and hyphens (max 10 characters)"

        )

    return ticker

def validate_date(date_str: str) -> Tuple[bool, Optional[datetime], Optional[str]]:

    if not date_str:

        return False, None, "Date is required"

    try:

        date_obj = datetime.strptime(date_str.strip(), '%Y-%m-%d')

        if date_obj > datetime.now():

            return False, None, "Date cannot be in the future"

        if date_obj.year < 2000:

            return False, None, "Date should be after year 2000"

        return True, date_obj, None

    except ValueError:

        return False, None, f"Invalid date format: '{date_str}'. Expected format: YYYY-MM-DD"

def validate_date_range(start_date: str, end_date: str) -> Tuple[bool, Optional[str]]:

    is_valid_start, start_obj, start_error = validate_date(start_date)

    if not is_valid_start:

        return False, f"Start date error: {start_error}"

    is_valid_end, end_obj, end_error = validate_date(end_date)

    if not is_valid_end:

        return False, f"End date error: {end_error}"

    if start_obj >= end_obj:

        return False, "Start date must be before end date"

    days_diff = (end_obj - start_obj).days

    if days_diff < 30:

        return False, f"Date range too short ({days_diff} days). Need at least 30 days of data"

    return True, None

def sanitize_input(text: str, max_length: int = 100) -> str:

    if not text:

        return ""

    text = text.strip()

    if len(text) > max_length:

        text = text[:max_length]

    text = re.sub(r'[<>\"\'%;()&+]', '', text)

    return text

