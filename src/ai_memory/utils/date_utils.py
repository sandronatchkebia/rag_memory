"""Date and time utility functions."""

from datetime import datetime, date, timezone
from typing import Union, Optional
import re


def parse_date(date_string: str, platform: Optional[str] = None) -> datetime:
    """Parse date strings from various platforms."""
    if not date_string:
        return datetime.now(timezone.utc)
    
    # Try ISO format first (most common)
    try:
        # Handle ISO format with Z or +00:00
        if date_string.endswith('Z'):
            date_string = date_string[:-1] + '+00:00'
        elif '+' in date_string and ':' in date_string.split('+')[1]:
            # Already has timezone info
            pass
        else:
            # Assume UTC if no timezone info
            date_string = date_string + '+00:00'
        
        return datetime.fromisoformat(date_string)
    except ValueError:
        pass
    
    # Try Unix timestamp
    try:
        if date_string.isdigit():
            timestamp = int(date_string)
            # Assume seconds if reasonable, milliseconds if large
            if timestamp > 1000000000000:  # After year 2000 in milliseconds
                timestamp = timestamp / 1000
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    except (ValueError, OSError):
        pass
    
    # Try common email formats
    email_formats = [
        "%a, %d %b %Y %H:%M:%S %z",  # Mon, 21 Aug 2023 14:48:00 +0000
        "%d %b %Y %H:%M:%S %z",      # 21 Aug 2023 14:48:00 +0000
        "%a, %d %b %Y %H:%M:%S",     # Mon, 21 Aug 2023 14:48:00
        "%d %b %Y %H:%M:%S",         # 21 Aug 2023 14:48:00
    ]
    
    for fmt in email_formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    
    # Try date-only formats
    date_formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%Y/%m/%d"
    ]
    
    for fmt in date_formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    
    # If all else fails, try to extract date components with regex
    date_match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', date_string)
    if date_match:
        year, month, day = map(int, date_match.groups())
        try:
            return datetime(year, month, day, tzinfo=timezone.utc)
        except ValueError:
            pass
    
    # Last resort: return current time
    print(f"Warning: Could not parse date '{date_string}', using current time")
    return datetime.now(timezone.utc)


def format_date(dt: Union[datetime, date], format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format a date/datetime object to string."""
    if isinstance(dt, date) and not isinstance(dt, datetime):
        dt = datetime.combine(dt, datetime.min.time())
    
    return dt.strftime(format_str)


def extract_date_from_text(text: str) -> Optional[datetime]:
    """Extract date information from text content."""
    if not text:
        return None
    
    # Common date patterns in text
    patterns = [
        # "yesterday", "today", "tomorrow"
        (r'\b(yesterday|today|tomorrow)\b', {
            'yesterday': -1,
            'today': 0,
            'tomorrow': 1
        }),
        
        # "last week", "next month", etc.
        (r'\b(last|next)\s+(week|month|year|day)\b', {
            'last': -1,
            'next': 1
        }),
        
        # "3 days ago", "2 weeks ago"
        (r'(\d+)\s+(day|week|month|year)s?\s+ago', {}),
        
        # "in 3 days", "in 2 weeks"
        (r'in\s+(\d+)\s+(day|week|month|year)s?', {}),
        
        # Date ranges like "last year" or "this month"
        (r'\b(this|last|next)\s+(year|month|week|day)\b', {})
    ]
    
    from datetime import timedelta
    
    for pattern, replacements in patterns:
        match = re.search(pattern, text.lower())
        if match:
            if 'yesterday' in pattern:
                return datetime.now(timezone.utc) + timedelta(days=replacements[match.group(1)])
            elif 'today' in pattern:
                return datetime.now(timezone.utc)
            elif 'tomorrow' in pattern:
                return datetime.now(timezone.utc) + timedelta(days=1)
            elif 'ago' in pattern:
                count = int(match.group(1))
                unit = match.group(2)
                if unit == 'day':
                    return datetime.now(timezone.utc) - timedelta(days=count)
                elif unit == 'week':
                    return datetime.now(timezone.utc) - timedelta(weeks=count)
                elif unit == 'month':
                    return datetime.now(timezone.utc) - timedelta(days=count*30)
                elif unit == 'year':
                    return datetime.now(timezone.utc) - timedelta(days=count*365)
            elif 'in' in pattern:
                count = int(match.group(1))
                unit = match.group(2)
                if unit == 'day':
                    return datetime.now(timezone.utc) + timedelta(days=count)
                elif unit == 'week':
                    return datetime.now(timezone.utc) + timedelta(weeks=count)
                elif unit == 'month':
                    return datetime.now(timezone.utc) + timedelta(days=count*30)
                elif unit == 'year':
                    return datetime.now(timezone.utc) + timedelta(days=count*365)
    
    return None


def is_recent(dt: datetime, days: int = 7) -> bool:
    """Check if a date is within recent days."""
    now = datetime.now(timezone.utc)
    delta = now - dt
    return delta.days <= days


def get_time_ago(dt: datetime) -> str:
    """Get human-readable time ago string."""
    now = datetime.now(timezone.utc)
    delta = now - dt
    
    if delta.days > 0:
        if delta.days == 1:
            return "yesterday"
        elif delta.days < 7:
            return f"{delta.days} days ago"
        elif delta.days < 30:
            weeks = delta.days // 7
            return f"{weeks} week{'s' if weeks > 1 else ''} ago"
        elif delta.days < 365:
            months = delta.days // 30
            return f"{months} month{'s' if months > 1 else ''} ago"
        else:
            years = delta.days // 365
            return f"{years} year{'s' if years > 1 else ''} ago"
    elif delta.seconds > 0:
        if delta.seconds < 60:
            return "just now"
        elif delta.seconds < 3600:
            minutes = delta.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            hours = delta.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
    else:
        return "just now"


def get_date_range_description(start_date: datetime, end_date: datetime) -> str:
    """Get human-readable description of a date range."""
    if start_date.date() == end_date.date():
        return f"on {format_date(start_date, '%B %d, %Y')}"
    
    start_year = start_date.year
    end_year = end_date.year
    
    if start_year == end_year:
        if start_date.month == end_date.month:
            return f"from {start_date.day} to {end_date.day} {format_date(start_date, '%B %Y')}"
        else:
            return f"from {format_date(start_date, '%B %d')} to {format_date(end_date, '%B %d, %Y')}"
    else:
        return f"from {format_date(start_date, '%B %d, %Y')} to {format_date(end_date, '%B %d, %Y')}"


def is_same_day(dt1: datetime, dt2: datetime) -> bool:
    """Check if two datetimes are on the same day."""
    return dt1.date() == dt2.date()


def is_same_week(dt1: datetime, dt2: datetime) -> bool:
    """Check if two datetimes are in the same week."""
    # Get ISO week numbers
    week1 = dt1.isocalendar()[1]
    week2 = dt2.isocalendar()[1]
    year1 = dt1.isocalendar()[0]
    year2 = dt2.isocalendar()[0]
    
    return year1 == year2 and week1 == week2


def is_same_month(dt1: datetime, dt2: datetime) -> bool:
    """Check if two datetimes are in the same month."""
    return dt1.year == dt2.year and dt1.month == dt2.month
