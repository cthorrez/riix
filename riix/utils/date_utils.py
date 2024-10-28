import re

PROG = re.compile(r'(\d+)([WwDdHhMmSs])')
SECONDS_PER_UNIT = {
    'W': 7 * 24 * 60 * 60,    # weeks to seconds
    'D': 24 * 60 * 60,        # days to seconds
    'H': 60 * 60,             # hours to seconds
    'M': 60,                  # minutes to seconds
    'S': 1                    # seconds
}

def get_duration(duration_str):
    """
    Parse duration strings like '7D', '1W', '24H' etc. into the number os seconds since epoch as an int
    
    Parameters:
    -----------
    duration_str : str
        String in format numberLetter where Letter is one of:
        W/w - weeks
        D/d - days
        H/h - hours
        M/m - minutes
        S/s - seconds
        
    Returns:
    --------
    duration : int
    """
    # Extract number and unit using regex
    match = PROG.match(duration_str)
    if not match:
        raise ValueError(f"Invalid duration format: {duration_str}")
    number = int(match.group(1))
    unit = match.group(2).upper()
    duration = int(number * SECONDS_PER_UNIT[unit])
    return duration