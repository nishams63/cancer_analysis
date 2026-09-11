import datetime
import random
import string

def generate_batch_id(prefix: str = 'BATCH') -> str:
    date_str = datetime.datetime.utcnow().strftime('%Y%m%d')
    suffix = ''.join(random.choices(string.digits, k=4))
    return f'{prefix}-{date_str}-{suffix}'

def generate_scenario_id(index: int, prefix: str = 'SYN-S') -> str:
    return f'{prefix}{index:03d}'

def generate_run_id(prefix: str = 'RUN') -> str:
    date_str = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
    suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f'{prefix}-{date_str}-{suffix}'
