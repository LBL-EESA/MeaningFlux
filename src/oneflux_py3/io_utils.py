import pandas as pd

AMF_COL_MAP = {
    'NEE': ['NEE', 'FC', 'NEE_F', 'FC_F'],
    'TA': ['TA', 'TA_F', 'TA_1_1_1', 'TA_ERA'],
    'VPD': ['VPD', 'VPD_F', 'VPD_ERA'],
    'RG':  ['SW_IN', 'SW_IN_F', 'SW_IN_ERA', 'RG', 'Rg', 'SW_IN_POT'],
}

def first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the candidate columns exist: {candidates}")

def load_amf_like_csv(path):
    df = pd.read_csv(path)
    tcol = None
    for cand in ['TIMESTAMP', 'TIMESTAMP_START', 'time', 'datetime']:
        if cand in df.columns:
            tcol = cand
            break
    if tcol is None:
        raise KeyError('No timestamp column found. Expected TIMESTAMP(_START) or time/datetime.')
    # Parse timestamp flexibly
    try:
        dt = pd.to_datetime(df[tcol], errors='coerce')
    except Exception:
        dt = pd.to_datetime(df[tcol].astype(str), errors='coerce')
    df.index = dt
    df.drop(columns=[tcol], inplace=True)
    return df
