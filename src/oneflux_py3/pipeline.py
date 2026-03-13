import pandas as pd
from .mds_gapfill import mds_fill_nee, mds_fill_met
from .partition_nt import partition_nee_nighttime

def process_site(df: pd.DataFrame,
                 nee_col='NEE', ta_col='TA', vpd_col='VPD', rg_col='RG',
                 do_partition=True):
    """Convenience pipeline:
    - fills TA/VPD/RG via MDC/climatology
    - fills NEE via MDS
    - partitions NEE (nighttime) to Reco & GPP if requested
    Returns a new DataFrame with *_f and partition outputs.
    """
    out = df.copy()
    out[f'{ta_col}_f']  = mds_fill_met(out[ta_col])
    out[f'{vpd_col}_f'] = mds_fill_met(out[vpd_col])
    out[f'{rg_col}_f']  = mds_fill_met(out[rg_col])
    out[f'{nee_col}_f'] = mds_fill_nee(out[nee_col], out[f'{rg_col}_f'], out[f'{ta_col}_f'], out[f'{vpd_col}_f'])
    if do_partition:
        part = partition_nee_nighttime(out[f'{nee_col}_f'], out[f'{ta_col}_f'], out[f'{rg_col}_f'])
        out['Reco_nt'] = part['Reco']
        out['GPP_nt']  = part['GPP']
        out['Rref']    = part['Rref']
        out['E0']      = part['E0']
        out['qf_partition_nt'] = part['qf_partition_nt']
    return out
