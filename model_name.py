def name_denoise_model(arch, nm, l, lr, ps, sf, n, nb, attn, r, d, rs, es):
    name = arch
    name += '_nm' if nm else ''
    name += f'_{l}'
    name += f'_lr{lr:.0e}'
    name += f'_ps{ps}'
    name += f'_sf{sf}'
    name += f'_n{n}'
    name += f'_nb{nb}'
    name += f'_{attn}{f'(r{r})' if attn in ['CBAM', 'Channel'] else ''}'
    name += f'_{attn}' if attn == 'Spatial' else ''
    name += f'_d{d}' if d > 1 else ''
    name += f'_rs{rs}' if rs != 1 else ''
    name += '_es' if es else ''
    return name

def name_superres_model(arch, l, lr, ps, sf, nb, es):
    name = arch
    name += f'_{l}'
    name += f'_lr{lr:.0e}'
    name += f'_ps{ps}'
    name += f'_sf{sf}'
    name += f'_nb{nb}'
    name += '_es' if es else ''
    return name

def name_estimator_model(ps, sf):
    name = f'estimator'
    name += f'_ps{ps}'
    name += f'_sf{sf}'
    return name