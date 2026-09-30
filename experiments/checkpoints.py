"""Shared validation for experiment checkpoint inputs."""
from pathlib import Path


def require_checkpoint_files(paths):
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError('Missing required checkpoint(s):\n' + '\n'.join(missing))


def copy_variant_state(target, source, prefixes):
    """Copy complete selected modules; missing weights must not stay random."""
    for prefix in prefixes:
        keys = [key for key in target if key.startswith(prefix + '.')]
        if not keys:
            raise ValueError(f'No model parameters match {prefix}')
        missing = [key for key in keys if key not in source]
        if missing:
            raise ValueError(f'Checkpoint lacks parameters for {prefix}: {missing}')
        for key in keys:
            if target[key].shape != source[key].shape:
                raise ValueError(f'Checkpoint shape mismatch for {key}')
            target[key] = source[key]
