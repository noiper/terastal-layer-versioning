"""Stable layer indices used by the historical VGG checkpoints."""
LAYERS = ('features.8', 'features.16', 'features.18', 'classifier.0', 'classifier.3')


def selected_layers(indices):
    if not indices or len(set(indices)) != len(indices):
        raise ValueError('Select at least one layer index, without duplicates')
    if any(index not in range(len(LAYERS)) for index in indices):
        raise ValueError('VGG layer indices must be between 0 and 4')
    return {LAYERS[index]: {'idx': index, 'r': 2} for index in sorted(indices)}
