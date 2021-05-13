def group_norm(num_groups=32):
    from torch.nn import GroupNorm as _GroupNorm

    class GroupNorm(_GroupNorm):
        def __init__(self, num_channels, *args, **kwargs):
            super().__init__(num_groups, num_channels, *args, **kwargs)

    return GroupNorm
