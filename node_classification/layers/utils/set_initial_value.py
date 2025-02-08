import torch.nn as nn

def set_initial_value(param, init_method, a=0, mode='fan_in', nonlinearity='relu'):
    if init_method == 'signed_kaiming_constant':
        fan = nn.init._calculate_correct_fan(param, mode)
        gain = nn.init.calculate_gain(nonlinearity)
        std = gain / (fan ** (1/2))
        param.data = std * param.data.sign()

    elif init_method == 'kaiming_normal':
        nn.init.kaiming_normal_(param, a=a, mode=mode, nonlinearity=nonlinearity)
    elif init_method == 'kaiming_uniform':
        nn.init.kaiming_uniform_(param, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        raise NotImplementedError