from contextlib import suppress
import mindspore as ms


def get_autocast(precision, model):
    if precision == 'amp':
        # MindSpore默认自动混合精度上下文
        return ms.amp.auto_mixed_precision(model)
    elif precision in ('amp_bfloat16', 'amp_bf16'):
        # MindSpore暂不支持bfloat16自动混合精度，可回退到float16
        return ms.amp.auto_mixed_precision(model)
    else:
        return suppress