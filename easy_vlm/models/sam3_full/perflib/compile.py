import torch


def recursive_fn_factory(fn):
    def recursive_fn(value):
        if isinstance(value, dict):
            return {k: recursive_fn(v) for k, v in value.items()}
        if isinstance(value, list):
            return [recursive_fn(v) for v in value]
        if isinstance(value, tuple):
            return tuple(recursive_fn(v) for v in value)
        if isinstance(value, torch.Tensor):
            return fn(value)
        if value is None or isinstance(value, (bool, int)):
            return value
        raise TypeError(f"Unexpected type {type(value)}")

    return recursive_fn


recursive_contiguous = recursive_fn_factory(lambda x: x.contiguous())
recursive_clone = recursive_fn_factory(torch.clone)


def compile_wrapper(
    fn, *, mode="max-autotune", fullgraph=True, dynamic=False, name=None
):
    compiled_fn = torch.compile(fn, mode=mode, fullgraph=fullgraph, dynamic=dynamic)

    def compiled_fn_wrapper(*args, **kwargs):
        with torch.autograd.profiler.record_function(
            f"compiled {fn}" if name is None else name
        ):
            cont_args = recursive_contiguous(args)
            cont_kwargs = recursive_contiguous(kwargs)
            result = compiled_fn(*cont_args, **cont_kwargs)
            return recursive_clone(result)

    return compiled_fn_wrapper


def shape_logging_wrapper(fn, keep_kwargs, enable_logging=False):
    seen_shapes = set()

    def get_shape(obj):
        if isinstance(obj, torch.Tensor):
            return obj.shape
        if isinstance(obj, (list, tuple)):
            if len(obj) > 1:
                return tuple(get_shape(x) for x in obj)
            return get_shape(obj[0])
        if isinstance(obj, dict):
            return tuple(sorted((k, get_shape(v)) for k, v in obj.items()))
        return type(obj).__name__

    def wrapper(*args, **kwargs):
        shapes = tuple(get_shape(arg) for arg in args) + tuple(
            (k, get_shape(v))
            for k, v in kwargs.items()
            if isinstance(v, (torch.Tensor, list))
            and (len(keep_kwargs) > 0 and k in keep_kwargs)
        )
        if shapes not in seen_shapes:
            seen_shapes.add(shapes)
            if enable_logging:
                print(f"[ShapeLogger] New input shapes for {fn.__qualname__}: {shapes}")
        return fn(*args, **kwargs)

    wrapper.enable_logging = enable_logging

    def set_logging(enabled=False):
        nonlocal enable_logging
        enable_logging = enabled
        wrapper.enable_logging = enable_logging

    wrapper.set_logging = set_logging
    return wrapper
