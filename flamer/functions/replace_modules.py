def replace_modules(subject, search, replace, channels_iter=None):
    for name, child in subject.named_children():
        if isinstance(child, search):
            if channels_iter:
                new_activation = replace(next(channels_iter))
            else:
                new_activation = replace()
            setattr(subject, name, new_activation)
        else:
            replace_modules(child, search, replace, channels_iter)
