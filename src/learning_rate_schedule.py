def learning_rate_scheduler(epoch):
    if epoch > 20:
        lr = 1e-3
    elif epoch > 30:
        lr = 1e-4
    elif epoch > 50:
        lr = 5e-5
    elif epoch > 100:
        lr = 1e-5
    elif epoch > 200:
        lr = 1e-6
    else:
        lr = 3e-2
    return lr