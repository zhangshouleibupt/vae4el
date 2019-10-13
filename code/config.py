Config = dict(
    voc_size = 50000,
    encoder_layers = 4,
    encoder_hidden_dim = 768,
    vae_hidden_dim = 128,
    vae_layers = 4,
    vae_decoder_hidden_dim = 128,
    vae_decoder_layers = 4,
    learning_rate = 0.01 * 0.001,
    batch_size = 32,
    epochs = 10,
    use_cuda = True,
    train_dataset_nums = 392458
)