Config = dict(
    encoder_layers = 4,
    encoder_hidden_dim = 768,
    vae_hidden_dim = 128,
    vae_latten_dim = 64,
    vae_layers = 4,
    vae_decoder_hidden_dim = 128,
    vae_decoder_layers = 4,
    learning_rate = 0.001 * 0.001,
    batch_size = 16,
    epochs = 6,
    use_cuda = True,
    train_dataset_nums = 392458,
    print_every = 40,
    check_interval = 5000,
    dropout = 0.0,
    print_every_on_test = 10,

)