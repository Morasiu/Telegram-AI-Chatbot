# Telegram-AI-Chatbot

It's a chatbot written using Tensorflow and Python.

I used scripts from [this](https://www.tensorflow.org/addons/tutorials/networks_seq2seq_nmt) tutorial. Thanks tensorflow!

## How to train
1. Clone or download this project.
1. Export your Telegram Data to `json` format.
1. Check `config.json` and choose your preferences.
1. Use `pipenv` to install dependencies
    ```bash
    pipenv sync
    ```
    > It's about 1.5 GB so... be prepared.

    >You can install it locally. Just create dir `.venv` here
1. Now you can run script using 
    ```
    > pipenv shell
    > python Trainer.py
    ```
    > I recommend running all 15 epochs.
1. After you trained use your model to predict some messages.

> Results can be... disappointing.


## Config

You can find recommended `congig.json` in this repo.

* "telegram_export_path" - `string`, path to your `result.json file`
* "max_data_size" - `int`, max size of your dataset. More == better model. But it all depends on your RAM or VRAM. If you exceed your RAM, change batch_size to smaller value.
* "batch_size" - `int`, more batch == faster training, but watch out for RAM.
* "epochs" - `int` - how many epoch should be ran in one `python Train.py`, more == better, but you can overtrain model. Watch out for loss value (less == better.)
* "embedding_dims": `int`, embbeded layers, recommended value is `256`, but you can experiment with it.
* "rnn_units": `int`, recurrent layers, recommended value is `1024`, but you can experiment with it.
* "dense_units": `int`, fully-connected layers, recommended value is `1024`, but you can experiment with it.
* "enable_special_char": `bool`, if set to `true`, emojis will be included in dataset. Set to `false` to remove emojis.
* "max_message_length": `int`, max length of a simple messsage. Longer messages will be ommited.
* "checkpoint_dir": `string`, path to save your training checkpoint. It can take a lot of space. (Over 2GB probably)
* "save_checkpoint": `bool`, if `true`, `train.py` will save checkpoint from time to time. `true` is recommended.
* "save_checkpoint_for_epoch": `int`, how often `train.py` will save checkpoints, starting from `1`. For example if value will be `2`, it will save checkpoint at `1`, `3`, `5` ...
* "test_every_epoch": `bool`, if set to `true` it will test model at the end of every epoch with random message from `examples`
* "examples": `arrays[string]`, array of messages, which would be used to test model at the end of every epoch.

## FAQ

1. What is your setup?

    I'm using Nvidia GTX 1070 tu train my model.

1. How much time will an epoch take?

    For 40 000 messages nad batch_size 64 one epochs took me 420s.

1.  How many messages should I include?

    Basically the more, the better.

1. Okay. I've trained it. Now what?

    Now you can use it some app or for bot. See `Chat.py` for example.


## HELP?

If you want some help in that fill the issue, but keep in mind that I'm just started to learning ML and stuff.

Do you have suggestion to improve this? Great. I'll be very happy to colaborate with you. Let's start with making a new `issue`.
