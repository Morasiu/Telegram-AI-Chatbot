import json, os
class Config:
    def __init__(self, path_to_config_file="./config.json"):
        if not os.path.exists(path_to_config_file):
            raise ValueError("config.json do not exist.")
        with open(path_to_config_file) as json_file:
            config = json.load(json_file)

        self.telegram_export_path = config["telegram_export_path"]
        self.max_data_size = config["max_data_size"]
        self.batch_size = config["batch_size"]
        self.embedding_dims = config["embedding_dims"]
        self.rnn_units = config["rnn_units"]
        self.dense_units = config["dense_units"]
        self.save_checkpoint = config["save_checkpoint"]
        self.checkpoint_dir = config["checkpoint_dir"]
        self.test_every_epoch = config["test_every_epoch"]
        self.examples = config["examples"]
        self.save_checkpoint_for_epoch = config["save_checkpoint_for_epoch"]
        self.epochs = config["epochs"]
        self.enable_special_char = config["enable_special_char"]
        self.max_message_length = config["max_message_length"]

        print("config.json loaded.")