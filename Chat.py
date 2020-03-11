import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

from Trainer import Trainer

######################
# Example chat app
######################
trainer = Trainer()
trainer.load_checkpoint()

while True:
    mess = input("Write your message: ")
    predictions = trainer.get_predictions(mess)
    trainer.print_predictions(mess, predictions, False)
    