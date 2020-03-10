import itertools, os


from utils.Config import Config
config = Config()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = config.tensorflow_logging_level

import tensorflow as tf
import tensorflow_addons as tfa
import numpy
import Trainer
import utils.dataset_helper as dataset_helper

##################################
#
# Simple example usage od an app.
#
##################################

dataset = dataset_helper.get_dataset(False)
my_tokenizer, _ = Trainer.get_tokenizer([x[1] for x in dataset])
their_tokenizer, their_data = Trainer.get_tokenizer([x[0] for x in dataset])

input_vocab_size = len(their_tokenizer.word_index) + 1
output_vocab_size = len(my_tokenizer.word_index) + 1

their_tensor_len = Trainer.max_len(their_data)

encoderNetwork = Trainer.EncoderNetwork(input_vocab_size)
decoderNetwork = Trainer.DecoderNetwork(output_vocab_size, their_tensor_len)

def get_predictions(input):
    #In this section we evaluate our model on a raw_input 
    #through the length of the model, for this we use greedsampler to run through the decoder
    #and the final embedding matrix trained on the data is used to generate embeddings
    input_raw= input 
    # Preprocess
    input_lines = [('<start> '+input_raw+'').lower()]
    input_sequences = my_tokenizer.texts_to_sequences(input_lines)
    input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences,
                                                                    maxlen=their_tensor_len,
                                                                    padding='post')
    inp = tf.convert_to_tensor(input_sequences)
    inference_batch_size = input_sequences.shape[0]
    encoder_initial_cell_state = [tf.zeros((inference_batch_size, config.rnn_units)),
                                tf.zeros((inference_batch_size, config.rnn_units))]

    encoder_emb_inp = encoderNetwork.encoder_embedding(inp)
    a, a_tx, c_tx = encoderNetwork.encoder_rnnlayer(encoder_emb_inp,
                                                    initial_state=encoder_initial_cell_state)

    start_tokens = tf.fill([inference_batch_size],my_tokenizer.word_index['<start>'])

    end_token = my_tokenizer.word_index['<end>']

    greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

    decoder_input = tf.expand_dims([my_tokenizer.word_index['<start>']]* inference_batch_size,1)
    decoder_emb_inp = decoderNetwork.decoder_embedding(decoder_input)

    decoder_instance = tfa.seq2seq.BasicDecoder(cell = decoderNetwork.rnn_cell, sampler = greedy_sampler,
                                                output_layer=decoderNetwork.dense_layer)
    decoderNetwork.attention_mechanism.setup_memory(a)
    #pass [ last step activations , encoder memory_state ] as input to decoder for LSTM
    decoder_initial_state = decoderNetwork.build_decoder_initial_state(inference_batch_size,
                                                                    encoder_state=[a_tx, c_tx],
                                                                    Dtype=tf.float32)

    # Since we do not know the target sequence lengths in advance, we use maximum_iterations to limit the translation lengths.
    # One heuristic is to decode up to two times the source sentence lengths.
    maximum_iterations = tf.round(tf.reduce_max(their_tensor_len) * 2)

    #initialize inference decoder
    decoder_embedding_matrix = decoderNetwork.decoder_embedding.variables[0] 
    (first_finished, first_inputs,first_state) = decoder_instance.initialize(decoder_embedding_matrix,
                                start_tokens = start_tokens,
                                end_token=end_token,
                                initial_state = decoder_initial_state)

    inputs = first_inputs
    state = first_state  
    predictions = numpy.empty((inference_batch_size,0), dtype = numpy.int32)   

    for i in range(maximum_iterations):
        outputs, next_state, next_inputs, finished = decoder_instance.step(i, inputs,state)
        inputs = next_inputs
        state = next_state
        outputs = numpy.expand_dims(outputs.sample_id,axis = -1)
        predictions = numpy.append(predictions, outputs, axis = -1)

    # #prediction based on our sentence earlier
    # conversation = "\n============================\n"
    # conversation =  conversation + "Ktoś: " + input_raw + "\n"
    # conversation = conversation + "\nHubertAI:\n"

    responses = []
    for i in range(len(predictions)):
        line = predictions[i,:]
        seq = list(itertools.takewhile( lambda index: index !=2, line))
        
        responses = (" ".join( [my_tokenizer.index_word[w] for w in seq])).split("<end>")
        responses = [x for x in responses if x.strip()]
        responses = list(dict.fromkeys(responses))
        # for response in responses:
        #     response = response.strip().capitalize() 
        #     response = "> " + response
        #     response += "\n"
        #     conversation += response
    return responses
    # conversation += "\n"

    # print(conversation)

    # with open("conversation.txt", "a", encoding="UTF-8") as conversation_file:
    #     conversation_file.write(conversation)

def conversation(mess, predictions):
    conversation = "\n============================\n"
    conversation =  conversation + "You: " + mess + "\n"
    conversation = conversation + "\nAI:\n"

    for response in predictions:
            response = response.strip().capitalize() 
            response = response.replace('xd', 'xD')
            response = "> " + response
            response += "\n"
            conversation += response


    conversation += "\n"

    print(conversation)

    with open("conversation.txt", "a", encoding="UTF-8") as conversation_file:
        conversation_file.write(conversation)


Trainer.load_checkpoint(Trainer.get_optimizer(), encoderNetwork, decoderNetwork)
while True:
    mess = input("Write your message: ")
    predictions = get_predictions(mess)
    conversation(mess, predictions)
    