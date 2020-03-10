import random, time, os, string, re, csv, itertools, json, unicodedata, numpy
from utils.Config import Config

__version__ = "0.1.0"

##### Telegram AI Chatbot #####
print("===========================")
print(f" Telegram AI Chatbot {__version__}")
print("===========================")

# ===== CONFIG =====
config = Config()
# ==================

# We need to set this before importing Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = config.tensorflow_logging_level

import pickle
from numpy import array
from numpy import argmax
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa
import utils.dataset_helper as dataset_helper

#ENCODER
class EncoderNetwork(tf.keras.Model):
    def __init__(self,input_vocab_size):
        super().__init__()
        self.encoder_embedding = tf.keras.layers.Embedding(input_dim=input_vocab_size,
                                                           output_dim=config.embedding_dims)
        self.encoder_rnnlayer = tf.keras.layers.LSTM(config.rnn_units, return_sequences=True, return_state=True)
    
#DECODER
class DecoderNetwork(tf.keras.Model):
    def __init__(self, output_vocab_size, x_tensor_len):
        super().__init__()
        self.decoder_embedding = tf.keras.layers.Embedding(input_dim=output_vocab_size,
                                                           output_dim=config.embedding_dims) 
        self.dense_layer = tf.keras.layers.Dense(output_vocab_size)
        self.decoder_rnncell = tf.keras.layers.LSTMCell(config.rnn_units)
        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()
        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(config.dense_units, None, config.batch_size * [x_tensor_len])
        self.rnn_cell =  self.build_rnn_cell(config.batch_size)
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler= self.sampler,
                                                output_layer=self.dense_layer)

    def build_attention_mechanism(self, units,memory, memory_sequence_length):
        # return tfa.seq2seq.LuongAttention(units, memory = memory, memory_sequence_length=memory_sequence_length)
        return tfa.seq2seq.BahdanauAttention(units, memory = memory, memory_sequence_length=memory_sequence_length)

    # wrap decodernn cell  
    def build_rnn_cell(self, batch_size):
        rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnncell, self.attention_mechanism,
                                                attention_layer_size=config.dense_units)
        return rnn_cell
    
    def build_decoder_initial_state(self, batch_size, encoder_state,Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size = batch_size, 
                                                                dtype = Dtype)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state) 
        return decoder_initial_state

def max_len(tensor):
    #print( np.argmax([len(t) for t in tensor]))
    return max(len(t) for t in tensor)

def loss_function(y_pred, y):
    #shape of y [batch_size, ty]
    #shape of y_pred [batch_size, Ty, output_vocab_size] 
    sparsecategoricalcrossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                                  reduction='none')
    loss = sparsecategoricalcrossentropy(y_true=y, y_pred=y_pred)
    mask = tf.logical_not(tf.math.equal(y,0))   #output 0 for y=0 else output 1
    mask = tf.cast(mask, dtype=loss.dtype)
    loss = mask* loss
    loss = tf.reduce_mean(loss)
    return loss

def train_step(input_batch, output_batch, encoder_initial_cell_state):
    #initialize loss = 0
    loss = 0
    with tf.GradientTape() as tape:
        encoder_emb_inp = encoderNetwork.encoder_embedding(input_batch)
        a, a_tx, c_tx = encoderNetwork.encoder_rnnlayer(encoder_emb_inp, 
                                                        initial_state =encoder_initial_cell_state)

        #[last step activations,last memory_state] of encoder passed as input to decoder Network
         
        # Prepare correct Decoder input & output sequence data
        decoder_input = output_batch[:,:-1] # ignore <end>
        #compare logits with timestepped +1 version of decoder_input
        decoder_output = output_batch[:,1:] #ignore <start>


        # Decoder Embeddings
        decoder_emb_inp = decoderNetwork.decoder_embedding(decoder_input)

        #Setting up decoder memory from encoder output and Zero State for AttentionWrapperState
        decoderNetwork.attention_mechanism.setup_memory(a)
        decoder_initial_state = decoderNetwork.build_decoder_initial_state(config.batch_size,
                                                                           encoder_state=[a_tx, c_tx],
                                                                           Dtype=tf.float32)
        
        #BasicDecoderOutput        
        outputs, _, _ = decoderNetwork.decoder(decoder_emb_inp,initial_state=decoder_initial_state,
                                               sequence_length=config.batch_size * [y_tensor_len-1])

        logits = outputs.rnn_output
        #Calculate loss

        loss = loss_function(logits, decoder_output)

    #Returns the list of all layer variables / weights.
    variables = encoderNetwork.trainable_variables + decoderNetwork.trainable_variables  
    # differentiate loss wrt variables
    gradients = tape.gradient(loss, variables)

    #grads_and_vars – List of(gradient, variable) pairs.
    grads_and_vars = zip(gradients,variables)
    optimizer.apply_gradients(grads_and_vars)
    return loss

#RNN LSTM hidden and memory state initializer
def initialize_initial_state():
    return [tf.zeros((config.batch_size, config.rnn_units)), tf.zeros((config.batch_size, config.rnn_units))]

def get_predictionsa(input):
    #In this section we evaluate our model on a raw_input 
    #through the length of the model, for this we use greedsampler to run through the decoder
    #and the final embedding matrix trained on the data is used to generate embeddings
    input_raw = input

    # Preprocess
    input_lines = [('<start> '+input_raw+'').lower()]
    input_sequences = my_tokenizer.texts_to_sequences(input_lines)
    input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences,
                                                                    maxlen=x_tensor_len,
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
    maximum_iterations = tf.round(tf.reduce_max(x_tensor_len) * 2)

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

    #prediction based on our sentence earlier
    conversation = "\n============================\n"
    conversation =  conversation + "Ktoś: " + input_raw + "\n"
    conversation = conversation + "\nHubertAI:\n"

    for i in range(len(predictions)):
        line = predictions[i,:]
        seq = list(itertools.takewhile( lambda index: index !=2, line))
        
        responses = (" ".join( [my_tokenizer.index_word[w] for w in seq])).split("<end>")
        responses = [x for x in responses if x.strip()]
        responses = list(dict.fromkeys(responses))
        for response in responses:
            response = response.strip().capitalize() 
            response = response.replace('xd', 'xD')
            response = "> " + response
            response += "\n"
            conversation += response

    conversation += "\n"

    print(conversation)

    with open("conversation.txt", "a", encoding="UTF-8") as conversation_file:
        conversation_file.write(conversation)

def get_tokenizer(data):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<UKN>", filters='')
    tokenizer.fit_on_texts(data)
    data_their = tokenizer.texts_to_sequences(data)
    data_their = tf.keras.preprocessing.sequence.pad_sequences(data_their, padding='post')
    return tokenizer, data_their

if __name__ == "__main__":
    # Get data from dataset
    all_data = dataset_helper.get_dataset(True)
    # Create placeholder for list with their messages
    raw_data_their = list()
    # Create placeholder for list with my messages
    raw_data_my = list()

    # split data for mien and thier
    for d in all_data:
        raw_data_their.append(d[0]), raw_data_my.append(d[1])

    # Create tokenizers
    their_tokenizer, their_data = get_tokenizer(raw_data_their)
    my_tokenizer, my_data = get_tokenizer(raw_data_my)

    # Split data for training
    X_train, X_test, Y_train, Y_test = train_test_split(their_data, my_data, test_size=0.2)

    buffer_size = len(X_train)
    steps_per_epoch = buffer_size//config.batch_size
    Dtype = tf.float32   #used to initialize DecoderCell Zero state

    x_tensor_len = max_len(their_data)
    y_tensor_len = max_len(my_data)

    input_vocab_size = len(their_tokenizer.word_index) + 1
    output_vocab_size = len(my_tokenizer.word_index) + 1

    # make dataset for training
    dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(buffer_size).batch(config.batch_size, drop_remainder=True)
    example_X, example_Y = next(iter(dataset))

    encoderNetwork = EncoderNetwork(input_vocab_size)
    decoderNetwork = DecoderNetwork(output_vocab_size, x_tensor_len)
    optimizer = tf.keras.optimizers.Adam()

    checkpoint_prefix = os.path.join(config.checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                    encoder=encoderNetwork,
                                    decoder=decoderNetwork)
    checkpoint.restore(tf.train.latest_checkpoint(config.checkpoint_dir))
    
    print(f"Learning started. Total epoch: {config.epochs}. Steps per epoch: {steps_per_epoch}. Data: {config.max_data_size}. Batch: {config.batch_size}")

    # Learning here
    for i in range(1, config.epochs+1):
        start = time.perf_counter()
        encoder_initial_cell_state = initialize_initial_state()
        total_loss = 0.0

        for (batch, (input_batch, output_batch)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(input_batch, output_batch, encoder_initial_cell_state)
            total_loss += batch_loss

            if (batch+1) % 5 == 0:
                print(f"Total loss: {batch_loss.numpy()}. Epoch {i} Batch {batch + 1} ")

        # saving (checkpoint) the model every 2 epochs
        if config.save_checkpoint:
            if (i + 1) % config.save_checkpoint_for_epoch == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
                print("Checkpoint saved.")

        end = time.perf_counter()
        print("Epoch ended. Time: "  + str(round((end - start), 5)) + " s.")
        
        # Test model with random exmaple
        # if config.test_every_epoch:
        #     random_exmaple = random.choice(config.examples)
        #     Chat.get_predictions(random_exmaple)8