import random, time, os, string, re, csv, itertools, json, unicodedata, numpy
from utils.Config import Config

# We need to set this before importing Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

import pickle
from numpy import array
from numpy import argmax
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa
import utils.dataset_helper as dataset_helper

class Trainer:
    def __init__(self):
        self.config = Config()
        self.all_data = dataset_helper.get_dataset(True)
        self.my_tokenizer, self.my_data = self.get_tokenizer([x[1] for x in self.all_data])
        self.their_tokenizer, self.their_data = self.get_tokenizer([x[0] for x in self.all_data])

        self.input_vocab_size = len(self.their_tokenizer.word_index) + 1
        self.output_vocab_size = len(self.my_tokenizer.word_index) + 1

        self.their_tensor_len = self.max_len(self.their_data)
        self.my_tensor_len = self.max_len(self.my_data)
        
        # x_tensor_len = max_len(their_data)
        # y_tensor_len = max_len(my_data)
        self.optimizer = tf.keras.optimizers.Adam()
        self.encoderNetwork = self.EncoderNetwork(self.input_vocab_size, self.config)
        self.decoderNetwork = self.DecoderNetwork(self.output_vocab_size, self.their_tensor_len, self.config)

        self.checkpoint  = tf.train.Checkpoint(optimizer=self.optimizer,
                                        encoder=self.encoderNetwork,
                                        decoder=self.decoderNetwork)


    #ENCODER
    class EncoderNetwork(tf.keras.Model):
        def __init__(self, input_vocab_size, config):
            super().__init__()
            self.config = config
            self.encoder_embedding = tf.keras.layers.Embedding(input_dim=input_vocab_size,
                                                            output_dim=self.config.embedding_dims)
            self.encoder_rnnlayer = tf.keras.layers.LSTM(self.config.rnn_units, return_sequences=True, return_state=True)
        
    #DECODER
    class DecoderNetwork(tf.keras.Model):
        def __init__(self, output_vocab_size, x_tensor_len, config):
            super().__init__()
            self.config = config
            self.decoder_embedding = tf.keras.layers.Embedding(input_dim=output_vocab_size,
                                                            output_dim=self.config.embedding_dims) 
            self.dense_layer = tf.keras.layers.Dense(output_vocab_size)
            self.decoder_rnncell = tf.keras.layers.LSTMCell(self.config.rnn_units)
            # Sampler
            self.sampler = tfa.seq2seq.sampler.TrainingSampler()
            # Create attention mechanism with memory = None
            self.attention_mechanism = self.build_attention_mechanism(self.config.dense_units, None, self.config.batch_size * [x_tensor_len])
            self.rnn_cell =  self.build_rnn_cell(self.config.batch_size)
            self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler= self.sampler,
                                                    output_layer=self.dense_layer)

        def build_attention_mechanism(self, units,memory, memory_sequence_length):
            # return tfa.seq2seq.LuongAttention(units, memory = memory, memory_sequence_length=memory_sequence_length)
            return tfa.seq2seq.BahdanauAttention(units, memory = memory, memory_sequence_length=memory_sequence_length)

        # wrap decodernn cell  
        def build_rnn_cell(self, batch_size):
            rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnncell, self.attention_mechanism,
                                                    attention_layer_size=self.config.dense_units)
            return rnn_cell
        
        def build_decoder_initial_state(self, batch_size, encoder_state,Dtype):
            decoder_initial_state = self.rnn_cell.get_initial_state(batch_size = batch_size, 
                                                                    dtype = Dtype)
            decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state) 
            return decoder_initial_state

    def max_len(self, tensor):
        #print( np.argmax([len(t) for t in tensor]))
        return max(len(t) for t in tensor)

    def loss_function(self, y_pred, y):
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

    def train_step(self, input_batch, output_batch, encoder_initial_cell_state):
        #initialize loss = 0
        loss = 0
        with tf.GradientTape() as tape:
            encoder_emb_inp = self.encoderNetwork.encoder_embedding(input_batch)
            a, a_tx, c_tx = self.encoderNetwork.encoder_rnnlayer(encoder_emb_inp, 
                                                            initial_state =encoder_initial_cell_state)

            #[last step activations,last memory_state] of encoder passed as input to decoder Network
            
            # Prepare correct Decoder input & output sequence data
            decoder_input = output_batch[:,:-1] # ignore <end>
            #compare logits with timestepped +1 version of decoder_input
            decoder_output = output_batch[:,1:] #ignore <start>


            # Decoder Embeddings
            decoder_emb_inp = self.decoderNetwork.decoder_embedding(decoder_input)

            #Setting up decoder memory from encoder output and Zero State for AttentionWrapperState
            self.decoderNetwork.attention_mechanism.setup_memory(a)
            decoder_initial_state = self.decoderNetwork.build_decoder_initial_state(self.config.batch_size,
                                                                            encoder_state=[a_tx, c_tx],
                                                                            Dtype=tf.float32)
            
            #BasicDecoderOutput        
            outputs, _, _ = self.decoderNetwork.decoder(decoder_emb_inp, 
                                                initial_state=decoder_initial_state,
                                                sequence_length=self.config.batch_size * [self.my_tensor_len - 1])

            logits = outputs.rnn_output
            #Calculate loss

            loss = self.loss_function(logits, decoder_output)

        #Returns the list of all layer variables / weights.
        variables = self.encoderNetwork.trainable_variables + self.decoderNetwork.trainable_variables  
        # differentiate loss wrt variables
        gradients = tape.gradient(loss, variables)

        #grads_and_vars – List of(gradient, variable) pairs.
        grads_and_vars = zip(gradients,variables)
        self.optimizer.apply_gradients(grads_and_vars)
        return loss

    #RNN LSTM hidden and memory state initializer
    def initialize_initial_state(self):
        return [tf.zeros((self.config.batch_size, self.config.rnn_units)), tf.zeros((self.config.batch_size, self.config.rnn_units))]

    def get_tokenizer(self, data):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<UKN>", filters='')
        tokenizer.fit_on_texts(data)
        data_their = tokenizer.texts_to_sequences(data)
        data_their = tf.keras.preprocessing.sequence.pad_sequences(data_their, padding='post')
        return tokenizer, data_their

    def load_checkpoint(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.config.checkpoint_dir)).expect_partial()
        print("Checkpoint loaded.")

    def get_predictions(self, input):
        #In this section we evaluate our model on a raw_input 
        #through the length of the model, for this we use greedsampler to run through the decoder
        #and the final embedding matrix trained on the data is used to generate embeddings
        input_raw= input 
        # Preprocess
        input_lines = [('<start> '+input_raw+'').lower()]
        input_sequences = self.my_tokenizer.texts_to_sequences(input_lines)
        input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences,
                                                                        maxlen=self.their_tensor_len,
                                                                        padding='post')
        inp = tf.convert_to_tensor(input_sequences)
        inference_batch_size = input_sequences.shape[0]
        encoder_initial_cell_state = [tf.zeros((inference_batch_size, self.config.rnn_units)),
                                    tf.zeros((inference_batch_size, self.config.rnn_units))]

        encoder_emb_inp = self.encoderNetwork.encoder_embedding(inp)
        a, a_tx, c_tx = self.encoderNetwork.encoder_rnnlayer(encoder_emb_inp,
                                                        initial_state=encoder_initial_cell_state)

        start_tokens = tf.fill([inference_batch_size], self.my_tokenizer.word_index['<start>'])

        end_token = self.my_tokenizer.word_index['<end>']

        greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

        decoder_input = tf.expand_dims([self.my_tokenizer.word_index['<start>']]* inference_batch_size,1)
        decoder_emb_inp = self.decoderNetwork.decoder_embedding(decoder_input)

        decoder_instance = tfa.seq2seq.BasicDecoder(cell=self.decoderNetwork.rnn_cell, 
                                                    sampler = greedy_sampler,
                                                    output_layer=self.decoderNetwork.dense_layer)

        self.decoderNetwork.attention_mechanism.setup_memory(a)
        #pass [ last step activations , encoder memory_state ] as input to decoder for LSTM
        decoder_initial_state = self.decoderNetwork.build_decoder_initial_state(inference_batch_size,
                                                                        encoder_state=[a_tx, c_tx],
                                                                        Dtype=tf.float32)

        # Since we do not know the target sequence lengths in advance, we use maximum_iterations to limit the translation lengths.
        # One heuristic is to decode up to two times the source sentence lengths.
        maximum_iterations = tf.round(tf.reduce_max(self.their_tensor_len) * 2)

        #initialize inference decoder
        decoder_embedding_matrix = self.decoderNetwork.decoder_embedding.variables[0] 
        (first_finished, first_inputs,first_state) = decoder_instance.initialize(decoder_embedding_matrix,
                                    start_tokens=start_tokens,
                                    end_token=end_token,
                                    initial_state=decoder_initial_state)

        inputs = first_inputs
        state = first_state  
        predictions = numpy.empty((inference_batch_size,0), dtype = numpy.int32)   

        for i in range(maximum_iterations):
            outputs, next_state, next_inputs, finished = decoder_instance.step(i, inputs,state)
            inputs = next_inputs
            state = next_state
            outputs = numpy.expand_dims(outputs.sample_id,axis = -1)
            predictions = numpy.append(predictions, outputs, axis = -1)

        responses = []
        for i in range(len(predictions)):
            line = predictions[i,:]
            seq = list(itertools.takewhile( lambda index: index !=2, line))
            
            responses = (" ".join( [self.my_tokenizer.index_word[w] for w in seq])).split("<end>")
            responses = [x for x in responses if x.strip()]
            responses = list(dict.fromkeys(responses))
        return responses

    def print_predictions(self, input, predictions, save_to_file = True):
        conversation = "\n============================\n"
        conversation =  conversation + "You: " + input + "\n"
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

    def train(self):
        # # Split data for training
        X_train, X_test, Y_train, Y_test = train_test_split(self.their_data, self.my_data, test_size=0.2)

        buffer_size = len(X_train)
        steps_per_epoch = buffer_size//self.config.batch_size
        Dtype = tf.float32   #used to initialize DecoderCell Zero state

        # make dataset for training
        dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(buffer_size).batch(self.config.batch_size, drop_remainder=True)
        example_X, example_Y = next(iter(dataset))

        checkpoint_prefix = os.path.join(self.config.checkpoint_dir, "ckpt")

        # checkpoint.restore(tf.train.latest_checkpoint(config.checkpoint_dir))

        print(f"Learning started. Total epoch: {self.config.epochs}. Steps per epoch: {steps_per_epoch}. Data: {self.config.max_data_size}. Batch: {self.config.batch_size}")

        # Learning here
        for i in range(1, self.config.epochs+1):
            start = time.perf_counter()
            encoder_initial_cell_state = self.initialize_initial_state()
            total_loss = 0.0

            for (batch, (input_batch, output_batch)) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss = self.train_step(input_batch, output_batch, encoder_initial_cell_state)
                total_loss += batch_loss

                if (batch+1) % 5 == 0:
                    print(f"Total loss: {batch_loss.numpy():f}. Epoch {i} Batch {batch + 1} ")

            # saving (checkpoint) the model every 2 epochs
            if self.config.save_checkpoint:
                if (i + 1) % self.config.save_checkpoint_for_epoch == 0:
                    self.checkpoint.save(file_prefix = checkpoint_prefix)
                    print("Checkpoint saved.")

            end = time.perf_counter()
            print("Epoch ended. Time: "  + str(round((end - start), 5)) + " s.")
            
            # Test model with random exmaple
            if self.config.test_every_epoch:
                random_exmaple = random.choice(self.config.examples)
                predictions = self.get_predictions(random_exmaple)
                self.print_predictions(random_exmaple, predictions)

##########################################################
#                         TRAIN

##########################################################

if __name__ == "__main__":
    __version__ = "0.1.0"

    ##### Telegram AI Chatbot #####
    print("===========================")
    print(f" Telegram AI Chatbot {__version__}")
    print("===========================")

    trainer = Trainer()
    trainer.load_checkpoint()
    predictions = trainer.train()