import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

FILE_PATH = '/home/manu/PycharmProjects/seq2seq/data/anna.txt'
params = {
    'batch_size': 128,
    'text_iter_step': 25,
    'seq_len': 200,
    'hidden_dim': 128,
    'n_layers': 2,
    'beam_width': 5,
    'display_step': 10,
    'generate_step': 100,
    'clip_norm': 5.0,
}


def parse_text(file_path):
    with open(file_path) as f:
        text = f.read()

    char2idx = {c: i + 3 for i, c in enumerate(set(text))}
    char2idx['<pad>'] = 0
    char2idx['<start>'] = 1
    char2idx['<end>'] = 2

    ints = np.array([char2idx[char] for char in list(text)])
    return ints, char2idx


def next_batch(ints):
    len_win = params['seq_len'] * params['batch_size']
    for i in range(0, len(ints) - len_win, params['text_iter_step']):
        clip = ints[i: i + len_win]
        yield clip.reshape([params['batch_size'], params['seq_len']])


def create_dict(s1, s2):
    return {'input': s1, 'output': s2}


def start_(x):
    _x = tf.fill([tf.shape(x)[0], 1], params['char2idx']['<start>'])
    return tf.concat([_x, x], 1)


def end_(x):
    _x = tf.fill([tf.shape(x)[0], 1], params['char2idx']['<end>'])
    return tf.concat([x, _x], 1)


def input_fn(ints):
    dataset1 = tf.data.Dataset.from_generator(
        lambda: next_batch(ints), tf.int32, tf.TensorShape([None, params['seq_len']]))
    dataset1 = dataset1.map(start_)

    dataset2 = tf.data.Dataset.from_generator(
        lambda: next_batch(ints), tf.int32, tf.TensorShape([None, params['seq_len']]))
    dataset2 = dataset2.map(end_)

    dataset = tf.data.Dataset.zip((dataset1, dataset2))
    dataset = dataset.map(create_dict)
    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()


def cell_fn():
    return tf.nn.rnn_cell.ResidualWrapper(
        tf.nn.rnn_cell.GRUCell(params['hidden_dim'],
                               kernel_initializer=tf.orthogonal_initializer()))


def multi_cell_fn():
    return tf.nn.rnn_cell.MultiRNNCell([cell_fn() for _ in range(params['n_layers'])])


def clip_grads(loss):
    variables = tf.trainable_variables()
    grads = tf.gradients(loss, variables)
    clipped_grads, _ = tf.clip_by_global_norm(grads, params['clip_norm'])
    return zip(clipped_grads, variables)


def seq2seq_model(features, labels, mode, params):
    ops = {}
    if mode == tf.estimator.ModeKeys.TRAIN:
        batch_sz = tf.shape(features['input'])[0]

        with tf.variable_scope('main', reuse=False):
            embedding = tf.get_variable('lookup_table', [params['vocab_size'], params['hidden_dim']])

            cells = multi_cell_fn()

            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=tf.nn.embedding_lookup(embedding, features['input']),
                sequence_length=tf.count_nonzero(features['input'], 1, dtype=tf.int32))

            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=cells,
                helper=helper,
                initial_state=cells.zero_state(batch_sz, tf.float32),
                output_layer=tf.layers.Dense(params['vocab_size']))

            decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder)

            logits = decoder_output.rnn_output

            output = features['output']

            ops['global_step'] = tf.Variable(0, trainable=False)

            ops['loss'] = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(
                logits=logits,
                targets=output,
                weights=tf.to_float(tf.ones_like(output))))

            ops['train'] = tf.train.AdamOptimizer().apply_gradients(
                clip_grads(ops['loss']), global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=ops['loss'],
                train_op=ops['train']
            )

    if mode == tf.estimator.ModeKeys.PREDICT:
        with tf.variable_scope('main', reuse=True):
            cells = multi_cell_fn()

            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=cells,
                embedding=tf.get_variable('lookup_table'),
                start_tokens=tf.tile(tf.constant(
                    [params['char2idx']['<start>']], dtype=tf.int32), [1]),
                end_token=params['char2idx']['<end>'],
                initial_state=tf.contrib.seq2seq.tile_batch(
                    cells.zero_state(1, tf.float32), params['beam_width']),
                beam_width=params['beam_width'],
                output_layer=tf.layers.Dense(params['vocab_size'], _reuse=True))

            decoder_out, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder,
                maximum_iterations=params['seq_len'])

            tf.identity(decoder_out[0].predicted_ids, name='predictions')
            predict = decoder_out.predicted_ids[:, :, 0]
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predict)


ints, params['char2idx'] = parse_text(FILE_PATH)
params['vocab_size'] = len(params['char2idx'])
params['idx2char'] = {i: c for c, i in params['char2idx'].items()}

est = tf.estimator.Estimator(
    model_fn=seq2seq_model,
    model_dir='model_dir', params=params)

print('Vocabulary size:', params['vocab_size'])

est.train(input_fn=lambda: input_fn(ints), steps=1000)

pre = list(est.predict(input_fn=lambda: input_fn(ints)))

print(pre)
