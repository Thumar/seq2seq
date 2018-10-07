import tensorflow as tf
import numpy as np

params = {
    'batch_size': 128,
    'iter_step': 25,
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
    for i in range(0, len(ints) - len_win, params['iter_step']):
        clip = ints[i: i + len_win]
        yield clip.reshape([params['batch_size'], params['seq_len']])


def input_fn(ints):
    dataset = tf.data.Dataset.from_generator(
        lambda: next_batch(ints), tf.int32, tf.TensorShape([None, params['seq_len']]))
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def start_sent(x):
    _x = tf.fill([tf.shape(x)[0], 1], params['char2idx']['<start>'])
    return tf.concat([_x, x], 1)


def end_sent(x):
    _x = tf.fill([tf.shape(x)[0], 1], params['char2idx']['<end>'])
    return tf.concat([x, _x], 1)


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


def forward(inputs, is_training):
    if is_training:
        batch_sz = tf.shape(inputs)[0]

        with tf.variable_scope('main', reuse=False):
            embedding = tf.get_variable('lookup_table', [params['vocab_size'], params['hidden_dim']])
            cells = multi_cell_fn()

            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=tf.nn.embedding_lookup(embedding, inputs),
                sequence_length=tf.count_nonzero(inputs, 1, dtype=tf.int32))

            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=cells,
                helper=helper,
                initial_state=cells.zero_state(batch_sz, tf.float32),
                output_layer=tf.layers.Dense(params['vocab_size']))

            decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder)

            logits = decoder_output.rnn_output
            return logits

    if not is_training:
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

            predict = decoder_out.predicted_ids[:, :, 0]
            return predict


text2ints, params['char2idx'] = parse_text('/home/manu/PycharmProjects/seq2seq/data/text')
params['vocab_size'] = len(params['char2idx'])
params['idx2char'] = {i: c for c, i in params['char2idx'].items()}



ops = {}
X = input_fn(text2ints)

logits = forward(start_sent(X), is_training=True)

ops['global_step'] = tf.Variable(0, trainable=False)

targets = end_sent(X)

ops['loss'] = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(
    logits=logits,
    targets=targets,
    weights=tf.to_float(tf.ones_like(targets))))

ops['train'] = tf.train.AdamOptimizer().apply_gradients(
    clip_grads(ops['loss']), global_step=ops['global_step'])

ops['generate'] = forward(None, is_training=False)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
i = 0
while i <= 1000:
    try:
        _, step, loss = sess.run([ops['train'], ops['global_step'], ops['loss']])
    except tf.errors.OutOfRangeError:
        break
    else:
        if step % params['display_step'] == 0 or step == 1:
            print("Step %d | Loss %.3f" % (step, loss))
        if step % params['generate_step'] == 0 and step > 1:
            text2ints = sess.run(ops['generate'])[0]
            print('\n' + ''.join([params['idx2char'][i] for i in text2ints]) + '\n')
    i = i + 1
