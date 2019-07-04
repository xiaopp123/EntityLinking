from utils import get_kb, get_train_data, get_char_dict
from utils import get_random, data_generator
from keras.models import Model
import keras.backend as K
from keras.layers import Input, Embedding, Lambda, Bidirectional, CuDNNLSTM,Dropout
from keras.layers import Conv1D, Dense, Concatenate, Multiply


dim = 128

def seq_maxpool(x):
    seq, mask = x
    seq -= (1 - mask) * 1e10

    return K.max(seq, 1)

id2kb, kb2id = get_kb()

train_data = get_train_data()

id2char, char2id = get_char_dict(id2kb, train_data)

random_order = get_random(len(train_data))

dev_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 == 0]
train_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 != 0]

sentence_in = Input(shape=(None, ))
mention_in = Input(shape=(None, ))
left_in = Input(shape=(None, ))
right_in = Input(shape=(None, ))
y_in = Input(shape=(None, ))
t_in = Input(shape=(None, ))

sentence_mask = Lambda(lambda x : K.cast(K.greater(K.expand_dims(x, 2), 0), "float32"))(sentence_in)
mention_mask = Lambda(lambda x : K.cast(K.greater(K.expand_dims(x, 2), 0), "float32"))(mention_in)

embedding = Embedding(len(id2char) + 2, dim)
#print(embedding.shape)

sentence_embed = embedding(sentence_in)
sentence_embed = Dropout(0.2)(sentence_embed)
sentence_embed = Lambda(lambda x : x[0] * x[1])([sentence_embed, sentence_mask])
sentence_embed = Bidirectional(CuDNNLSTM(dim // 2, return_sequences=True))(sentence_embed)
sentence_embed = Lambda(lambda x : x[0] * x[1])([sentence_embed, sentence_mask])
sentence_embed = Bidirectional(CuDNNLSTM(dim // 2, return_sequences=True))(sentence_embed)
sentence_embed = Lambda(lambda x : x[0] * x[1])([sentence_embed, sentence_mask])

print(sentence_embed)
h = Conv1D(dim, 3, activation='relu', padding='same')(sentence_embed)
print(h)
left_p = Dense(1, activation='sigmoid')(h)
right_p = Dense(1, activation='sigmoid')(h)
print('left_p', left_p)
print('right_p', right_p)

#实体预测
entity_mode = Model(sentence_in, [left_p, right_p])

y = Lambda(lambda x : K.expand_dims(x, 2))(y_in)
print("y:", y)
sentence_label_emb = Concatenate()([sentence_embed, y])
sentence_label_emb = Dropout(0.2)(sentence_label_emb)
sentence_label_emb = Conv1D(dim, 3, padding='same')(sentence_label_emb)


mention_emb = embedding(mention_in)
mention_emb = Dropout(0.2)(mention_emb)

mention_emb = Lambda(lambda x : x[0] * x[1])([mention_emb, mention_mask])
mention_emb = Bidirectional(CuDNNLSTM(dim // 2, return_sequences=True))(mention_emb)
mention_emb = Lambda(lambda x: x[0] * x[1])([mention_emb, mention_mask])
mention_emb = Bidirectional(CuDNNLSTM(dim // 2, return_sequences=True))(mention_emb)
mention_emb = Lambda(lambda x: x[0] * x[1])([mention_emb, mention_mask])


print("sentence_label_emb: ", sentence_label_emb)
print("mention_emb: ", mention_emb)

sentence_label_emb = Lambda(seq_maxpool)([sentence_label_emb, sentence_mask])
mention_emb = Lambda(seq_maxpool)([mention_emb, mention_mask])

print("sentence_label_emb: ", sentence_label_emb)
print("mention_emb: ", mention_emb)

sent_mention = Multiply()([sentence_label_emb, mention_emb])
sent_mention = Concatenate()([sentence_label_emb, mention_emb, sent_mention])
sent_mention = Dense(dim, activation='relu')(sent_mention)

pt = Dense(1, activation='sigmoid')(sent_mention)

train_model = Model([sentence_in, mention_in, left_in, right_in, y_in, t_in], [left_p, right_p, pt])

left_in = K.expand_dims(left_in, 2)
right_in = K.expand_dims(right_in, 2)

left_loss = K.binary_crossentropy(left_in, left_p)
left_loss = K.sum(left_loss * sentence_mask) / K.sum(sentence_mask)

right_loss = K.binary_crossentropy(right_in, right_p)
right_loss = K.sum(right_loss * sentence_mask) / K.sum(sentence_mask)

pt_loss = K.mean(K.binary_crossentropy(t_in, pt))

loss = left_loss + right_loss + pt_loss

train_model.add_loss(loss)
train_model.compile(optimizer=Adam(1e-3))
train_model.summary()

train_D = data_generator(train_data)

