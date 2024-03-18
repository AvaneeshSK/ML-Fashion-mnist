# Fashion mnist


from keras.datasets import fashion_mnist

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import tensorflow as tf

from sklearn.metrics import accuracy_score

train_images, train_labels = fashion_mnist.load_data()[0]
test_images, test_labels = fashion_mnist.load_data()[1]

text_labels = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
int_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# preprocessing : rescaling imgs
def rescale(img):
    return tf.image.convert_image_dtype(img, dtype=tf.float32)

# training imgs
train_rescaled_imgs = []
for img in train_images:
    train_rescaled_imgs.append(rescale(img))
train_rescaled_imgs = np.array(train_rescaled_imgs)

# testing imgs
test_rescaled_imgs = []
for img in test_images:
    test_rescaled_imgs.append(rescale(img))
test_rescaled_imgs = np.array(test_rescaled_imgs)

# see raw vs normalized img 
fig, (ax_1, ax_2) = plt.subplots(ncols=2)
ax_1.imshow(train_images[0])
ax_2.imshow(train_rescaled_imgs[0]) 
fig.suptitle('Raw Img (left) vs Normalized Img (right)')

# manual one hot encoding labels: (for CategoricalCrossentropy())
# train_labels_onehotencoded = [label==int_labels for label in train_labels]
# train_labels_onehotencoded_int = []
# for label in train_labels_onehotencoded:
#     temp = []
#     for each in label:
#         if each == True:
#             temp.append(1)
#         else:
#             temp.append(0)
#     train_labels_onehotencoded_int.append(temp)
# train_labels_onehotencoded_int = np.array(train_labels_onehotencoded_int)

# modelling : (5 models/methods)

models_used = ['ANN Model', 'CNN Model', 'ANN + CNN Model']

# 1. ANN Model :
ann_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    # tf.keras.layers.experimental.preprocessing.Rescaling(1/255), # can do this without manual rescaling
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=len(text_labels), activation='softmax')
])

ann_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

history = ann_model.fit(
    x=train_rescaled_imgs, 
    y=train_labels,
    epochs=100,
    verbose=2,
    batch_size=64,
    shuffle=True,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=3),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    ]
)

ann_history_df = pd.DataFrame(history.history)

preds = ann_model.predict(test_rescaled_imgs)
preds_max = []
for pred in preds:
    preds_max.append(np.argmax(pred))
ann_score = accuracy_score(y_pred=preds_max, y_true=test_labels)

# 2. CNN Model : 
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)), 
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation='relu'
    ),
    tf.keras.layers.MaxPool2D(
        padding='same',
        pool_size=2
    ),
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation='relu'
    ),
    tf.keras.layers.MaxPool2D(
        padding='same',
        pool_size=2
    ),
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation='relu'
    ),
    tf.keras.layers.MaxPool2D(
        padding='same',
        pool_size=2
    ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=len(text_labels),activation='softmax')
])

cnn_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

history = cnn_model.fit(
    x=train_rescaled_imgs, 
    y=train_labels,
    epochs=100,
    verbose=2,
    batch_size=64,
    shuffle=True,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=3),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    ]
)

cnn_history_df = pd.DataFrame(history.history)

preds = cnn_model.predict(test_rescaled_imgs)
preds_max = []
for pred in preds:
    preds_max.append(np.argmax(pred))
cnn_score = accuracy_score(y_pred=preds_max, y_true=test_labels)

# 3. ANN + CNN Model :
ann_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=128, activation='relu')
])

cnn_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)), 
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation='relu'
    ),
    tf.keras.layers.MaxPool2D(
        padding='same',
        pool_size=2
    ),
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation='relu'
    ),
    tf.keras.layers.MaxPool2D(
        padding='same',
        pool_size=2
    ),
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation='relu'
    ),
    tf.keras.layers.MaxPool2D(
        padding='same',
        pool_size=2
    ),
    tf.keras.layers.GlobalAveragePooling2D()
])

inputs = [ann_model.input, cnn_model.input]
outputs = [ann_model.output, cnn_model.output]

output_concat_layer = tf.keras.layers.Concatenate()(outputs)
output_final_layer = tf.keras.layers.Dense(units=len(text_labels), activation='softmax')(output_concat_layer)

ann_cnn_model = tf.keras.Model(
    inputs=inputs, 
    outputs=output_final_layer
)

ann_cnn_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

history = ann_cnn_model.fit(
    x=[train_rescaled_imgs, train_rescaled_imgs], 
    y=train_labels,
    verbose=2,
    epochs=100,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=3),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    ],
    batch_size=64,
    shuffle=True
)

ann_cnn_history_df = pd.DataFrame(history.history)

preds = ann_cnn_model.predict([test_rescaled_imgs, test_rescaled_imgs])
preds_max = []
for pred in preds:
    preds_max.append(np.argmax(pred))
ann_cnn_score = accuracy_score(y_pred=preds_max, y_true=test_labels)



# epochs vs accuracy
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2)
ax1.plot(ann_history_df['accuracy'], label='ANN History')
ax2.plot(cnn_history_df['accuracy'], label='CNN History')
ax3.plot(ann_cnn_history_df['accuracy'], label='ANN + CNN History')
ax1.set(xlabel='Epochs', ylabel='Accuracy', title='Epochs vs Accuracy')
ax1.set_xticks(range(0, len(ann_cnn_history_df)))
ax1.legend()
ax2.set(xlabel='Epochs', ylabel='Accuracy', title='Epochs vs Accuracy')
ax2.set_xticks(range(0, len(ann_cnn_history_df)))
ax2.legend()
ax3.set(xlabel='Epochs', ylabel='Accuracy', title='Epochs vs Accuracy')
ax3.set_xticks(range(0, len(ann_cnn_history_df)))
ax3.legend()


# compare scores
bars = ax4.bar(models_used, [ann_score, cnn_score, ann_cnn_score], color=['salmon', 'lightblue', 'pink'])
handles = []
for bar in bars:
    handles.append(Line2D([0], [0], color=bar.get_facecolor(), linewidth=5))
ax4.legend(handles=handles, labels=models_used, loc='lower right')
for p in bars.patches:
    ax4.annotate(p.get_height(), (p.get_x() + p.get_width() / 2, p.get_height()), xytext=(10, 10), ha='center', va='center', textcoords='offset points')
ax4.set_xticks(range(0, len(models_used)))
ax4.set_xticklabels(models_used, rotation=0)

# look at 4 random predictions:
# generate 4 random unique numbers between 0 and 10000 (size of test imgs) for ANN CNN Model
rand_ints = []
for i in range(0, 4):
    while True:
        random_number = np.random.randint(0, len(test_rescaled_imgs))
        if random_number not in rand_ints:
            rand_ints.append(random_number)
            break

fig, ((ax5, ax6), (ax7, ax8)) = plt.subplots(ncols=2, nrows=2)
ax5.imshow(test_rescaled_imgs[rand_ints[0]])
ax6.imshow(test_rescaled_imgs[rand_ints[1]])
ax7.imshow(test_rescaled_imgs[rand_ints[2]])
ax8.imshow(test_rescaled_imgs[rand_ints[3]])

ax5.set(title=f'Predicted : {text_labels[preds_max[rand_ints[0]]]} || Actual : {text_labels[test_labels[rand_ints[0]]]}')
ax6.set(title=f'Predicted : {text_labels[preds_max[rand_ints[1]]]} || Actual : {text_labels[test_labels[rand_ints[1]]]}')
ax7.set(title=f'Predicted : {text_labels[preds_max[rand_ints[2]]]} || Actual : {text_labels[test_labels[rand_ints[2]]]}')
ax8.set(title=f'Predicted : {text_labels[preds_max[rand_ints[3]]]} || Actual : {text_labels[test_labels[rand_ints[3]]]}')

plt.show()
