import matplotlib.pylab as plt
import numpy as np
import zlib
import glob
import random
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
import os
from generator_class import DataGenerator

#GPU/CPU Selection
gpu_setting = 'y'

print(tf.config.get_visible_devices('GPU'))
print(tf.config.list_physical_devices())

#Collecting Input
print()
print('--------------------------------------------------------------------------------------------')
path = glob.glob('event_files/*.info')
n_channels = 3
num_nu = int(input('Number of neutrinos: '))
if num_nu > 139301:
    num_nu = 139301
    print('Only 139301 events')
include_tau = input('Include taus? (y/n) ')
num_epochs = int(input('Number of epochs: '))
batch_size = int(input('Batch size: '))
test_name = input('Description ')
if test_name == 'd':
    test_name = 'complex_%s_view_'%(n_channels)+'%s_epoch_'%(num_epochs)+'%s_nu'%(num_nu)
    print('Using name: ' + test_name)
print('--------------------------------------------------------------------------------------------')
print()

#Helper Functions
cells=500
planes=500
views=3
def get_info(file):
    with open(file, 'rb') as info_file:
        info = info_file.readlines()
        truth = {}
        truth['NuPDG'] = int(info[7].strip())
        truth['NuEnergy'] = float(info[1])
        truth['LepEnergy'] = float(info[2])
        truth['Interaction'] = int(info[0].strip()) % 4
    return truth

def get_data_and_labels(files):
    data_maps = []
    data_labels = []
    for file in files:
        pdg = abs(get_info(file)['NuPDG'])
        if pdg == 1:
            truth_label = 0
        elif pdg == 12:
            truth_label = 1
        elif pdg == 14:
            truth_label = 2
        elif pdg == 16:
            truth_label = 3
        data_labels.append(truth_label)
        image = file[:-5]+'.gz' 
        data_maps.append(get_pixels_map(image))
    return data_maps, data_labels

def normalize(data):
    #Takes input of numpy array
    if not isinstance(data, np.ndarray):
        raise Exception('Must use array as input')
    if np.amax(data) == 0:
        return np.zeros(np.shape(data))
    data_range = np.amax(data)-np.amin(data)
    return (data - np.amin(data))/data_range

def get_pixels_map(file_name):
    file = open(file_name, 'rb').read()
    pixels_map = np.frombuffer(zlib.decompress(file), dtype=np.uint8 )
    pixels_map = pixels_map.reshape(views, planes, cells)
    return pixels_map

def plot_image(pixel_maps):
    fig, axs = plt.subplots(1, 3, figsize=(15, 10))
    fig.suptitle('Pixel Maps')        
    titles = ['U', 'V', 'Z']
    for i in range(3):
        maps = np.swapaxes(pixel_maps[i], 0, 1)
        axs[i].imshow(maps, interpolation='none', cmap = 'plasma')
        axs[i].set_xlabel('Wire')
        axs[i].set_ylabel('TDC')
        axs[i].title.set_text(titles[i])
    plt.show()
    
file_list_all = glob.glob(path)[:num_nu]
file_list = []
if include_tau != 'y':
    for f in file_list_all:
        if get_info(f)['NuPDG'] != 16 and get_info(f)['NuPDG'] != -16:
            file_list.append(f)
else:
    file_list = file_list_all

split = int(.8*len(file_list))
allfiles, testfiles = file_list[:split], file_list[split:]
random.shuffle(allfiles)

print()
print('Number of events in training set', len(allfiles), 'Test events', len(testfiles))
print()

#Create CNN (Not the news network)
model = models.Sequential()
model.add(layers.InputLayer(input_shape=(500, 500, n_channels)))
model.add(layers.Conv2D(256, (3, 3), activation='relu', data_format="channels_last"))
model.add(layers.AveragePooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu', data_format="channels_last"))
model.add(layers.AveragePooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu', data_format="channels_last"))
model.add(layers.AveragePooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu', data_format="channels_last"))
model.add(layers.AveragePooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu', data_format="channels_last"))
model.add(layers.AveragePooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu', data_format="channels_last"))
model.add(layers.AveragePooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu', data_format="channels_last"))
model.add(layers.AveragePooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.summary()

#Fitting data
model.compile(optimizer=optimizers.Adam(learning_rate = 1e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
#custom_callback = CustomCallback()
print()
print('Powering up generator...')
print()
    
partition = {'train':allfiles[:int(.8*len(allfiles))], 'validation':allfiles[int(.8*len(allfiles)):]}

params = {'batch_size':batch_size,'dim':(500, 500), 'n_channels':n_channels}

train_generator = DataGenerator(partition['train'], **params)
validation_generator = DataGenerator(partition['validation'], **params)
history = model.fit(train_generator, validation_data=validation_generator, epochs=num_epochs)

#Saving model summary
with open('/home/aupton/dune_cnn/model_save/'+test_name+'_summary.txt', 'w') as f:

    model.summary(print_fn=lambda x: f.write(x + '\n'))
    
#Save model so no need to retrain
model_path = '/home/aupton/dune_cnn/model_save/'+test_name
model.save(model_path)
print()
print('Model saved to: ' + model_path)

#Plotting Loss/Accuracy
plt.figure()
plt.title('Accuracy for %s Images'%(len(file_list)))
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0,1.1])
accuracy_path = '/home/aupton/dune_cnn/model_plot/'+test_name+'_accuracy.pdf'
plt.savefig(accuracy_path)
print('Accuracy plot saved to: ' + accuracy_path)

plt.figure()
plt.title('Loss for %s Images'%(len(file_list)))
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower left')
loss_path = '/home/aupton/dune_cnn/model_plot/'+test_name+'_loss.pdf'
plt.savefig(loss_path)
print('Loss plot saved to: ' + loss_path)


#Evaluation
print()
print('Evaluating accuracy/loss...')

test_labels = get_data_and_labels(testfiles)[1]

print('Made test labels')

test_generator = DataGenerator(testfiles, **params)
test_loss, test_acc = model.evaluate(test_generator, verbose=4)

print()
print('--------------------------------------------------------------------------------------------')
print('Accuracy %s'%(test_acc))
print('--------------------------------------------------------------------------------------------')

#Confusion Matrix
print()
print('Producing confusion matrix...')
print()
if include_tau == 'y':
    num_labels = 4
else:
    num_labels = 3

predictions = model.predict(test_generator)
predict = []
for i in range(len(predictions)):
    predict.append(list(predictions[i]).index(max(list(predictions[i]))))
con_mat = tf.math.confusion_matrix(np.asarray(test_labels[:len(predict)]), predict, num_classes = num_labels)
con_mat_norm = [con_mat[i]/sum(con_mat[i])*100 for i in range(len(con_mat))]
con_mat_norm = np.round(con_mat_norm, 2)
plt.figure()
for i in range(len(con_mat)):
    for j in range(len(con_mat[i])):
        plt.text(j-.2, i+.075, str(np.round(con_mat_norm[i][j], 1))+'%', c='k', backgroundcolor='w')
plt.imshow(con_mat_norm)
plt.xlabel('Predicted')
plt.ylabel('True')
if include_tau == 'y':
    labels = ['NC', 'e', 'mu', 'tau']
    plt.xticks([0, 1, 2, 3], labels)
    plt.yticks([0, 1, 2, 3], labels)
else:
    labels = ['NC', 'e', 'mu']
    plt.xticks([0, 1, 2], labels)
    plt.yticks([0, 1, 2], labels)
plt.colorbar(label='Percent of True Values Predicted')
plt.title('Normalized Confusion Matrix')
plt.savefig('/home/aupton/dune_cnn/model_plot/'+test_name+'_con_mat.pdf')
