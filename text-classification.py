# import datasetu a rozdělení textu
import pandas
dataset = pandas.read_csv('amazon_cells_labelled.txt', names=['sentence', 'label'], sep='\t')
sentences = dataset['sentence'].values
labels = dataset['label'].values

# rozdělení datasetu na trénovací data a testovací data
from sklearn.model_selection import train_test_split
sentences_train, sentences_test, labels_train, labels_test = train_test_split(sentences, labels, test_size=0.25, random_state=1000)

# předzpracování dat
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(binary=True)
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_test)
y_train = labels_train
y_test = labels_test

# vytvoření modelu neuronové sítě - vícevrstvý perceptron
from keras.models import Sequential
from keras import layers

# počet atributů
input_dim = X_train.shape[1]
print(input_dim)

# konfigurace/parametry sítě
activation_function = 'swish'
epochs = 30
optimizer = 'Nadam'

# vytvoření modelu sítě
model = Sequential()
model.add(layers.Dense(5, input_dim=input_dim, activation=activation_function))
model.add(layers.Dense(1, activation=activation_function))
model.compile(loss='hinge', optimizer=optimizer, metrics=['accuracy'])
model.summary()

# natrénování sítě
history = model.fit(X_train, y_train, epochs=epochs, verbose=True, batch_size=10)

# vyhodnocení sítě
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Správnost modelu na trénovacích datech: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Správnost modelu na testovacích datech:  {:.4f}".format(accuracy))

# vizualizace
from matplotlib import pyplot

print(history.history.keys())
accuracy_train = history.history['accuracy']
loss_train = history.history['loss']

epochs_accuracy = range(1, len(accuracy_train) + 1)
epochs_loss = range(1, len(loss_train) + 1)

# graf průběhu ztrátové funkce
pyplot.plot(epochs_loss, loss_train, 'bo', label='Ztráta při trénování')
pyplot.title('Průběh ztrátové funkce během trénování')
pyplot.xlabel('Epocha')
pyplot.ylabel('Hodnota ztrátové funkce')
pyplot.legend()
pyplot.savefig('out/text-classification-'+activation_function+'-'+optimizer+'-'+str(epochs)+'-loss'+'.png')
pyplot.clf()

# graf úspěšnosti při trénování
pyplot.plot(epochs_accuracy, accuracy_train, 'bo', label='Úspěšnost při trénování')
pyplot.title('Průběh úspěšnosti během trénování')
pyplot.xlabel('Epocha')
pyplot.ylabel('Úspěšnost')
pyplot.legend()
pyplot.savefig('out/text-classification-'+activation_function+'-'+optimizer+'-'+str(epochs)+'-accuracy'+'.png')
pyplot.clf()