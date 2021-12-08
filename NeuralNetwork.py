import numpy
import scipy.special

class neuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes 
        self.onodes = outputnodes

        # Матрицы весовых коэффициентов связей wih и who.
        # Весовые коэффициенты связей между узлом i и узлом j
        # следующего слоя обозначены как w__i:
        # wll w21
        # wl2 w22 и т.д.
        
        try:
            self.wih = numpy.load('WIH.npy')
            self.who = numpy.load('WHO.npy')
        except: #если нет сохраненных весов
            self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5) 
            self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)

        self.lr = learningrate

        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list, targets_list) :

        inputs = numpy.array(inputs_list, ndmin=2).T 
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)

        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)

        final_outputs = self.activation_function(final_inputs)
        output_errors = targets - final_outputs

        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        pass

    def save_weights(self):
        numpy.save('WIH', self.wih)
        numpy.save('WHO', self.who)

    # опрос нейронной сети
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

def main():
    input_nodes = 784
    hidden_nodes = 100 
    output_nodes = 10
    learning_rate = 0.3

    n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
     
    training_data_file = open("mnist_train.csv", 'r') #файл с тренировочными данными 
    training_data_list = training_data_file.readlines () 
    training_data_file.close()

    for record in training_data_list:

        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] =0.99
        n.train(inputs, targets) 
        pass
    n.save_weights()
if __name__ == "__main__":
    main()