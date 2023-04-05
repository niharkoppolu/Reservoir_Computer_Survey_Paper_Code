"""
Project: RC Frameworks: 1-Step Prediction Comparison

This program Completes a timing analysis of different RC Frameworks completing a 1-step prediction task using the Mackey Glass dataset.
The program is organized like so: 

1. Data_Processor function: function used to process the Mackey_Glass data and turn it into training and testing data that can be used by the RCs

2. Reservoir_Framework class: Abstract Base Class used as structure for child classes. Each child class implements a RC using a different framework. This class contains the implementation of:
    - Find_Min_Timesteps: used to find the minimum number of training timesteps it takes to reach 0.01 nrmse
    - Time_Training: used to find the amount of time it takes to train the RC to reach 0.01 nrmse


3. Each of the following is a child class used to implement RC with the framework specified. 
    - 3a. ReservoirPy
    - 3b. EasyESN (DOESNT WORK)
    - 3c. pyRCN


Tlab Meeting Notes:

1. - Use a monte carlo simulation to determine overall training time, time per timestep over 100's of runs: Complete

2. - Do same comparison with same task 1000 nodes instead of 100: as the reservoir grows should 

3. - After doing the 2 1-step pred with 100 and 1000 nodes, try more complex task 
"""




#___________________________________________________________________________________________________________
#___________________________________________________________________________________________________________
#imports used by all classes (No Framework imports)
import numpy as np
import matplotlib.pyplot as plt
#from easyesn import PredictionESN
#import timeit #used to time code
import time #used to time code
from reservoirpy.observables import mse, nrmse, rmse
import statistics
from abc import ABC, abstractmethod

#Global Variables
Mackey_Glass_txt_FilePath = "Mackey_Glass.txt"


#___________________________________________________________________________________________________________
#___________________________________________________________________________________________________________
#Creating Data Set
def Data_Processor():
    #Reading in data as list of strings (each index is 1 line in the file)
    data_file = open(Mackey_Glass_txt_FilePath, "r")
    raw_data_strings = data_file.readlines()
    float_data = []

    for str_number in raw_data_strings:
        float_data.append(float(str_number.split("\n")[0]))

    #turning mackey glass array into numpy array
    mackey_glass_final = np.asarray(float_data).reshape(-1,1) #reservoirpy expects array to be in a specific shape
    print(mackey_glass_final.shape) #debug

    #Print Data Set as Graph
    
    plt.figure(figsize=(10, 3))
    plt.title("Mackey Glass Data")
    plt.xlabel("First 200 Timesteps (10000 total in dataset)")
    plt.plot(mackey_glass_final[0:200], label="Mackey_Glass_Data_Set", color="blue")
    plt.legend()
    plt.show()

    return mackey_glass_final

#Create global that Reservoir_Framework can use
mackey_glass_final = Data_Processor()


#___________________________________________________________________________________________________________
#___________________________________________________________________________________________________________
#Super class for all Reservoir Frameworks
class Reservoir_Framework(ABC): #must pass in ABC parameter to make class abstract

    def __init__(self, Set_Num_Nodes):
        self.training_timesteps = 100
        self.test_set_begin = 9000
        self.test_set_end = self.test_set_begin + 500

        self.X_train = mackey_glass_final[0:self.training_timesteps] 
        self.Y_train = mackey_glass_final[1:self.training_timesteps + 1] 

        self.calculated_nrmse = -1
        self.calculated_mse = -1

        self.reservoir_nodes = Set_Num_Nodes
        self.leakage_rate = 0.5
        self.spectral_radius = 0.9

        #used for data collection
        self.av_timestep = -1
        self.stand_dev_timestep = -1

        self.av_training_time = -1
        self.stand_dev_train_time = -1
        pass

#abstract methods
    @abstractmethod
    def Train(self):
        pass

    @abstractmethod
    def Test(self):
        pass

    @abstractmethod
    def Reset_Reservoir(self):
        pass

#END of abstract methods
#_________________________________________________________________________________________________________________________________________________

#Non-Abstract methods


    #This function is used to find the minimum number of timesteps it takes to Train to 100% accuracy
    def Find_Min_Timesteps(self):
        
        lowest_possible_timesteps_flag = False #this is set to true when reaching lowest number of timesteps with 0.01 nrmse
        nrmse_0_01_flag = False #used to check nrmse of previous_timestep training was 0.01 when the current nrmse is greater

        previous_timesteps = -1 #prev. number of timesteps used to train previous reservoir

        while lowest_possible_timesteps_flag == False:
            self.Reset_Reservoir()
            self.Train()
            self.calculated_nrmse = self.Test()
            #DEBUG
            print("\nnrmse: ", self.calculated_nrmse)

            #if it hasn't reached required accuracy add more timesteps, OR if previous timesteps had reached required accuracy exit
            #if training timesteps is not
            if self.calculated_nrmse > 0.01 and self.training_timesteps < 2000:
                if nrmse_0_01_flag == True:
                    self.training_timesteps = previous_timesteps
                    lowest_possible_timesteps_flag = True
                    pass

                else:    
                    previous_timesteps = self.training_timesteps
                    self.training_timesteps = self.training_timesteps + 50
                    pass

                pass



            #if it has reached required accuracy check to see if less timesteps would still work 
            else:
                nrmse_0_01_flag = True
                previous_timesteps = self.training_timesteps
                self.training_timesteps = self.training_timesteps - 1
                pass

            #DEBUG
            """
            if self.calculated_nrmse < 0.1:
                lowest_possible_timesteps_flag = True
                print("\nnrmse: " + str(self.calculated_nrmse))
            """


            pass
        #will not be used in paper, but will be used by to determine if same timesteps for all RC 
        print("Personal Metric: The minimum no. of timesteps it takes to reach accuracy of 0.01 nrmse is: " + str(self.training_timesteps))
        return self.training_timesteps



#_________________________________________________________________________________________________________________________________________________
                

    #may implement in base  
    #Only call after completing Find_Min_Timesteps
    def Time_Training(self):

        #overall_training_time = timeit.timeit(stmt="self.Train()", globals=globals(), number=1)

        #used to time Training (not sure if I should be using this method)
        #link to methods for finding Execution Time: https://pynative.com/python-get-execution-time-of-program/
        
        
        #--------Monte Carlo Simulation for Calculating overall training time------#
        self.Reset_Reservoir()

        monte_carlo_sim_size = 10 #set to 1000 when generating data
        overall_training_time_array = [0] * monte_carlo_sim_size
        print(len(overall_training_time_array))

        i = 0
        while i < monte_carlo_sim_size:
            #May need to add Reset Call here
            start_time = time.process_time()
            self.Train()
            end_time = time.process_time()
            overall_training_time_array[i] = end_time - start_time
            i = i + 1
            pass
        #--------Monte Carlo Simulation for Calculating overall training time------#

        print("Training time of all trials before averaging: ",overall_training_time_array) #will comment out when generating data for paper
        print("Standard Deviation all trials: ", statistics.stdev(overall_training_time_array))
        
        train_step_array = np.array(overall_training_time_array) / self.training_timesteps
        self.av_training_time = sum(overall_training_time_array) / len(overall_training_time_array)


        print("\nRelevent: It takes ", self.av_training_time, " to train  to 0.01 nrmse (over ", self.training_timesteps, " timesteps.)\n")
        self.stand_dev_train_time = np.std(overall_training_time_array)
        print("std dev of Total Trainind Time: ", self.stand_dev_train_time)

        self.av_timestep = self.av_training_time / self.training_timesteps
        print("\n\nRelevent: Time per Train step: ", self.av_timestep, "\n")
        
        self.stand_dev_timestep = np.std(train_step_array)
        print("std dev of Time per Trainstep: ", self.stand_dev_timestep)
        

        return self.av_training_time


#___________________________________________________________________________________________________________
#___________________________________________________________________________________________________________
#Framework 1: ReservoirPy (import of framework is placed in init)
class ReservoirPy(Reservoir_Framework): #superclass containing in brackets

    def __init__(self, Set_Num_Nodes):
        from reservoirpy.nodes import Reservoir, Ridge
        super().__init__(Set_Num_Nodes) #must call init function of superclass
        

        self.reservoir = Reservoir(self.reservoir_nodes, lr = self.leakage_rate, sr = self.spectral_radius)
        self.readout = Ridge(ridge=1e-7)

        self.esn_model = self.reservoir >> self.readout #connects reservoir and ridge

        pass

    #override abstract Train method
    def Train(self):
        
        self.X_train = mackey_glass_final[0:self.training_timesteps] #must add self for class variables
        self.Y_train = mackey_glass_final[1:self.training_timesteps + 1] 

        self.esn_model = self.esn_model.fit(self.X_train, self.Y_train, warmup=10) #training RC
        #print(self.reservoir.is_initialized, self.readout.is_initialized, self.readout.fitted) #used to check if training done

        pass

    
    #override abstract Test method
    def Test(self):

        Y_pred = self.esn_model.run(mackey_glass_final[self.test_set_begin : self.test_set_end])
        Y_actual = mackey_glass_final[self.test_set_begin + 1 : self.test_set_end + 1]

        #used to graph
        """
        plt.figure(figsize=(10, 3))
        plt.title("Predicted and Actual Mackey_Glass Timeseries.")
        plt.xlabel("$t$")
        plt.plot(Y_pred, label="Predicted ", color="blue")
        plt.plot(Y_actual, label="Real ", color="red")
        plt.legend()
        plt.show()
        """
        

        #this is used to calculate nrmse and mse
        self.calculated_nrmse = nrmse(Y_pred, Y_actual)
        self.calculated_mse = (Y_pred, Y_actual)

        return self.calculated_nrmse


    #override abstract Reset method
    def Reset_Reservoir(self):
        from reservoirpy.nodes import Ridge

        self.readout = Ridge(ridge=1e-7)
        self.esn_model = self.reservoir >> self.readout #connects reservoir and ridge

        pass


#___________________________________________________________________________________________________________
#___________________________________________________________________________________________________________
#Framework 2: pyESN
class pyESN(Reservoir_Framework):
    
    def __init__(self):
        import pyESN
        super().__init__() #must call init function of superclass
        
        self.esn_model = pyESN.ESN(n_inputs = 1, n_outputs = 1, n_reservoir = self.reservoir_nodes, 
            spectral_radius = self.spectral_radius, random_state=42)

        


        pass

    #override abstract Train method
    def Train(self):
        
        self.X_train = mackey_glass_final[0:self.training_timesteps] #must add self for class variables
        self.Y_train = mackey_glass_final[1:self.training_timesteps + 1] 

        #pred_training = esn.fit(np.ones(trainlen),data[:trainlen]) 
        #This is used to predict entire Mackey Glass Series.
        #Ones are inserted as input because the input doesn't matter, the goal is to learn the series
        
        pred_training = self.esn_model.fit(self.X_train, self.Y_train) #This is used for 1-step timestep prediction
        #pred_training variable isn't used for much
        pass

    
    #override abstract Test method
    def Test(self):

        Y_pred = self.esn_model.predict(mackey_glass_final[self.test_set_begin : self.test_set_end])
        Y_actual = mackey_glass_final[self.test_set_begin + 1 : self.test_set_end + 1]

        #used to graph
        """
        plt.figure(figsize=(10, 3))
        plt.title("Predicted and Actual Mackey_Glass Timeseries.")
        plt.xlabel("$t$")
        plt.plot(Y_pred, label="Predicted ", color="blue")
        plt.plot(Y_actual, label="Real ", color="red")
        plt.legend()
        plt.show()
        """

        #this is used to calculate nrmse and mse
        self.calculated_nrmse = nrmse(Y_pred, Y_actual)
        self.calculated_mse = (Y_pred, Y_actual)

        return self.calculated_nrmse


    #override abstract Reset method
    def Reset_Reservoir(self):
        import pyESN

        self.esn_model = pyESN.ESN(n_inputs = 1, n_outputs = 1, n_reservoir = self.reservoir_nodes, 
            spectral_radius = self.spectral_radius, random_state=42)
        pass


#___________________________________________________________________________________________________________
#___________________________________________________________________________________________________________
#Framework 3: pyRCN (import of framework is placed in init)
class pyRCN(Reservoir_Framework): #superclass containing in brackets

    def __init__(self, Set_Num_Nodes):
        from pyrcn.echo_state_network import ESNRegressor, ESNClassifier
        from sklearn.linear_model import Ridge

        super().__init__(Set_Num_Nodes) #must call init function of superclass

        # Hyperparameter optimization ESN
        initially_fixed_params = {'hidden_layer_size': self.reservoir_nodes,
                          'input_activation': 'identity',
                          'bias_scaling': 0.0,
                          'reservoir_activation': 'tanh',
                          'leakage': self.leakage_rate,
                          'bidirectional': False,
                          'k_rec': 10,
                          'wash_out': 0,
                          'continuation': False,
                          'alpha': 1e-5,
                          'random_state': 42,
                          'requires_sequence': False,
                          'spectral_radius': self.spectral_radius}

        self.esn = ESNRegressor(regressor=Ridge(), **initially_fixed_params) 
        #old self.esn = ESNRegressor()
        self.esn
        pass

    #override abstract Train method
    def Train(self):
        
        self.X_train = mackey_glass_final[0:self.training_timesteps] #must add self for class variables
        self.Y_train = mackey_glass_final[1:self.training_timesteps + 1] 

        self.esn.fit(self.X_train.reshape(-1, 1), self.Y_train)
        pass

    
    #override abstract Test method
    def Test(self):

        Y_pred = self.esn.predict(mackey_glass_final[self.test_set_begin : self.test_set_end])
        Y_actual = mackey_glass_final[self.test_set_begin + 1 : self.test_set_end + 1]
        """
        #used to graph
        
        plt.figure(figsize=(10, 3))
        plt.title("Predicted and Actual Mackey_Glass Timeseries.")
        plt.xlabel("$t$")
        plt.plot(Y_pred, label="Predicted ", color="blue")
        #plt.plot(Y_actual, label="Real ", color="red")
        plt.legend()
        plt.show()
        """
        
        

        #this is used to calculate nrmse and mse
        self.calculated_nrmse = nrmse(Y_pred, Y_actual)
        self.calculated_mse = (Y_pred, Y_actual)

        return self.calculated_nrmse


    #override abstract Reset method
    def Reset_Reservoir(self):

        from pyrcn.echo_state_network import ESNRegressor, ESNClassifier
        from sklearn.linear_model import Ridge
        initially_fixed_params = {'hidden_layer_size': self.reservoir_nodes,
                          'input_activation': 'identity',
                          'bias_scaling': 0.0,
                          'reservoir_activation': 'tanh',
                          'leakage': self.leakage_rate,
                          'bidirectional': False,
                          'k_rec': 10,
                          'wash_out': 0,
                          'continuation': False,
                          'alpha': 1e-5,
                          'random_state': 42,
                          'requires_sequence': False,
                          'spectral_radius': self.spectral_radius}

        self.esn = ESNRegressor(regressor=Ridge(), **initially_fixed_params) 

        pass

#___________________________________________________________________________________________________________
#___________________________________________________________________________________________________________
#This function is used separately to Run Tests on Frameworks to determine length of Timesteps
def Time_Step_Tests(Framework_RC, Framework_Name):

    Res_Size = [100, 500, 1000, 5000]
    #Test Below
    #Res_Size = [100, 200]

    #Comprehensive_Framework_List = [[ReservoirPy(Res_Size[0]), ReservoirPy(Res_Size[1]), ReservoirPy(Res_Size[2]), ReservoirPy(Res_Size[3])],
    #                                [pyRCN(Res_Size[0]), pyRCN(Res_Size[1]), pyRCN(Res_Size[2]), pyRCN(Res_Size[3])]]

    Av_Timestep_List = []
    Timestep_Error = []

    Av_Time_List = []
    Total_Time_Error = []

    for res_size in Res_Size:
        
        print("\n\n\n\n_______________________________________________________________________________________________")
        print("_______________________________________________________________________________________________")
        print(res_size, " Node Reservoir:\n\n")
        reservoir = Framework_RC(res_size)
        reservoir.Find_Min_Timesteps()
        #link to methods for finding Execution Time: https://pynative.com/python-get-execution-time-of-program/
        reservoir.Time_Training()

        Av_Timestep_List.append(reservoir.av_timestep)
        Timestep_Error.append(reservoir.stand_dev_timestep)

        Av_Time_List.append(reservoir.av_training_time)
        Total_Time_Error.append(reservoir.stand_dev_train_time)
        pass


    #___________________________________________________________________________________________________________
    #Graph of Training Timesteps of different Reservoir Computer Sizes
    print("\n\n\n\n_______________________________________________________________________________________________")
    print("_______________________________________________________________________________________________")
    print(Framework_Name + "Graph 1: Training Timesteps\n\n")


    #https://problemsolvingwithpython.com/06-Plotting-with-Matplotlib/06.07-Error-Bars/ : used this for creating plot
    #____________________________Graph 1: Timestep of Dif. Reservoir Sizes_______
    fig, ax = plt.subplots()

    ax.errorbar(Res_Size, Av_Timestep_List,
                yerr=Timestep_Error,
                alpha=0.5,
                ecolor='black',
                capsize=10)

    ax.set_xlabel('Reservoir Size')
    ax.set_ylabel('Time per Trainstep (Sec')
    ax.set_title(Framework_Name + ' Graph 1: Training Timestep Length')

    plt.savefig('Data_FrameworkComp/' + Framework_Name + '_Graph 1:_time_step_plot.png')
    
    #____________________________Graph 2: Total Training Time of Dif. Reservoir Sizes_______
    print(Framework_Name + "Graph 2: Total Training Time\n\n")
    fig, ax = plt.subplots()

    ax.errorbar(Res_Size, Av_Time_List,
                yerr=Total_Time_Error,
                alpha=0.5,
                ecolor='black',
                capsize=10)


    ax.set_xlabel('Reservoir Size')
    ax.set_ylabel('Total Train Time (Sec)')
    ax.set_title(Framework_Name + ' Graph 2: Total Training Time Length')


    plt.show
    plt.savefig('Data_FrameworkComp/' + Framework_Name + '_Graph_2:_total-time_plot.png')
    pass


#___________________________________________________________________________________________________________
#___________________________________________________________________________________________________________
#Main Point of Execution

Framework_dict = {
  "ReservoirPy": ReservoirPy,
  "pyRCN": pyRCN,
}

print("\n\nStart ReservoirPy Tests\n\n")
#This completes reservoirpy tests
Time_Step_Tests(Framework_dict["ReservoirPy"], 'ReservoirPy')

#Need to Do some debugging of pyRCN code
print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n_______________________________________________________________________________________________")
print("_______________________________________________________________________________________________")
print("\n\nStart pyRCN Tests\n\n")
Time_Step_Tests(Framework_dict["pyRCN"], 'pyRCN')

"""
#Testing pyRCN: Not done yet - Need to Find way how to set Reservoir sizes
print("\n\nTesting pyRCN\n\n")
pyRCN_test = pyRCN(100)
pyRCN_test.Train()
pyRCN_test.Test()
"""