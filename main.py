import os
import pickle
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union
import re


if not os.path.exists('../models'):
    os.makedirs('../models')
if not os.path.exists('../plots'):
    os.makedirs('../plots')


class DLModel:
    """
        Model Class to approximate the Z function as defined in the assignment.
    """

    def __init__(self):
        """Initialize the model."""
        self.Z0 = [None] * 10
        self.L = None
    
    def get_predictions(self, X, Z_0=None, w=10, L=None) -> np.ndarray:
        """Get the predictions for the given X values.

        Args:
            X (np.array): Array of overs remaining values.
            Z_0 (float, optional): Z_0 as defined in the assignment.
                                   Defaults to None.
            w (int, optional): Wickets in hand.
                               Defaults to 10.
            L (float, optional): L as defined in the assignment.
                                 Defaults to None.

        Returns:
            np.array: Predicted score possible
        """
        return Z_0*(1-np.exp(-L*X/Z_0))

    def calculate_loss(self, Params, X, Y, w=10) -> float:
        """ Calculate the loss for the given parameters and datapoints.
        Args:
            Params (list): List of parameters to be optimized.
            X (np.array): Array of overs remaining values.
            Y (np.array): Array of actual average score values.
            w (int, optional): Wickets in hand.
                               Defaults to 10.

        Returns:
            float: Mean Squared Error Loss for the model parameters 
                   over the given datapoints.
        """
        Z0= Params[w-1]
        L=Params[-1]
        predict = self.get_predictions(X, Z0,w,L)
        loss = np.sum((Y - predict)**2)
        return loss

    
    def save(self, path):
        """Save the model to the given path.

        Args:
            path (str): Location to save the model.
        """
        with open(path, 'wb') as f:
            pickle.dump((self.L, self.Z0), f)
    
    def load(self, path):
        """Load the model from the given path.

        Args:
            path (str): Location to load the model.
        """
        with open(path, 'rb') as f:
            (self.L, self.Z0) = pickle.load(f)


def get_data(data_path) -> Union[pd.DataFrame, np.ndarray]:
    """
    Loads the data from the given path and returns a pandas dataframe.

    Args:
        path (str): Path to the data file.

    Returns:
        pd.DataFrame, np.ndarray: Data Structure containing the loaded data
    """
    data = pd.read_csv(data_path)
    return data


def preprocess_data(data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
    """Preprocesses the dataframe by
    (i)   removing the unnecessary columns,
    (ii)  loading date in proper format DD-MM-YYYY,
    (iii) removing the rows with missing values,
    (iv)  anything else you feel is required for training your model.

    Args:
        data (pd.DataFrame, nd.ndarray): Pandas dataframe containing the loaded data

    Returns:
        pd.DataFrame, np.ndarray: Datastructure containing the cleaned data.
    """
    #changing incorrect Date Format
    months = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}

    for idx in data.index:
        date = str(data['Date'][idx])
        if not re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', date):
            new_str = date.split(" ")
            dd = new_str[1].split("-")[0]
            if len(dd) == 1:
                dd = "0" + dd
            mm = months[new_str[0]]
            yr = new_str[2]
            data.loc[:,('Date','idx')] = str(dd) + "/" + str(mm) + "/" + str(yr)

    data = data[data['Error.In.Data'] == 0]
    Inn_data = data[data['Innings'] == 1]
    Inn_data = Inn_data[['Over','Runs', 'Wickets.in.Hand','Total.Runs', 'Runs.Remaining']]
    Inn_data['over.remaining'] = 50
    Inn_data['over.remaining'] = Inn_data['over.remaining'] - Inn_data['Over']

    return Inn_data


def train_model(Inn_data: Union[pd.DataFrame, np.ndarray], model: DLModel) -> DLModel:
    """Trains the model

    Args:
        data (pd.DataFrame, np.ndarray): Datastructure containg the cleaned data
        model (DLModel): Model to be trained
    """
    l = 0
    param = [1]*10
    param.append(l)
    for w in range(0,10):
        groupbywick = Inn_data[Inn_data['Wickets.in.Hand'] == w+1]
        Data = pd.DataFrame({'X': groupbywick['over.remaining'], 'y': groupbywick['Runs.Remaining']})

        X,y = Data['X'],Data['y']

        opt_param=sp.optimize.minimize(model.calculate_loss ,x0=param, args=(X,y,w+1),method='L-BFGS-B')
        param[w]=opt_param.x[w]
        param[-1]=opt_param.x[-1]
    model.Z0 = param[:-1]
    model.L = param[-1]

    return model


def plot(model: DLModel, plot_path: str) -> None:
    """ Plots the model predictions against the number of overs
        remaining according to wickets in hand.

    Args:
        model (DLModel): Trained model
        plot_path (str): Path to save the plot
    """
    fig = plt.subplots(figsize=(8, 6))
    for i in range(9,-1,-1):
        over = np.linspace(0, 50, num=100)
        predict = model.get_predictions(over, model.Z0[i], i, model.L)
        plt.plot(over,predict)

    plt.xlabel('Overs remaining')
    plt.ylabel('Average Runs')
    plt.xlim((0, 50))
    plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    plt.grid()
    plt.title("The Duckworth-Lewis Method ")
    plt.legend(['w=10', 'w=9', 'w=8', 'w=7', 'w=6', 'w=5', 'w=4', 'w=3', 'w=2', 'w=1'])
    plt.savefig(plot_path)
    plt.show()

def print_model_params(model: DLModel) -> List[float]:
    '''
    Prints the 11 (Z_0(1), ..., Z_0(10), L) model parameters

    Args:
        model (DLModel): Trained model
    
    Returns:
        array: 11 model parameters (Z_0(1), ..., Z_0(10), L)

    '''
    print("11 Model Parameters :-")
    print(f"Z_0: {model.Z0}")
    print(f"L: {model.L}")
    
    parameters = [model.Z0 , model.L]
    
    return parameters


def calculate_loss(model: DLModel, Inn_data: Union[pd.DataFrame, np.ndarray]) -> float:
    '''
    Calculates the normalised squared error loss for the given model and data

    Args:
        model (DLModel): Trained model
        data (pd.DataFrame or np.ndarray): Data to calculate the loss on
    
    Returns:
        float: Normalised squared error loss for the given model and data
    '''
    
    param = model.Z0
    param.append(model.L)
    size=0
    mse=0
    
    for w in range(0,10):
        groupbywick = Inn_data[Inn_data['Wickets.in.Hand'] == w+1]
        Data = pd.DataFrame({'X': groupbywick['over.remaining'], 'y': groupbywick['Runs.Remaining']})
        X,y = Data['X'],Data['y']
        
        #cumulate loss over all datapoints
        mse+=model.calculate_loss(param,X,y,w+1)
        size+=len(X)
        
    #normalize loss
    mse/=size
    print('MSE: ',mse)
   
    return mse


def main(args):
    """Main Function"""

    data = get_data(args['data_path'])  # Loading the data
    print("Data loaded.")
    
    # Preprocess the data
    data = preprocess_data(data)
    print("Data preprocessed.")
    
    model = DLModel()  # Initializing the model
    model = train_model(data, model)  # Training the model
    #model.save(args['model_path'])  # Saving the model
    
    plot(model, args['plot_path'])  # Plotting the model
    
    # Printing the model parameters
    print_model_params(model)

    # Calculate the normalised squared error
    calculate_loss(model, data)

if __name__ == '__main__':
    args = {
        "data_path": "../data/04_cricket_1999to2011.csv",
        "model_path": "../models/model.pkl",  # ensure that the path exists
        "plot_path": "../plots/plot.png",  # ensure that the path exists
    }
    main(args)
