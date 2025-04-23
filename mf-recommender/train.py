# Import NumPy and PyTorch
import numpy as np
import torch
import duckdb
import os
from itertools import product


# Import PyTorch Ignite
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss
from ignite.metrics import MeanSquaredError

# Import Tensorboard
from tensorboardX import SummaryWriter

# Import Utility Functions
from loader import Loader
from datetime import datetime

# Import the Model Script
from MFBiases import *
import pickle

def split_dataset(rating_matrix, test_proportion=0.2):
    from sklearn.model_selection import train_test_split
    import numpy as np
    # Sample data (features and labels)
    X = rating_matrix[:,0:2]
    y = rating_matrix[:,2].type(torch.float32)

    # Splitting dataset (80% train, 20% test)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=test_proportion, random_state=42)

    splitted_dataset = {
        'train_x': train_x,
        'train_y': train_y,
        'test_x': test_x,
        'test_y': test_y,
    }

    return splitted_dataset


def load_dataset_file():
    base_dir = 'the-movies-dataset/'
    ratings_table_file = base_dir + 'ratings_small.csv'
    
    db_path = "ratings.duckdb"
    # Check if the file exists and delete it
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Deleted existing database: {db_path}")

    # Connect and create a persistent database file
    con = duckdb.connect(db_path)
    
    # Create a persistent table (stored on disk, not memory)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS ratings AS 
        SELECT * FROM read_csv_auto('{ratings_table_file}')
    """)

    ## Add adicional ratings to ratings table - for evaluation in app

    # Get unique users and items
    unique_users = con.execute("SELECT DISTINCT userId FROM ratings").fetchall()
    unique_movies = con.execute("SELECT DISTINCT movieId FROM ratings").fetchall()
    num_ratings = con.execute("SELECT count(*) FROM ratings").fetchone()[0]

    # Create user and item index mappings
    user2idx = {user[0]: i for i, user in enumerate(unique_users)}
    movie2idx = {movie[0]: i for i, movie in enumerate(unique_movies)}

    num_users = len(unique_users)
    num_movies = len(unique_movies)

    # Initialize a user-item matrix with zeros
    rating_matrix = torch.zeros((num_ratings, 3), dtype=torch.int64)

    # Fetch ratings and fill the matrix
    ratings = con.execute("SELECT userId, movieId, rating FROM ratings").fetchall()

    i = 0
    for user_id, movie_id, rating in ratings:
        user_idx = user2idx[user_id]
        movie_idx = movie2idx[movie_id]
        # rating_matrix[user_idx, movie_idx] = rating
        rating_matrix[i][0] = user_idx
        rating_matrix[i][1] = movie_idx
        rating_matrix[i][2] = rating
        i+=1

    # Output
    splitted_dataset = split_dataset(rating_matrix)

    #Prepair reverse index
    idx2user_id = { idx: user_id for user_id, idx in user2idx.items()}
    idx2movie_id = { idx: movie_id for movie_id, idx in movie2idx.items()}

    splitted_dataset.update(
        {
            'n_users': num_users,
            'n_movies': num_movies,
            'idx2user': idx2user_id,
            'idx2movie': idx2movie_id,
            'all_ratings': rating_matrix,
        }
    )
    
    return splitted_dataset


def train(rating_dataset):
    # We have a bunch of feature columns and last column is the y-target
    # Note Pytorch is finicky about need int64 types
    train_x = rating_dataset['train_x']
    train_y = rating_dataset['train_y']

    # We've already split the data into train & test set
    test_x = rating_dataset['test_x']
    test_y = rating_dataset['test_y']

    # Extract the number of users and number of items
    n_users = int(rating_dataset['n_users'])
    n_movies = int(rating_dataset['n_movies'])

    # Define the Hyper-parameters
    lr = 1e-2  # Learning Rate
    k = 10  # Number of dimensions per user, item
    c_bias = 1e-6  # New parameter for regularizing bias
    c_vector = 1e-6  # Regularization constant

    # Setup logging
    log_dir = 'runs/simple_mf_02_bias_' + str(datetime.now()).replace(' ', '_')
    writer = SummaryWriter(log_dir=log_dir)

    # Instantiate the model class object
    model = MF(n_users, n_movies, writer=writer, k=k, c_bias=c_bias, c_vector=c_vector)
    
    def log_training_loss(engine, log_interval=500):
        """
        Function to log the training loss
        """
        model.itr = engine.state.iteration  # Keep track of iterations
        if model.itr % log_interval == 0:
            fmt = "Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
            # Keep track of epochs and outputs
            msg = fmt.format(engine.state.epoch, engine.state.iteration, len(train_loader), engine.state.output)
            print(msg)

    # Use Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create a supervised trainer
    trainer = create_supervised_trainer(model, optimizer, model.loss)

    # Use Mean Squared Error as evaluation metric
    metrics = {'evaluation': MeanSquaredError()}

    # Create a supervised evaluator
    evaluator = create_supervised_evaluator(model, metrics=metrics)

    def log_validation_results(engine):
        """
        Function to log the validation loss
        """
        # When triggered, run the validation set
        evaluator.run(test_loader)
        # Keep track of the evaluation metrics
        avg_loss = evaluator.state.metrics['evaluation']
        print("Epoch[{}] Validation MSE: {:.2f} ".format(engine.state.epoch, avg_loss))
        writer.add_scalar("validation/avg_loss", avg_loss, engine.state.epoch)

    # Load the train and test data
    train_loader = Loader(train_x, train_y, batchsize=1024)
    test_loader = Loader(test_x, test_y, batchsize=1024)

    trainer.add_event_handler(event_name=Events.ITERATION_COMPLETED, handler=log_training_loss)

    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=log_validation_results)

    # Run the model for 50 epochs
    trainer.run(train_loader, max_epochs=50)

    # Save the model to a separate folder
    torch.save(model.state_dict(), 'models/mf_biases.pth')

    return model


def predict_in_batches(model, inputs, batch_size=10):
    model.eval()  # Set model to evaluation mode
    predictions = []

    with torch.no_grad():  # Disable gradient tracking for inference
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]  # Slice the batch
            batch_pred = model(batch)  # Get predictions
            predictions.append(batch_pred)  # Store predictions

    return torch.cat(predictions, dim=0)  # Concatenate all batches into a single tensor


def prepair_all_recommendations(model, rating_dataset):

    n_users = rating_dataset['n_users']
    all_users_indexes = list(range(0,n_users))
    user_item_pairs = list(product(all_users_indexes, all_users_indexes))

    # Convert to PyTorch tensor
    tensor_pairs = torch.tensor(user_item_pairs, dtype=torch.int64)

    # Group by first element
    grouped_pairs = [tensor_pairs[tensor_pairs[:, 0] == x] for x in range(0,n_users)]

    predictions = {}
    for pairs in grouped_pairs:
        preds = predict_in_batches(model, pairs, batch_size=100)
        user_idx = int(pairs[0][0])
        predictions[user_idx] = preds
    
    idx2user = rating_dataset['idx2user']
    idx2movie = rating_dataset['idx2movie']
    mapped_preds = {}
    for user_idx, ratings in predictions.items():
        mapped_ratings = [(idx2movie[movie_idx], float(rating)) for movie_idx, rating in enumerate(ratings)]
        mapped_preds[idx2user[user_idx]] = sorted(mapped_ratings, key=lambda t: t[1], reverse=True)

    return mapped_preds


    
    
# Load preprocessed data
rating_dataset = load_dataset_file()
my_model = train(rating_dataset)
recommendations = prepair_all_recommendations(my_model, rating_dataset)

# Persist recommendations
with open("recs.pkl", "wb") as file:
    pickle.dump(recommendations, file)

print("Training Done.");
