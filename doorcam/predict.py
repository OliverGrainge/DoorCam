import datetime
from datetime import timedelta
from glob import glob
from typing import Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import NearestNeighbors

from train import TripletModel


def extract_name_from_path(path: str) -> str:
    """
    Extracts a name from a given file path.

    This function takes a file path as input and returns the name of the file
    without its extension. It assumes that the file name is the last part of the
    path after the last '/' character and that the file extension (if present)
    is the part after the last '.' character in the file name.

    Args:
        path (str): The full path of the file from which the name is to be extracted.

    Returns:
        str: The extracted name of the file without its extension.
    """
    name_with_extension = path.split("/")[-1]
    name = name_with_extension.split(".")[0]
    return name


class Identifier:
    """
    A class to identify individuals using a triplet model and maintain an entry record.

    Attributes:
        config (dict): Configuration settings for the identifier.
        embeddings (np.ndarray): Loaded embeddings from the identification database.
        names (np.ndarray): Names corresponding to the embeddings.
        model (TripletModel): The trained model for generating embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        if config["inference"]["model_name"] == "":
            
            self.model = TripletModel()
        else:
            self.model = TripletModel.load_from_checkpoint(
                "model_checkpoints/" + config["inference"]["model_name"]
            )
        self.embeddings, self.names = self.load_id_database()

    def inference(self, x: Image) -> Union[dict, None]:
        """
        Performs inference to identify an individual from an image.

        Args:
            x (Image): The image of the individual to identify.

        Returns:
            dict or None: A dictionary with the identified individual's name and timestamp if identified,
                          otherwise, a dictionary with 'unknown' and timestamp.
        """

        features = self.model.predict(x)
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(self.embeddings)
        distances, indices = nn.kneighbors([features])
        if distances[0][0] <= self.config["inference"]["similarity_threshold"]:
            name = self.names[indices[0][0]]
            timestamp = datetime.datetime.utcnow()
            entry = {"Name": [name], "Timestamp": [timestamp]}
            return entry
        else:
            timestamp = datetime.datetime.utcnow()
            entry = {"Name": ["unknown"], "Timestamp": [timestamp]}
            return entry

    def load_id_database(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads the identification database comprising images and their corresponding names.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing arrays of embeddings and names.
        """

        images = glob("static/people/*")
        names = [extract_name_from_path(pth) for pth in images]
        images = [Image.open(img) for img in images]
        embeddings = np.vstack([self.model.predict(img) for img in images])
        return embeddings, names

    def load_record(self) -> pd.DataFrame:
        """
        Loads the entry record from a CSV file, or creates a new DataFrame if the file does not exist.

        Returns:
            pd.DataFrame: The loaded or newly created entry record DataFrame.
        """
        try:
            record = pd.read_csv("entry_record.csv")
            record['Timestamp'] = pd.to_datetime(record['Timestamp'])
            record = record.head(20)
        except:
            timestamp = datetime.datetime.utcnow()
            record_dict = {"Name": ["oliver grainge"], "Timestamp":[timestamp]}
            record = pd.DataFrame.from_dict(record_dict)
            record['Timestamp'] = pd.to_datetime(record['Timestamp'])
        return record

    def save_record(self, record: pd.DataFrame) -> None:
        """
        Saves the entry record to a CSV file.

        Args:
            record (pd.DataFrame): The entry record DataFrame to save.
        """
        record = record.head(20)
        record['Timestamp'] = pd.to_datetime(record['Timestamp'])
        one_week_ago = datetime.datetime.utcnow() - timedelta(weeks=1)
        record = record[record['Timestamp'] > one_week_ago]
        record.to_csv("entry_record.csv", index=False)

    def add_entryrecord(self, entry: dict) -> None:
        """
        Adds a new entry to the entry record.

        Args:
            entry (dict): The entry dictionary to add to the record.
        """
        new_record = pd.DataFrame.from_dict(entry)
        record = pd.concat([self.load_record(), new_record])
        self.save_record(record)

    def add_person(self, name: str, x: Image) -> None:
        """
        Saves an image of a person with the specified name.

        This function takes a name and a PIL Image object as input and saves the image
        in a designated directory with the person's name as the filename.

        Args:
            name (str): The name of the person. This name is used as the filename.
            x (Image): The PIL Image object of the person to be saved.

        Returns:
            None: This function does not return anything. It saves the image to disk.
        """
        x.save("statice/people/" + name + ".jpg")
