from pymongo import MongoClient
from pymongo.errors import ConnectionFailure


class Connection:
    """Class docstring"""
    def __init__(self, username, password) -> None:
        self.username = username
        self.password = password
        self.host = None
        self.port = None
        self.db_name = None

    def connect(self, db_name) -> MongoClient:
        """Method docstring"""
        try:
            self.db_name = db_name
            client = MongoClient(f"mongodb+srv://{self.username}:{self.password}@cluster0.ytnpg.mongodb.net/?retry"
                                 f"Writes=true&w=majority")
            db_connection = client[f"{self.db_name}"]
        except ConnectionFailure:
            print(f"Error while  trying to connect - {ConnectionFailure}")

        return db_connection