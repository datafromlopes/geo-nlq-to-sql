from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class PostgreSQL:
    def __init__(self, user, password, database):
        self.__engine = create_engine(f"postgresql://{user}:{password}@localhost/{database}")
        self.__session = sessionmaker(bind=self.__engine)

    def get_session(self):
        return self.__session()

    def get_engine(self):
        return self.__engine

    def execute_query(self, query):
        with self.__engine.connect() as connection:
            result = connection.execute(query)
            return result.fetchall()

    def close(self):
        self.__engine.dispose()
