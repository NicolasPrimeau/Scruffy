from pymongo import MongoClient
import datetime

DB_NAME = "AI2048"


def get_high_score(agent_name, db_name=DB_NAME):
    with MongoClient() as client:
        record = client[db_name].misc.find_one({"name": "high_score", "agent_name": agent_name})
        return record["value"] if record is not None else 0


def set_high_score(agent_name, score, db_name=DB_NAME):
    with MongoClient() as client:
        client[db_name].misc.update_one({"name": "high_score", "agent_name": str(agent_name)},
                                        {"$max": {"value": score}}, upsert=True)


def save_score(agent_name, score, db_name=DB_NAME):
    with MongoClient() as client:
        client[db_name][str(agent_name) + "_scores"].insert_one(
            {"reward": score, "time": datetime.datetime.now().timestamp()})
        client[db_name].misc.update_one({"name": "high_score", "agent_name": str(agent_name)},
                                        {"$max": {"value": score}}, upsert=True)


def get_scores(agent_name, db_name=DB_NAME):
    with MongoClient() as client:
        return client[db_name][str(agent_name) + "_scores"].find()


def scores_count(agent_name, db_name=DB_NAME):
    with MongoClient() as client:
        return client[db_name][str(agent_name) + "_scores"].count()


def save_error(agent_name, error, db_name=DB_NAME):
    with MongoClient() as client:
        client[db_name][str(agent_name) + "_errors"].insert_one(
            {"error": error, "time": datetime.datetime.now().timestamp()})
