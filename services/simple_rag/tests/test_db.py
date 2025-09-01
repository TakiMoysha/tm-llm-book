import pytest
import chromadb

from db import chroma_client


@pytest.fixture(autouse=True, name="chroma_test_collection")
def _chrome_client_intialize():
    collection = chroma_client.create_collection(name="test_collection", metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        documents=[
            "Bananas are radioactive: because they contain potassium, a tiny fraction of which is the natural radioactive isotope potassium-40. ",
            "Octopuses have three hearts: and blue blood.                                                                                       ",
            "Earth is squashed: at the poles and bulges at the equator, a shape often described as having a “waistline”.                        ",
            "The Mona Lisa has no eyebrows .                                                                                                    ",
            "Cows are sacred: in India, where they hold significant importance in the primary religions.                                        ",
            "Hot water can freeze faster than cold water . This phenomenon is known as the Mpemba effect.                                       ",
            "A day on Venus is longer than its year .                                                                                           ",
            "About 10% of the population is left-handed .                                                                                       ",
            "Australia is wider than the Moon .                                                                                                 ",
            "Besides water, tea is the most popular beverage in the world .                                                                     ",
        ],
    )


def test_chroma_query():
    collection = chroma_client.collection("test_collection")
    result = collection.query(query_texts=["What is the capital of France?"], n_results=3)
    print(result)
    assert True
