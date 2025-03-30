import argparse
import vector_database
from langchain_openai import ChatOpenAI


def perform_query():

    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Load in db
    db = vector_database.load_vector_db()

    # Search the DB.
    results = vector_database.db_similarity_search(query=query_text, db=db)
    if results is None:
        print(f"Unable to find matching results.")
        return

    # Parse db chunks
    llm_promp = vector_database.parse_db_results(results, query_text)
    print(llm_promp)

    # LLM prediction
    model = ChatOpenAI()
    response_text = model.predict(llm_promp)
    print("LLM respone: ",response_text)


if __name__ == "__main__":
    perform_query()