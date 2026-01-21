import argparse
import asyncio

from .infra import MessageRepository


async def main():
    message_repository = MessageRepository()

    await message_repository.get_messages_content(limit=200, offset=0)


# ===========================================================
# TESTS
# ===========================================================
import pytest
import tiktoken


@pytest.mark.target
async def test_count_message_tokens():
    ...



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    asyncio.run(main())





import sqlite3
import json

def convert_types(value):
    if value.startswith('{') and value.endswith('}'):
        try:
            return json.loads(value)
        except:
            return value
    return value

def type_aware_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        value = row[idx]
        if isinstance(value, str):
            value = convert_types(value)
        d[col[0]] = value
    return d

with sqlite3.connect(':memory:') as conn:
    conn.row_factory = lambda cursor, row: (row[0],) 
    conn.execute('CREATE TABLE config(id INTEGER, settings TEXT)')
    conn.execute('INSERT INTO config VALUES(1, \'{"theme":"dark"}\')')
    
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM config')
    config = cursor.fetchone()
    
    print(config['settings']['theme'])  # 'dark' (automatically parsed JSON)

