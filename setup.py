import numpy as np
import sqlite3
import json
import re
import requests

from bs4 import BeautifulSoup

program_data_input = "data/program.txt"
course_data_input = "data/courses.txt"

############################
#  Data sourcing/scraping  #
############################
def make_data():
    ## Scrape degree content
    page = requests.get("https://ai.meng.duke.edu/degree")
    soup = BeautifulSoup(page.content, 'html.parser')
    degree = soup.find('div', id='content-body')
    degree_data = degree.get_text()
    degree_data = re.sub(r'\n\n', '\n', degree_data)

    ## Scrape course data
    page = requests.get("https://ai.meng.duke.edu/courses")
    soup = BeautifulSoup(page.content, 'html.parser')
    course = soup.find('div', id='content-body')
    course_data = course.get_text()
    course_data = re.sub(r'\n\n', '\n', course_data)


    # Save data to a file
    with open('data/program.txt', 'w') as f:
        f.write(degree_data)

    with open('data/courses.txt', 'w') as f:
        f.write(course_data)

##############
#  Chunking  #
##############
def chunk_text(input_file, output_file):
    # load file
    with open(input_file, 'r') as file:
        data = file.read().replace('\n', ' ')

    # split into words
    words = data.split(' ')

    # chunk into 100 word chunks with 25 word overlap
    obj = {
        "chunks": []
    }

    for i in range(0, len(words), 300):
        str_chunk = ""
        for i in words[i:i+450]:
            str_chunk += i + " "
        obj["chunks"].append(str_chunk)

    print(f'Generated {len(obj["chunks"])} chunks from a total of {len(words)} words')

    # save chunks to file
    with open(output_file, 'w') as file:
        json.dump(obj, file)

################################
# Save embeddings in database  #
################################
def save_embeddings(chunks_file):    
    from sentence_transformers import SentenceTransformer

    db = sqlite3.connect('data/db.sqlite')
    db.execute("CREATE TABLE IF NOT EXISTS embeddings (id INTEGER PRIMARY KEY, embedding TEXT, original TEXT)")

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # load chunks json object from file
    with open(chunks_file, 'r') as file:
        obj = json.load(file)

    print(f"Loaded {len(obj['chunks'])} chunks")

    # generate embeddings and store in db

    for c in obj['chunks']:
        embedding = model.encode(c)
        print(embedding)
        # add row to sqlite db with embedding and original text
        db.execute("INSERT INTO embeddings (embedding, original) VALUES (?, ?)", (json.dumps(embedding.tolist()), c))
        db.commit()

    db.close()

if __name__ == "__main__":
    program_data_output = "data/program_chunks.json"
    course_data_output = "data/courses_chunks.json"

    make_data()

    chunk_text(program_data_input, program_data_output)
    chunk_text(course_data_input, course_data_output)

    save_embeddings(program_data_output)
    save_embeddings(course_data_output)