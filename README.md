# Demo Backend

## Flask App

The app will be running at http://127.0.0.1:5000/.

Run main.py to start the App.

```bash
python main.py
```

Find the list of the modules and packages required in requirements.txt.

## Endpoints

### 1. Semantic Search and Response Generation

- **Endpoint:** `/llm`
- **Method:** POST

  ```json
  {
    "prompt": "user query"
  }
  ```

### 2. PandasAI Data Analysis

- **Endpoint:** `/pandasai`
- **Method:** POST

  ```json
  {
    "table_name": "table name",
    "prompt": "user prompt"
  }
  ```

### 3. Get DataFrame Header

- **Endpoint:** `/header`
- **Method:** GET

  ```json
  {
    "table_name": "table name"
  }
  ```

## Testing with Jupyter Notebook

Testing cases can be found in 'test.ipynb'

## Running the App Using Docker

Build the docker image:

```bash
docker build -t <name-of-image> .
```

Run the docker image:

```bash
 docker run -e API_KEY= <YOUR-OPENAI-API-KEY> -p 5000:5000 <name-of-image>
```

The app will be running at http://127.0.0.1:5000/ in your browser.

## Running app using Virtual enviroment

It’s advisable to create a virtual environments (optional but recommended)

E.g
python3.9 -m venv myenv
source myenv/bin/activate

Then install dependencies

pip install -r requirements.txt

Run this once to install spacy english data on the environment

python3.9 -m spacy download en

Finally run the file :

python3.9 main.py
