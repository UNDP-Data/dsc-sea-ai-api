## Flask App 

The app will be running at http://127.0.0.1:5000/.

Run main.py to start the App.
``` bash
python main.py
```

Find the list of the modules and packages required in requirements.txt.

## Running the App Using Docker

Build the docker image:
```bash
docker build -t <name-of-image> .
```
Run the docker image:
```bash
 docker run -e API_KEY= <YOUR-OPENAI-API-KEY> -p 5000:5000 backend-flask
```

The app will be running at http://127.0.0.1:5000/ in your browser.