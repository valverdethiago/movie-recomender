# movie-rec

## Prerequisites
- Python 3.x  
- Node 18.18
- Yarn 1.22

## How to Install and Run
1. Clone the repository:

    ```bash
    git clone https://github.com/valverdethiago/movie-recomender
    ```
### Backend
<details>
<summary>Backend Setup</summary>

2. go to mf-recommender folder
   ```bash
    cd mf-recommender
    ```
3. Download the dataset [from kagle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=movies_metadata.csv)  to folder `mf-recommender`.


4. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    ```

5. Activate the virtual environment:

    ```bash
    source venv/bin/activate
    ```

6. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

7. Make sure the movie and ratings data are available in the `the-movies-dataset` folder.

8. Prepair recommendations. It will generate the file `ratings.duckdb` and `recs.pkl`. It will take less then a minute to run.
    ```bash
    python train.py
    ```

9. Run the application. If the first time, it will take about 3 minutes create duckdb file `movies.db`. Next time, it will just use the existing file.

    ```bash
    python app.py
    ```
   This will start the Flask development server at `http://127.0.0.1:5000/`.
</details>

### Frontend
<details>
<summary>Frontend Setup</summary>
 <details>
 <summary>Install Node and Yarn (in case you don't have)</summary>

 1. Follow this steps:

  https://lendingclub.atlassian.net/wiki/spaces/UP/pages/34900504/How+to+Install+Node+and+Yarn

2. Check if your node version is 18.18 or newer

    ```bash
    node -v
    ```
3. If node version is older than 18.18, update to new one

    ```bash
   nvm install 18.18.0
   nvm use 18
   ```
    
 </details>

1. go to client folder
   ```bash
    cd client
    ```
2. use yarn to Install

    ```bash
    yarn
    ```
3. Build
   ```bash
   npm run build
   ```
4. To run your app
   ```bash
   npm run preview
   ```
5. Access the page
   ```bash
   http://localhost:4173
   ```
</details>

### Just Run
<details>
<summary>To run server and app</summary>

1. go to mf-recommender folder
   ```bash
    cd mf-recommender
    ```
2. start server
    ```bash
    python app.py
    ```
3. go to client folder
    ```bash
    cd ../client
    ```
4. run app
    ```bash
   npm run dev -- --open
    ```
</details>
