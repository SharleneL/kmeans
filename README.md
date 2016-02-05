# K-means for Text - By Shalin Luo
1. **kmeans.py**
    - A Python script implementing K-means clustering for indexed documents
    - Support 2 centroid selection methods: **random** & **K-means++**
    - Support 2 vector term weight calculation methods: **tf** & **tf-idf**
    - To run the script from command line: `python kmeans.py [cluster_num] [-general/-customize] [-random/-kpp] [file_path] > [output-file-path]`
2. **analytics.py**
    - A Python script to do corpus analytics, including document count, word count, unique word count, etc.