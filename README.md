python==3.11.5  
pandas==2.2.1  
scikit_learn==1.3.2  
torch==2.1.0+cu118  
tqdm==4.66.1  
ucimlrepo==0.0.3  
fastapi==0.110.1  

Install dependencies
```
pip install -r requirements.txt
```

You may also want to install uvicorn to run the web server:
```
pip install uvicorn[standard]
```

Then, create the model using
```
python create_model.py
```

If you are using uvicorn, run the web server using the following:
```
uvicorn main:app --reload
```

To make an API call, make a `POST` request to `/predict` with a body like the following:
```
{
  "features": [20.44,21.78,133.8,1293,0.0915,0.1131,0.09799,0.07785,0.1618,0.05557,0.5781,0.9168,4.218,72.44,0.006208,0.01906,0.02375,0.01461,0.01445,0.001906,24.31,26.37,161.2,1780,0.1327,0.2376,0.2702,0.1765,0.2609,0.06735]
}
```
which should return the diagnosis
```
{
  "diagnosis": "M"
}
```