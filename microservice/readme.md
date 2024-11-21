# Working Microservice

This code is a simple FastAPI microservice that can be used to serve the autogluon models trained on the various streamgage nodes. You will need to install all the Python libraries (see requirements.txt) and place all the autogluon models in a ../models folder. A sample model has been trained and saved to the ../models folder for your testing. 

Note: the microservice does some pretty intense data munging to get the data shaped into the series format that Autogluon likes. Inspect the sample streamgage notebooks to understand how the data is shaped. 

To run the mircorservice simply pass the following command via the command line `uvicorn app:app` and then point your browser to http://localhost:8000/predict/{site_code}/{forecast_length}, where the example model site_code = 01388500, and forecast_length = 1 ; http://localhost:8000/predict/01388500/1

You can also use curl as well such as `curl -X GET "http://localhost:8000/predict/01388500/1"`

Currently no other model types are supported. 