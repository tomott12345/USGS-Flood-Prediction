# Example Flood Prediction at a USGS streamgage node

Flood prediction is a very important task for emergency response officials, engineers, and hydrologists. Timely forecasts are of the utmost importance in protecting life and property. Several [simulations](https://www.nssl.noaa.gov/projects/flash/#:~:text=The%20Flooded%20Locations%20And%20Simulated,saving%20lives%20and%20protecting%20infrastructure.) already exist that can forecast several hours in advance for a particular streamgage or node in a watershed.

Why create another flood prediction process? The simple answer is that machine learning forecasting models can trained and deployed at the edge (on the streamgage) that will allow researchers to remotely turn on additional monitoring tools if the machine learning model inferences a large rise in streamgage height will occur over a predefined time window.

The following repository uses three different open source packages to forecast streamgage height for the Schuylkill River at Conshohocken, PA (streamgage #01473730). We brute force H2O-3 into a time series forecasting use case and use out of box time series forecasting packages of NeuralProphet and AutoGluon. 

The inspiration for these ML models comes from the paper [Data-Driven Flood Alert System (FAS) Using Extreme Gradient Boosting (XGBoost) to Forecast Flood Stages](https://www.researchgate.net/publication/358910939_Data-Driven_Flood_Alert_System_FAS_Using_Extreme_Gradient_Boosting_XGBoost_to_Forecast_Flood_Stages). 
