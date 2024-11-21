# Data Driven Flood Prediction at a USGS streamgage nodes

Flood prediction is a very important task for emergency response officials, engineers, research scientists, and hydrologists. Timely forecasts are of the utmost importance in protecting life and property. Several [simulations](https://www.nssl.noaa.gov/projects/flash/#:~:text=The%20Flooded%20Locations%20And%20Simulated,saving%20lives%20and%20protecting%20infrastructure.) exist that can [forecast several hours in advance](https://water.noaa.gov/about/nwm) for a particular streamgage or node in a watershed. These simulations are a physics driven model approach as opposed to a data driven model, which we analyze here.

The purpose for a data driven approach is to evaluate whether or not flood prediction models can trained and deployed at the edge (on the streamgage) that will allow researchers to remotely turn on additional monitoring tools if the machine learning model inferences a large rise in streamgage height will occur over a predefined time window.

There are two subset folders in this repository. All of them use a combination of open source algorithms ranging from NeuralProphet, XGBoost, and Autogluon. The subset folders are broken out into two locations in the NJ and PA area: The Pompton River crossing in Pompton Plans NJ (site #01388500) and the Schuylkill River at Conshohocken, PA (site #01473730). Two additional sites exist for development purposes. 

The inspiration for this analysis comes from the paper [Data-Driven Flood Alert System (FAS) Using Extreme Gradient Boosting (XGBoost) to Forecast Flood Stages](https://www.researchgate.net/publication/358910939_Data-Driven_Flood_Alert_System_FAS_Using_Extreme_Gradient_Boosting_XGBoost_to_Forecast_Flood_Stages). 

