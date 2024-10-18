# Example Flood Prediction at a USGS streamgage node

Flood prediction is a very important task for emergency response officials, engineers, and hydrologists. Several simulations already exist that can forecast several hours in advance for a particular streamgage or node in a watershed.

Why create another flood prediction process? The simple answer is that machine learning forecasting models can trained and deployed at the edge (on the streamgage) that will allow researchers to remotely turn on additional monitoring tools if the machine learning model inferences a large rise in streamgage height will occur over a predefined time window.

The following repository uses three different open source packages to forecast streamgage height for the Schuylkill River at Conshohocken, PA (streamgage #01473730). We brute force H2O-3 into a time series forecasting use case and use out of box time series forecasting packages of NeuralProphet and AutoGluon. 
