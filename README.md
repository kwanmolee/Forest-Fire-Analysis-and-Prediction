# Forest-Fire-Analysis-and-Prediction


It is part of the full project, https://github.com/JsnDg/ECE143-Project. 
# ECE143-Project:Prediction of Forest Fires Based on Meteorological Analysis
Authors: Group 11:Zijian Ding, Gitika Karumuri, Kunmao Li, Jianing Zhang

Project Description:
We have analyzed a dataset collected from forests in the northeast region of Portugal with the utilization of FWI system. Firstly, we established relationships between the meteorological factors(e.g. temperatrue, relative humidity, wind) as well as model features(e.g. FFMC, DMC) correlated to forest fires. Having acquired the weights and influence of different factors on forest fires, We then applied the relationship to building a prediction system for forest fires. The system will calculate the parametrs input and give feedback on the probability of potential forest fires using SVM.

Modules used:
   Please install all of the modules mentioned below before running the files.<br /> 
  Data Processing:<br />
  -Pandas<br />
  -numpy<br />
  Data Visualization:<br />
  -matplotlib.pyplot<br />
  -seaborn<br />
  -mpl_toolkits.mplot3d<br />
  -pyehcart<br />
  Regression Models:<br />
  -sklearn:datasets, linear_model,sklearn.metrics<br />
  -scipy.interpolate

Dataset link:http://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/ rates fire hazards based on both natural factors, such as humidity, as well as artificial factors, such as litter. All data is collected from Montesinho park, located in Portugal, with more than 570 instances through 2001 to 2003. 


File Overview: Each folder contains data analysis sorted by their folder names, every picture shown in presentation(group11.pptx) can be found in final_presentation.ipynb.


Reference
  1.[Cortez and Morais, 2007] P. Cortez and A. Morais. A Data Mining Approach to Predict Forest Fires using Meteorological Data. In J. Neves, M. F. Santos and J. Machado Eds., New Trends in Artificial Intelligence, Proceedings of the 13th EPIA 2007 - Portuguese Conference on Artificial Intelligence, December, Guimarães, Portugal, pp. 512-523, 2007. APPIA, ISBN-13 978-989-95618-0-9. Available at: www.dsi.uminho.pt/~pcortez/fires.pdf <br />    
  2.[UCI Machine Learning Repository] Forest Fires Data Set. Available at: http://archive.ics.uci.edu/ml/datasets/Forest+Fires



