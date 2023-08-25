**04_cricket_1999to2011.csv**: This file contains data on ODI matches from 1999 to 2011. It is taken from this site-
http://eamonmcginn.com.s3-website-ap-southeast-2.amazonaws.com/

**Problem statement**:  Using the first innings data alone in the above data set, find the best fit 'run production functions' in terms of wickets-in-hand w and overs-to-go u. Assume the model Z(u,w) = Z0(w)[1 - exp{-Lu/Z0(w)}]. Use the sum of squared errors loss function, summed across overs, wickets, and data points for those overs and wickets. Provide a plot of the ten functions, and report the (11) parameters associated with the (10) production functions, and the normalised squared error.

**Alice_21275_Report1.pdf**: Contains the report of how data processing was done and final report after applying duckworth-lewis method to above dataset. This method could be applied to any cricket dataset and even you can win at different platform like dream11 to predict how the match can go after or before any interrruption like rain.

**main.py**: Contains possible code for the above problem
**plot.png**: Output

Note- regression forces all slopes to be equal at u = 0

