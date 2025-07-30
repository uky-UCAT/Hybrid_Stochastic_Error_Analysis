This is the library for analyzing errors in different parts of the Stochastic-Analog-Binary conversion step of hybrid split stochastic representation used in Pro-STAR. 

Files:
1. ErrorSplitMultiply - Calculates the error in multiplication of two stochastic numbers after splitting.
2. StoA_error - Calculates the error in Stochastic to Analog conversion (unused)
3. AtoB_error - Calculates the whole error in Stochastic to Analog and then, binary conversion.
4. error_result - Displays the resulting MAE and Max error of the above two steps without needing to run the entire code again.

Usage:

1. Run ErrorSplitMultiply.py to get the multiplication error. This takes time and is already done and result is stored under resources.
2. Run AtoB_error.py to calculate error is entire Stochastic to Binary conversion. It uses SPICE results which needs to be named "processed.csv". Data is given.
3. Just to see results, run error_result.py. Result of StoA_error.py can be ignored as both StoA and AtoB is combined during SPICE analysis in AtoB_error.
