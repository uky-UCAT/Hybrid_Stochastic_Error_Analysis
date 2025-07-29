from utils.SCONNAOps import (
    countOneInBS,
    stochasticMUL,
    stochasticADD,
    stochasticSUB,
)
from utils.SCONNAUtils import *
import itertools
from itertools import permutations
import numpy as np
import pandas as pd

""" Summary of this Code
The purpose of this code is to test the stochastic compututation versus conventional computation of operations like addition, substraction and multiplication 
1. It will accept binary bitwidth
2. For the bitwidth find all the combination of operand_1 and operand_2 that are possible. 
For example, for bitwidth = 2, the possible binary values are [0,1,2,3] where max value is 2**(bitwidth)-1. and
the possible (operand_1, operand_2) combinations : [0,0], [0,1],[0,2],[0,3], [1,0], [1,1] etc
3. After generating these combinations, depending on the operation being tested find the output value in binary which are true outputs
For example, if the operation is ADD for a combination such as (2,1) -> result = 3, similarly for all combinations and all operations
4. Convert all the combinations of binary values into stochastic bitstreams depending on the operation
5. Perform all operations in stochastic domain for each stochastic bitstream combinations
6. Find the result of the operation obtained in stochastic domain 


"""


bitwidth = 4
operand1_range = torch.arange(0, 2**(bitwidth-1))
operand2_range = torch.arange(0, 2**bitwidth)
# print(operand1_range)
operation = "MUL"


result_list = []
for operand1 in operand1_range:
    for operand2 in operand2_range:
        result = {}
        # * Decimal output computation
        # operand1 = torch.tensor(155)
        # operand2 = torch.tensor(132)

        output = operand1 * operand2
        
        # * Stochastic Output computation
        operand1_b = getDecimalToBinary(operand1, bitwidth)
        operand2_b = getDecimalToBinary(operand2, bitwidth)
        
        operand1_msb = getBinaryToDecimal(operand1_b[: bitwidth // 2],bitwidth//2)
        operand1_lsb = getBinaryToDecimal(operand1_b[bitwidth // 2 :],bitwidth//2)
        
        operand2_msb = getBinaryToDecimal(operand2_b[: bitwidth // 2],bitwidth//2)
        operand2_lsb = getBinaryToDecimal(operand2_b[bitwidth // 2 :],bitwidth//2)
        
        operand1_msb_sc = getDecimalToUnary(operand1_msb, bitwidth//2)
        operand1_lsb_sc = getDecimalToUnary(operand1_lsb, bitwidth//2)
        
        # operand2_msb_sc = getDecimalToUnaryMul(abs(operand2_msb), bitwidth//2)
        # operand2_lsb_sc = getDecimalToUnaryMul(abs(operand2_lsb), bitwidth//2)
        operand2_msb_sc = getDecimalToUnary(abs(operand2_msb), bitwidth//2)
        operand2_lsb_sc = getDecimalToUnary(abs(operand2_lsb), bitwidth//2)
        
        p1 = stochasticMUL(operand1_msb_sc, operand2_msb_sc)
        p2 = stochasticMUL(operand1_msb_sc, operand2_lsb_sc)
        p3 = stochasticMUL(operand1_lsb_sc, operand2_msb_sc)
        p4 = stochasticMUL(operand1_lsb_sc, operand2_lsb_sc)
        
        # output_sc_bs = stochasticMUL(operand1_sc, operand2_sc)
        
        # output_sc = countOneInBS(p1) * 2**bitwidth + countOneInBS(p2) * 2**(bitwidth // 2) + countOneInBS(p3) * 2**(bitwidth // 2) + countOneInBS(p4)
        # output_sc = countOneInBS(p1) * 2**(bitwidth // 2) + countOneInBS(p2) + countOneInBS(p3)  + countOneInBS(p4) / 2 ** (bitwidth//2)
        output_sc = countOneInBS(p1) * 2**(bitwidth // 2) + countOneInBS(p2) + countOneInBS(p3) 
        
        result["Decimal_OP1"] = operand1.numpy()
        result["Decimal_OP2"] = operand2.numpy()
        result["Decimal_OPT"] = output.numpy()
        operand1_sc = np.concatenate((operand1_msb_sc, operand1_lsb_sc))
        operand2_sc = np.concatenate((operand2_msb_sc, operand2_lsb_sc))
        result["SC_OP1"] = np.array(operand1_sc)
        result["SC_OP2"] = np.array(operand2_sc)
        result["Decimal_OPT_Scaled"] = output.numpy()/2**bitwidth
        # result["SC_OPT_BS"] = output_sc_bs.numpy()
        result["SC_OPT"] = output_sc.numpy()
        result["Error"] = abs(
            ((result["Decimal_OP1"] / 2**bitwidth) * (result["Decimal_OP2"] / 2**bitwidth))
            - ((result["SC_OPT"] / 2**bitwidth))
        )
        result_list.append(result)
        # print("Direct:"+str(result['Decimal_OPT']/2**bitwidth))
        # print("Result:"+str(result['SC_OPT']))
    #     break
    # break
mae = np.mean([abs(r["Error"]) for r in result_list])
print("Mean Absolute Error: ", mae)

max_error = np.max([abs(r["Error"]) for r in result_list])
print("Max Error: ", max_error)

resultDf = pd.DataFrame(result_list)
fileName = str(bitwidth) + "_bit_" + operation + ".csv"
resultDf.to_csv("resources/StochasticMULV2/" + fileName, index=False)
