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


bitwidth = 8

# for sign bit included in 8 bits
operand1_range = torch.arange(0, 2**(bitwidth-1))
operand2_range = torch.arange(0, 2**(bitwidth-1))

# for separate sign bit
# operand1_range = torch.arange(0, 2**(bitwidth))
# operand2_range = torch.arange(0, 2**(bitwidth))
operation = "MUL"


result_list = []
for operand1 in operand1_range:
    for operand2 in operand2_range:
        result = {}
        # * Decimal output computation

        output = operand1 * operand2
        
        # * Stochastic Output computation
        
        #convert the operands to binary
        operand1_b = getDecimalToBinary(operand1, bitwidth)
        operand2_b = getDecimalToBinary(operand2, bitwidth)
        
        #split binary into MSB and LSB
        operand1_msb = getBinaryToDecimal(operand1_b[: bitwidth // 2],bitwidth//2)
        operand1_lsb = getBinaryToDecimal(operand1_b[bitwidth // 2 :],bitwidth//2)

        operand2_msb = getBinaryToDecimal(operand2_b[: bitwidth // 2],bitwidth//2)
        operand2_lsb = getBinaryToDecimal(operand2_b[bitwidth // 2 :],bitwidth//2)
        
        # convert operand1's MSB and LSB parts to TCU (zero padded here in software but is actually padded after conversion and result is same)
        operand1_msb_sc = getDecimalToUnary(operand1_msb, bitwidth//2)
        operand1_lsb_sc = getDecimalToUnary(operand1_lsb, bitwidth//2)
        
        # pass operand2's MSB and LSB through the bit shifting network after TCU conversion
        operand2_msb_sc = getDecimalToUnaryMul(abs(operand2_msb), bitwidth//2)
        operand2_lsb_sc = getDecimalToUnaryMul(abs(operand2_lsb), bitwidth//2)
    
        # 3 bits of the MSBs to get 8 bits SC
        operand1_msb_sc_1 = getDecimalToUnary(operand1_msb, bitwidth//2-1) # this is same as taking last 8 bits from operand1_msb_sc
        operand2_msb_sc_1 = getDecimalToUnaryMul(abs(operand2_msb), bitwidth//2-1)
        
        # M1.M2 is 8 bits multiplication
        p1 = stochasticMUL(operand1_msb_sc_1, operand2_msb_sc_1)
        
        # M1.L2 is 16 bits multiplication where M1 is padded with 8 zeros in the front (in TCU all bits are in the rightmost so it doesnt affect)
        p2 = stochasticMUL(operand1_msb_sc, operand2_lsb_sc)
        
        # L1.M2 is 16 bits where M2 is zero padded from 3 bits to 4 bits before SC conversion
        p3 = stochasticMUL(operand1_lsb_sc, operand2_msb_sc)
        
        # L1.L2 is 16 bits
        p4 = stochasticMUL(operand1_lsb_sc, operand2_lsb_sc)
        
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
        result["SC_OPT"] = output_sc.numpy()
        result["Error"] = abs(
            ((result["Decimal_OP1"] / 2**bitwidth) * (result["Decimal_OP2"] / 2**bitwidth))
            - ((result["SC_OPT"] / 2**bitwidth))
        )
        result_list.append(result)

mae = np.mean([abs(r["Error"]) for r in result_list])
print("Mean Absolute Error: ", mae)

max_error = np.max([abs(r["Error"]) for r in result_list])
print("Max Error: ", max_error)

resultDf = pd.DataFrame(result_list)
fileName = str(bitwidth) + "_bit_" + operation + ".csv"
resultDf.to_csv("resources/StochasticMULV2/" + fileName, index=False)
