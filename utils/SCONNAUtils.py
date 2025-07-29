from curses import REPORT_MOUSE_POSITION
import torch
import math
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
StoBLookUp = pd.read_csv("resources/8_bit_MUL.csv")
print(device)


# https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
def getDecimalToBinary(input, bitwidth):
    """This takes a tensor with decimal values and returns their binary representation

    Args:
        input (_type_): tensor
        bitwidth (_type_): The number of bits

    Returns:
        _type_: _description_
    """
    mask = 2 ** torch.arange(bitwidth - 1, -1, -1)
    return input.unsqueeze(-1).int().bitwise_and(mask).ne(0).byte()


def getBinaryToDecimal(input, bitwidth):
    """This method accepts a tensor containing binary values and returns the decimal of the binary

    Args:
        input (_type_): _description_
        bitwidth (_type_): _description_

    Returns:
        _type_: _description_
    """
    mask = 2 ** torch.arange(bitwidth - 1, -1, -1).to(input.device, input.dtype)
    return torch.sum(mask * input, -1)


def getDecimalToUnary(input, bitwidth, encode="TC"):
    """This method accepts an input in decimal format and generates an unary transisition encoded representation of the input.
        The length of the unary bit stream depends on the bitwidth
        For negative values it generate -1 so that during convolution operation it can work with negative weights
    Args:
        input (int): The input decimal that needs to be transistion encoded
        bitwidth (int): The length of the stochastic unary bit stream
        encode (str, optional): _description_. Defaults to 'TC'.  TC represents transistion encoding
    Update:
    1) Method now looks for the stochastic bitstream from the lookup table available in resources rather than generating

    """
    # unary_bs = StoBLookUp[StoBLookUp["Decimal_OP1"] == abs(input.item())].iloc[0][
    #     "SC_OP1"
    # ]
    # unary_bs = unary_bs.replace("]", "").replace("[", "").replace("\n", "").split(" ")
    # if input.item() < 0:
    #     unary_bs = [int(i) * -1 for i in unary_bs]
    # else:
    #     unary_bs = [int(i) for i in unary_bs]
    # return unary_bs
    if encode == "TC":
        if input > 0:
            unary_code = torch.cat(
                (torch.zeros(2**bitwidth - input), torch.ones(input)), 0
            ).int()
        else:
            unary_code = torch.cat(
                (
                    torch.zeros(2**bitwidth - abs(input)),
                    torch.ones(abs(input)) * (-1),
                ),
                0,
            ).int()
        # print("Unary Code ", unary_code)
        return unary_code
    else:
        print("Other encoding schemes are not implemented")


def getDecimalToUnaryADD(input, bitwidth, encode="TC"):
    """This method accepts an input in decimal format and generates an unary transisition encoded representation of the input.
        The representation is to used for the second operand of addition. To maintain negative correlation with the operand 1
        the outputs is reverse order of getDecimalToUnary
        The length of the unary bit stream depends on the bitwidth

    Args:
        input (_type_):  The input decimal that needs to be transistion encoded
        bitwidth (_type_): The length of the stochastic unary bit stream
        encode (str, optional): _description_. Defaults to "TC". TC represents transistion encoding
    """
    if encode == "TC":
        unary_code = torch.cat(
            (torch.ones(input), torch.zeros((2 ** (bitwidth + 1)) - input)), 0
        ).int()
        return unary_code
    else:
        print("Other encoding schemes are not implemented")


def twoBitDecoder(bit):
    """

    Args:
        bit (_type_): _description_

    Returns:
        _type_: _description_
    """
    if bit == 1:
        return torch.tensor([1, 0])
    else:
        return torch.tensor([0, 0])


def rightShiftRegister(input, enable, shifts=1):
    """This takes in an tensor with shift registers and their corresponding enable signals. Depending on the enable signal of the
    shift register, the values of register are right shifted and consequently msb is replaced by 1.

    Args:
        input (tensor): ex [[1,0],[1,0],[1,0]]
        enable (_type_): ex [1,0,0]
        shifts (int, optional): _description_. Defaults to 1.

    Returns:
        (tensor) : ex [[1,1],[1,0],[1,0]]
    """
    enable_idx = 0
    while enable_idx < len(enable):
        if enable[enable_idx] == 1:
            shifted_input = torch.roll(input[enable_idx], shifts, 0)
            # updating msb of the shifted register
            shifted_input[0] = 1
            input[enable_idx] = shifted_input
        enable_idx = 1 + enable_idx
    return input


# ! A new and better verison is getDecimalToUnaryMulV2
def getDecimalToUnaryMulV1(input, bitwidth, encode="TC"):
    """The input is generally in decimel
    1. Convert the decimel input into binary
    2. Take the most significant bit of binary value and send it to decoder. The output of decoder 1->10 0->00
    3. The remaining bits are to be converted to decimal, and are sent to the getDecimalToUnary with bitwidth 2**RB where RB (remaining bits) = number of bits after
    removing the two most significant bits
    4. The output of the getDecimalToUnary is URB
    6. The most significant two bits are mapped onto two right shift registers. A total of (2**bitwidth)/2 right shift registers are used.
    Intialially, all the registers are filled with decoded MSB. The enable signals of these shift registers is URB bits expect the most significant bit of URB.
    7. Each shift register is shifted by 1 depending on the enable signal from URB bits.
    8. The bits present in the MSB decoder followed by the values in the shift register give the unary

    Args:
        input (_type_): _description_
        bitwidth (_type_): _description_
        encode (str, optional): _description_. Defaults to 'TC'.

    """
    # print("Input ", input)
    # unary_bs = StoBLookUp[StoBLookUp["Decimal_OP2"] == abs(input.item())].iloc[0][
    #     "SC_OP2"
    # ]
    # unary_bs = unary_bs.replace("]", "").replace("[", "").replace("\n", "").split(" ")

    # unary_bs = [int(i) for i in unary_bs]

    # return unary_bs

    binary_input = getDecimalToBinary(input, bitwidth).reshape(-1)
    msb = binary_input[0]
    decoded_msb = twoBitDecoder(msb)
    rb = binary_input[1:]
    rb_decimal = getBinaryToDecimal(rb, bitwidth - 1)
    rb_unary = getDecimalToUnary(rb_decimal, bitwidth - 1)
    shift_registers = decoded_msb.repeat((len(rb_unary) - 1, 1))
    shifted_registers = rightShiftRegister(shift_registers, rb_unary[1:])
    shifted_registers = shift_registers.reshape(
        -1,
    )
    decoded_msb = decoded_msb.reshape(
        -1,
    )

    unary_code = torch.cat((decoded_msb, shifted_registers))
    # print("Binary Input ", binary_input)
    # print("Decoded MSB ", decoded_msb)
    # print("Shifted Registers ", shifted_registers)
    # print("Unary Code :", unary_code)

    return unary_code


def num2tuple(num):
    return num if isinstance(num, tuple) else (num, num)


def conv2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    h_w, kernel_size, stride, pad, dilation = (
        num2tuple(h_w),
        num2tuple(kernel_size),
        num2tuple(stride),
        num2tuple(pad),
        num2tuple(dilation),
    )
    pad = num2tuple(pad[0]), num2tuple(pad[1])

    h = math.floor(
        (h_w[0] + sum(pad[0]) - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
    )
    w = math.floor(
        (h_w[1] + sum(pad[1]) - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
    )

    return h, w


# ! updated on 11/2/2022: Changes improve the computing accuracy
def getDecimalToUnaryMul(input, bitwidth, encode="TC"):
    """The input is generally in decimel
    1. Convert the decimel input into binary
    2. Take the most significant bit of binary value and send it to decoder. The output of decoder 1->10 0->00
    3. The remaining bits are to be converted to decimal, and are sent to the getDecimalToUnary with bitwidth 2**RB where RB (remaining bits) = number of bits after
    removing the two most significant bits
    4. The output of the getDecimalToUnary is Unary respresentation of RB (URB)
    6. The most significant two bits are mapped onto two right shift registers. A total of (2**bitwidth)/2 right shift registers are used.
    Intialially, all the registers are filled with decoded MSB.
    UPDATE:
    URB is reversed first and enable signals of the shift registers are URB bits expect the least significant bit of URB.
    7. Each shift register is shifted by 1 depending on the enable signal from URB bits.
    8. The bits present in the MSB decoder followed by the values in the shift register give the unary

    Args:
        input (_type_): _description_
        bitwidth (_type_): _description_
        encode (str, optional): _description_. Defaults to 'TC'.

    """
    # print("Input ", input)
    # unary_bs = StoBLookUp[StoBLookUp["Decimal_OP2"] == abs(input.item())].iloc[0][
    #     "SC_OP2"
    # ]
    # unary_bs = unary_bs.replace("]", "").replace("[", "").replace("\n", "").split(" ")

    # unary_bs = [int(i) for i in unary_bs]

    # return unary_bs

    binary_input = getDecimalToBinary(input, bitwidth).reshape(-1)
    msb = binary_input[0]
    decoded_msb = twoBitDecoder(msb)
    rb = binary_input[1:]
    rb_decimal = getBinaryToDecimal(rb, bitwidth - 1)
    rb_unary = getDecimalToUnary(rb_decimal, bitwidth - 1)
    rb_unary = torch.flip(rb_unary, [0])
    shift_registers = decoded_msb.repeat((len(rb_unary) - 1, 1))
    shifted_registers = rightShiftRegister(shift_registers, rb_unary[:-1])
    shifted_registers = shift_registers.reshape(
        -1,
    )
    decoded_msb = decoded_msb.reshape(
        -1,
    )

    unary_code = torch.cat((decoded_msb, shifted_registers))
    # print("Binary Input ", binary_input)
    # print("Decoded MSB ", decoded_msb)
    # print("Shifted Registers ", shifted_registers)
    # print("Unary Code :", unary_code)

    return unary_code


def num2tuple(num):
    return num if isinstance(num, tuple) else (num, num)


def conv2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    h_w, kernel_size, stride, pad, dilation = (
        num2tuple(h_w),
        num2tuple(kernel_size),
        num2tuple(stride),
        num2tuple(pad),
        num2tuple(dilation),
    )
    pad = num2tuple(pad[0]), num2tuple(pad[1])

    h = math.floor(
        (h_w[0] + sum(pad[0]) - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
    )
    w = math.floor(
        (h_w[1] + sum(pad[1]) - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
    )

    return h, w


# # ! Testing the decimal to unary conversion
# operand1 = getDecimalToUnary(torch.tensor([2]) * -1, 2)
# operand2 = getDecimalToUnaryMul(torch.tensor([2]), 2)
# # operand3 = getDecimalToUnaryADD(torch.tensor([5]), 3)

# print("Operand 1 :", operand1)
# print("Operand 2 :", operand2)
# print("Operand 3 :", operand3)
# print(rightShiftRegister(torch.tensor([0, 1])))
