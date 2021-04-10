import torch


# train = {"s1": 1, "s2": 2}
# valid = {"s1": 1, "s2": 2}
# test = {"s1": 1, "s2": 2}

# def run(input):
#     return input + 1

# x = eval('run')(10)
# print(x)

# i = 0
# for split in ['s1', 's2']:
#     for data_type in ['train', 'valid', 'test']:
#         i += 1
#         # 对data_type【变量名]取split值
#         eval(data_type)[split] = i
        

# print(train)
# print(valid)
# print(test)

weight = torch.FloatTensor(3).fill_(1)
print(weight)