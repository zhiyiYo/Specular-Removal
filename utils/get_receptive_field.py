def receptive_field(kernel_sizes, strides):
    RF = 1
    for j in range(len(kernel_sizes), 0, -1):
        RF = (RF-1)*strides[j-1] + kernel_sizes[j-1]
    return RF


kernel_sizes = [7]*3
strides = [1]*3
print(receptive_field(kernel_sizes, strides))
