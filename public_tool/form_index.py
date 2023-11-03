def form_index(lengths, index):
    # input:
    #   lengths
    #   index: the number of lengths
    # output:
    #   begin_index
    #   end_index

    begin_index = sum(lengths[0:index])
    end_index = begin_index + lengths[index]

    return begin_index, end_index
