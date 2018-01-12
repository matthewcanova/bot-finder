import math


def binarize(value, bits):
    value = round(value)
    binaries = {}
    if value >= math.pow(2, bits):
        for x in range(bits):
            binaries[x] = 1
        return binaries

    for x in range(bits-1, -1, -1):
        binaries[x] = value // int(math.pow(2, x))
        value = value % int(math.pow(2, x))
    return binaries


# 1. length of screen name (6 bit representation)
def feat_1(original_features, new_features, start):
    bits = 6
    binaries = binarize(original_features[1], bits)
    for key in binaries:
        if binaries[key] == 1:
            new_features[start+key] = binaries[key]
    return start+bits


# 2. length of description (10 bit representation)
def feat_2(original_features, new_features, start):
    bits = 10
    binaries = binarize(original_features[2], bits)
    for key in binaries:
        if binaries[key] == 1:
            new_features[start+key] = binaries[key]
    return start+bits


# 3. longevity: days (15 bit representation)
def feat_3(original_features, new_features, start):
    bits = 15
    binaries = binarize(original_features[3], bits)
    for key in binaries:
        if binaries[key] == 1:
            new_features[start+key] = binaries[key]
    return start+bits

#
# # 4. longevity: hours
# def feat_4():
#     return 0
#
#
# # 5. longevity: minutes
# def feat_5():
#     return 0
#
#
# # 6. longevity: seconds
# def feat_6():
#     return 0


# 7. number of following (20 bit representation)
def feat_7(original_features, new_features, start):
    bits = 20
    binaries = binarize(original_features[7], bits)
    for key in binaries:
        if binaries[key] == 1:
            new_features[start+key] = binaries[key]
    return start+bits


# 8. numberof followers (20 bit representation)
def feat_8(original_features, new_features, start):
    bits = 20
    binaries = binarize(original_features[8], bits)
    for key in binaries:
        if binaries[key] == 1:
            new_features[start+key] = binaries[key]
    return start+bits


# 9. the ratio of the number of following and followers (* 100 and then binarize 15 bits)
def feat_9(original_features, new_features, start):
    bits = 15
    binaries = binarize(original_features[9] * 100, bits)
    for key in binaries:
        if binaries[key] == 1:
            new_features[start+key] = binaries[key]
    return start+bits


# 10. the number of posted tweets (15 bit representation)
def feat_10(original_features, new_features, start):
    bits = 15
    binaries = binarize(original_features[10], bits)
    for key in binaries:
        if binaries[key] == 1:
            new_features[start+key] = binaries[key]
    return start+bits


# 11. the number of posted tweets per day (*10 and then 8 bit representation)
def feat_11(original_features, new_features, start):
    bits = 8
    binaries = binarize(original_features[11] * 10, bits)
    for key in binaries:
        if binaries[key] == 1:
            new_features[start+key] = binaries[key]
    return start+bits


# 12. the average number of links in tweets (*10 and 8 bit representation)
def feat_12(original_features, new_features, start):
    bits = 8
    binaries = binarize(original_features[12] * 10, bits)
    for key in binaries:
        if binaries[key] == 1:
            new_features[start+key] = binaries[key]
    return start+bits


# 13. the average number of unique links in tweets (*10 and 8 bit representation)
def feat_13(original_features, new_features, start):
    bits = 8
    binaries = binarize(original_features[13] * 10, bits)
    for key in binaries:
        if binaries[key] == 1:
            new_features[start+key] = binaries[key]
    return start+bits


# 14. the average numer of username in tweets (*10 and 8 bit representation)
def feat_14(original_features, new_features, start):
    bits = 8
    binaries = binarize(original_features[14] * 10, bits)
    for key in binaries:
        if binaries[key] == 1:
            new_features[start+key] = binaries[key]
    return start+bits


# 15. the average numer of unique username in tweets (*10 and 8 bit representation)
def feat_15(original_features, new_features, start):
    bits = 8
    binaries = binarize(original_features[15] * 10, bits)
    for key in binaries:
        if binaries[key] == 1:
            new_features[start+key] = binaries[key]
    return start+bits


# 16. the change rate of number of following (1 bit for +/- and then *10 and 15 bits)
def feat_16(original_features, new_features, start):
    sign = 0 if original_features[16] > 0 else 1
    bits = 15
    binaries = binarize(abs(original_features[16] * 10), bits)
    for key in binaries:
        if binaries[key] == 1:
            new_features[start+key] = binaries[key]
    new_features[start+15] = sign
    return start+bits
