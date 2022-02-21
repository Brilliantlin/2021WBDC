def getset(a,b):
    '''input series'''
    a = set(a)
    b = set(b)
    com = a & b
    a_b = a - b
    b_a = b - a
    print('公有个数： %s \n a独有个数:%s \n b独有个数: %s ' % (len(com),len(a_b),len(b_a)))