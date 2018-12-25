import math
def beta_function(A, S, delta, k, Vmax, c):
    # print (utils.fast_cts.__name__)
    # print("This is temp {}".format(temp))
    # print("This is k")
    # print(k)
    if k < 1:
        k = 1
    # print("c is : {}".format(c))
    # print("k is : {}".format(k))
    # print("S is : {}".format(S))
    # print("A is : {}".format(A))
    # print("delta is : {}".format(delta))
    # # print("c*(k-1)*(k-1)*S*A is : {}".format(c*(k-1)*(k-1)*S*A))
    # print("c*(k1)*(k1)*S*A/delta is : {}".format(c*(k)*(k)*S*A/delta))
    # print("math.log(c*(k-1)*(k-1)*S*A/delta is : {}".format(math.log(c*(k)*(k)*S*A/delta)))
    # #k = math.maximum(k,1)
    # z = 5
    # assert(math.isnan(5))
    assert (not math.isnan(math.log(c * k * k * S * A / delta))), "log of left is nan"
    left = math.sqrt(k * math.log(c * k * k * S * A / delta))
    assert (not math.isnan(left)), " left side of beta is Nan"

    if k == 1:
        right = 0;
    else:
        right = math.sqrt((k - 1) * math.log(c * (k - 1) * (k - 1) * S * A / delta))  # the error is here
    assert (not math.isnan(right)), " right side of beta is Nan"

    beta = k * Vmax * (left - (1 - 1 / k) * right)
    assert (not math.isnan(beta)), " right side of beta is Nan"
    return beta

def main():
    delta = 0.01
    S = 100000000
    Vmax = 100000
    c = 5
    A = 6
    k = 10
    beta = beta_function(A, S, delta, k, Vmax, c)

    while(abs(beta - 1) > 0.5):
        k = k*2
        beta = beta_function(A, S, delta, k, Vmax, c)
        print("beta: {}, k: {}".format(beta,k))



if __name__ == "__main__":
    main()