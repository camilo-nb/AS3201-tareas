def dBm2W(W):
    """Converts an arbitrary power `W` in dBm to W."""
    return 10 ** ((W - 3) / 10)

def T_rec(T_hot, T_cold, y_factor):
    """Calculates receiver noise temperature via hot & cold temperatures and y factor."""
    return (T_hot - y_factor * T_cold) / (y_factor - 1)

def Y_factor(P_hot, P_cold):
    """Y factor via hot & cold power in W."""
    return P_hot / P_cold

T_hot = 300 # K
T_cold = 77 # K

W_hot = -44.5 # dBm
W_cold = -47.94 # dBm

Y = Y_factor(dBm2W(W_hot), dBm2W(W_cold))

print("Y = {}".format(Y))
print("T_rec = {} K".format(T_rec(T_hot, T_cold, Y)))
