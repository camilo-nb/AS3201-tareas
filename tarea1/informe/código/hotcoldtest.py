def dBm2W(P):
    """Converts an arbitrary power `P` in dBm to W."""
    return 10 ** ((P - 3) / 10)

def T_rec(T_hot, T_cold, y_factor):
    """Calculates receiver noise temperature via hot & cold temperatures and y factor."""
    return (T_hot - y_factor * T_cold) / (y_factor - 1)

def y_factor(w_hot, w_cold):
    """Y factor via hot & cold power in W."""
    return w_hot / w_cold

T_hot = 300 # K
T_cold = 77 # K

P_hot = -44.5 # dBm
P_cold = -47.94 # dBm

print("T_rec = {} K".format(T_rec(T_hot, T_cold, y_factor(dBm2W(P_hot), dBm2W(P_cold)))))
