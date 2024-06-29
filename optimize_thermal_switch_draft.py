import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

## Time properties
T_END = 100  # seconds
DT = 0.01  # seconds

## Value properties
RTOL = 1e-3

TEMP_MAX = 120  # degrees
PWR_MAX = 500  # Watts

a = np.array([[0, 1], [1, 1]], dtype=float)
b = np.array([[0], [1]], dtype=float)


t = np.linspace(0, T_END, int(T_END / DT))
u = np.zeros(len(t))
y = np.zeros(len(t))
tau = 20
k = 100
t1 = 60
delay = 6  # seconds
t0 = 25  # degrees

fig, ax = plt.subplots()
ax.set_xlim(0, T_END)
ax.set_ylim(0, TEMP_MAX)
plt.subplots_adjust(left=0.1, bottom=0.26, top=0.95, right=0.95)
endline = plt.axvline(color="r", linestyle="--", label="Sim stop")
setline = plt.axhline(color="b", linestyle="--", label="Set point")
curves = plt.plot(t, y, t, u, lw=1)
text = plt.text(0, 0, "", fontsize=10, color="orange")
delayBuf = []


def ss_step(x, u, dt):
    global delayBuf, delay
    delayBuf.append(u)
    if len(delayBuf) > (delay / DT):
        u = delayBuf.pop(0)
        return x + (a @ x + b * u) * dt
    return x + (a @ x + b * 0) * dt


def binary_search(f, left: float, right: float, tolerance: float):
    while left < right:
        midpoint = (left + right) / 2.0
        f_mid = f(midpoint)
        if abs(f_mid) < tolerance:
            return midpoint
        elif f_mid > 0:
            right = midpoint - tolerance / 2
        else:
            left = midpoint + tolerance / 2
    return right


def detectStopTime(t_sw, data, prev_dy, xmax, ymax):
    """
    Find maximum value stationary point of system response
    """
    if len(data) > 1:
        dy = data[-1] - data[-2]
        if abs(data[-1]) >= abs(ymax):
            ymax = data[-1]
            xmax = max(len(data) - 1, t_sw)
        if (np.sign(dy) - np.sign(prev_dy)) == -2:
            return True, prev_dy, xmax, ymax
        prev_dy = dy
    return False, prev_dy, xmax, ymax


def simulate(tSwitch: float):
    global delayBuf, t0, t1
    endX, endY, diff, state, i_sw, y, stop = (
        0,
        0,
        0,
        np.array([0.0, 0.0]),
        int(round(tSwitch / DT)),
        [],
        False,
    )

    delayBuf.clear()
    if t1 <= 0:
        y = [0.0]
        u[:] = 0
    else:
        u[:i_sw] = 1
        u[i_sw:] = 0

        for i in range(len(t)):
            state = ss_step(state, u[i], DT)
            y.append(state[0][1])
            if not stop:
                stop, diff, endX, endY = detectStopTime(i_sw, y, diff, endX, endY)
            else:
                break
    return y, (endX + 2) * DT


def update(val=0):
    global delay, t0, t1
    # Take settings from user controls
    a[1][0] = -1 / (tauSlider.val * tauSlider.val)
    a[1][1] = -(tauSlider.val + tauSlider.val) / (tauSlider.val * tauSlider.val)
    b[1] = kSlider.val / (tauSlider.val * tauSlider.val)
    t0 = t0Slider.val
    t1 = t1Slider.val - t0
    delay = delaySlider.val

    if t1 > 0:
        # Search for optimal value using binary search algorithm
        t_switch = binary_search(lambda x: simulate(x)[0][-1] - t1, 0, T_END, RTOL)
    else:
        t_switch = 0.0

    # Simulate found optimal solution curve and redraw all
    y, imax = simulate(t_switch)
    endline.set_xdata([imax])
    setline.set_ydata([t1 + t0])
    curves[0].set_ydata(y + np.ones(len(y)) * t0)
    curves[0].set_xdata(t[: len(y)])
    curves[1].set_ydata(np.multiply(u, 100))
    curves[1].set_xdata(t)
    text.set_position([t_switch, u[0]])
    text.set_text(f"tSW={t_switch:.2f} s")
    fig.canvas.draw_idle()


delaySlPos = plt.axes([0.1, 0.16, 0.82, 0.05], facecolor="lightgoldenrodyellow")
tauSlPos = plt.axes([0.1, 0.12, 0.82, 0.05], facecolor="lightgoldenrodyellow")
t1SlPos = plt.axes([0.1, 0.08, 0.82, 0.05], facecolor="lightgoldenrodyellow")
t0SlPos = plt.axes([0.1, 0.04, 0.82, 0.05], facecolor="lightgoldenrodyellow")
kSlPos = plt.axes([0.1, 0, 0.82, 0.05], facecolor="lightgoldenrodyellow")

delaySlider = Slider(ax=delaySlPos, label="τd", valmin=DT * 2, valmax=40, valinit=delay)
tauSlider = Slider(ax=tauSlPos, label="τ1", valmin=DT * 2, valmax=40, valinit=tau)
t1Slider = Slider(ax=t1SlPos, label="T1", valmin=0, valmax=TEMP_MAX, valinit=t1)
t0Slider = Slider(ax=t0SlPos, label="T0", valmin=0, valmax=60, valinit=t0)
kSlider = Slider(ax=kSlPos, label="K", valmin=DT, valmax=PWR_MAX, valinit=k)

delaySlider.on_changed(update)
tauSlider.on_changed(update)
t1Slider.on_changed(update)
t0Slider.on_changed(update)
kSlider.on_changed(update)

update()
plt.show()
