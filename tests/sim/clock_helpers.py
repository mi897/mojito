"""Helpers shared by the servo and sensor tests (not a test module: unittest discover only collects test*.py)."""


def fast_forward(model, n_steps, dt):
    """Put a fresh servo or sensor at step `n_steps` without running it (running there takes tens of seconds).
    Sets the step counter it uses, and also `_time`, the accumulated float clock an earlier version compared against
    send time + latency, to the value that version would have reached, so these tests fail on that version."""
    accumulated_time = 0.0
    for _ in range(n_steps):
        accumulated_time += dt
    model._step_index, model._time = n_steps, accumulated_time
