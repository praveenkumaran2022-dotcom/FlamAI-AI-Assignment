# FlamAI AI / R&D Assignment — Parametric Curve Recovery

## Result

Recovered unknown parameters from the supplied `xy_data.csv`:

- **theta (θ) ≈ 30.0000°**
- **M ≈ 0.030000**
- **X ≈ 55.0000**

The assignment constrains the parameters to:

- 0° < θ < 50°
- −0.05 < M < 0.05
- 0 < X < 100
- 6 < t < 60

## Model

\[
x=t\cos(\theta)-e^{M|t|}\sin(0.3t)\sin(\theta)+X
\]

\[
y=42+t\sin(\theta)+e^{M|t|}\sin(0.3t)\cos(\theta)
\]

## Key idea

The data can be transformed into a coordinate system aligned with the unknown angle.

Let:

\[
u=x-X,\qquad v=y-42
\]

Project onto the direction of the straight component:

\[
t=u\cos(\theta)+v\sin(\theta)
\]

Project onto the perpendicular direction:

\[
q=-u\sin(\theta)+v\cos(\theta)
\]

The sinusoidal/exponential component is perpendicular to the main direction, so the model becomes:

\[
q=e^{M|t|}\sin(0.3t)
\]

This reduces the fitting problem to only three unknowns: θ, M and X. `scipy.optimize.least_squares` is then used with the assignment's parameter bounds.

## Run

```bash
py -m pip install -r requirements.txt
py main.py
```

## Desmos equation

Use radians in Desmos:

```text
(t*cos(30*pi/180)-e^(0.03*abs(t))*sin(0.3*t)*sin(30*pi/180)+55,
42+t*sin(30*pi/180)+e^(0.03*abs(t))*sin(0.3*t)*cos(30*pi/180))
```

Set the parameter domain to:

```text
6 <= t <= 60
```

After entering the equation, use Desmos' Share/Copy Link option to generate the personal Desmos URL for the submission.

## Files

- `xy_data.csv` — supplied assignment data
- `main.py` — parameter estimation and visualization
- `requirements.txt` — Python dependencies
- `README.md` — explanation and Desmos equation
