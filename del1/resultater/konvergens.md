# Konvergensstudie – utkrager

Generert av `del1_kilder/Project_start_with_TODO/Konvergensstudie.py`.

Utkrager L = 10 m, H = 2 m, t = 0.1 m, E = 2.1e+11 Pa, ν = 0.3, endelast P = 3.0e+06 N (jevnt fordelt på nodene i fri ende), fast innspent venstre ende.
Spenninger er nodalt midlet over elementene som deler noden. Tall i parentes er relativt avvik fra referansen.

### Antall elementer

| Noder | ndof | T3 | Q4 | T6 | Q9 |
|---|---:|---:|---:|---:|---:|
| 9×5 | 90 | 64 | 32 | 16 | 8 |
| 17×9 | 306 | 256 | 128 | 64 | 32 |
| 33×17 | 1122 | 1024 | 512 | 256 | 128 |
| 65×33 | 4290 | 4096 | 2048 | 1024 | 512 |

### Tuppforskyvning v [m] i (x, y) = (L, H/2)

Referanse: Q9 på fineste nett, v = 0.07335 m. Til sammenligning: Euler–Bernoulli 0.07143 m, Timoshenko (k = 5/6) 0.07366 m.

| Noder | ndof | T3 | Q4 | T6 | Q9 |
|---|---:|---:|---:|---:|---:|
| 9×5 | 90 | 0.04614 (37.1 %) | 0.06301 (14.1 %) | 0.07238 (1.3 %) | 0.07282 (0.7 %) |
| 17×9 | 306 | 0.06370 (13.1 %) | 0.07038 (4.0 %) | 0.07308 (0.4 %) | 0.07320 (0.2 %) |
| 33×17 | 1122 | 0.07061 (3.7 %) | 0.07256 (1.1 %) | 0.07327 (0.1 %) | 0.07331 (0.1 %) |
| 65×33 | 4290 | 0.07263 (1.0 %) | 0.07315 (0.3 %) | 0.07333 (0.0 %) | 0.07335 (0.0 %) |

### Bøyespenning σx [Pa] i (x, y) = (L/2, 0)

Referanse: bjelketeori σx = M·c/I = 2.250e+08 Pa.

| Noder | ndof | T3 | Q4 | T6 | Q9 |
|---|---:|---:|---:|---:|---:|
| 9×5 | 90 | 1.164e+08 (48.3 %) | 1.994e+08 (11.4 %) | 2.140e+08 (4.9 %) | 2.253e+08 (0.1 %) |
| 17×9 | 306 | 1.777e+08 (21.0 %) | 2.192e+08 (2.6 %) | 2.222e+08 (1.2 %) | 2.250e+08 (0.0 %) |
| 33×17 | 1122 | 2.066e+08 (8.2 %) | 2.242e+08 (0.4 %) | 2.243e+08 (0.3 %) | 2.250e+08 (0.0 %) |
| 65×33 | 4290 | 2.175e+08 (3.3 %) | 2.251e+08 (0.1 %) | 2.248e+08 (0.1 %) | 2.250e+08 (0.0 %) |

### Skjærspenning τxy [Pa] i (x, y) = (L/2, H/2)

Referanse: bjelketeori τmax = 1,5·P/A = 2.250e+07 Pa.

| Noder | ndof | T3 | Q4 | T6 | Q9 |
|---|---:|---:|---:|---:|---:|
| 9×5 | 90 | 1.719e+07 (23.6 %) | 1.428e+07 (36.5 %) | 2.607e+07 (15.9 %) | 3.563e+07 (58.3 %) |
| 17×9 | 306 | 2.066e+07 (8.2 %) | 2.022e+07 (10.1 %) | 2.336e+07 (3.8 %) | 2.559e+07 (13.7 %) |
| 33×17 | 1122 | 2.199e+07 (2.3 %) | 2.191e+07 (2.6 %) | 2.273e+07 (1.0 %) | 2.327e+07 (3.4 %) |
| 65×33 | 4290 | 2.237e+07 (0.6 %) | 2.235e+07 (0.7 %) | 2.256e+07 (0.3 %) | 2.269e+07 (0.9 %) |

### Bøyespenning σx [Pa] i innspenningshjørnet (x, y) = (0, 0)

Bjelketeori gir 4.500e+08 Pa, men hjørnet er et singulært punkt (fast innspenning i både x og y). Verdien vokser når nettet forfines og konvergerer ikke.

| Noder | ndof | T3 | Q4 | T6 | Q9 |
|---|---:|---:|---:|---:|---:|
| 9×5 | 90 | 2.034e+08 | 3.942e+08 | 4.287e+08 | 4.780e+08 |
| 17×9 | 306 | 3.419e+08 | 4.639e+08 | 4.581e+08 | 5.047e+08 |
| 33×17 | 1122 | 4.256e+08 | 5.094e+08 | 4.882e+08 | 5.473e+08 |
| 65×33 | 4290 | 4.846e+08 | 5.618e+08 | 5.354e+08 | 6.152e+08 |
