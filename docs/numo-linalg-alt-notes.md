# numo-linalg-altで見つけたこと

計測と関門の途中で見つけた、numo-linalg-alt 0.10.1側の不具合と、性能の差の原因を記録する。cumoの問題は[cumo-issues.md](cumo-issues.md)にある。どれも、このリポジトリでは直していない。

## 0. `expm`が、1-ノルムの大きい行列で精度を失う（答えが誤る）

**このリストで一番重い。cumoも同じアルゴリズムを移しているので、Cumoも同じように誤る（[cumo-issues.md](cumo-issues.md)の2）。**

- `expm(a, ord = 8)`はPadé(8)に固定で、scalingの回数を要素の絶対値の最大max|aᵢⱼ| から決める（`n_sqr = log2(max|aᵢⱼ|).to_i + 1`）。Padé(8)の精度が保てるかどうかを決めるのは行列のノルム（1-ノルムなど）なので、要素は小さくても1-ノルムが大きい行列では、scalingが足りないまま近似する。
- 再現（fp64）。要素が[−1, 1]の対称なn × n行列Eについて、`expm(E)`と、`eigh`から作るV diag(e^λ) Vᵀ の相対差を見た。

  | n | 1-ノルム | scipy | Numo |
  |---|---|---|---|
  | 16 | 6.2 | 2.4e-14 | 2.0e-15 |
  | 64 | 26 | 5.0e-15 | 5.8e-11 |
  | 256 | 97 | 5.0e-15 | 2.3e-5 |
  | 1024 | 368 | 9.7e-15 | 0.99 |

- **最初はexpm(A)·expm(−A) = Iで確かめたが、これは指標として使えなかった。** ‖e^A‖‖e^(−A)‖ が大きいと、正しい実装でも誤差が膨らむ（scipyでもn = 1024でmax|expm(A) expm(−A) − I| = 2.7e4）。その指標で書いた最初の表は取り下げた。
- CのA_c（安定な系の行列。要素は3程度、1-ノルムはnとともに増える）では、scipyの`expm`との相対差がn = 256で3.2e-7、n = 1024で0.94%、n = 4096で115%（Cumo）だった。CuPyの`expm`は、同じ行列で1.9e-13以内。
- 比較として、CuPyの`expm`はPadé(13)に固定で、トレースを引いたあとの1-ノルムからscalingを決める（Higham 2005の簡略版）。scipyはAl-Mohy & Higham 2009の、次数を選ぶ版である。
- CとFの関門（small）はn = 8とn = 16で1-ノルムが小さいので、最初はこれを捕まえられなかった。いまはCのsmall fixtureに対称な256 × 256のE_wideを置き、`expm_vs_eigh_wide_symmetric`（≤ 1e-10）で確かめている。NumoとCumoでは2.95e-5で落ち、`known_failures.json`に既知の失敗として記録してある。numpyは5.8e-15、CuPyは1.1e-14で通る。cumoはmaster `90935207`（#584）で直り、4.2e-15で通るようになったので、記録はNumoだけに直した（2026-10-03）。

## 1. SFloatの`lstsq`が、呼ぶたびに標準出力へデバッグ出力を書く

- 再現: `Numo::Linalg.lstsq(Numo::SFloat.new(6, 3).rand, Numo::SFloat.new(6).rand)`を呼ぶと、`start of SGELSD`（2回）、`overdetermined, path 1`、`overdetermined, path 1a`、`end of SGELSD`の5行が出る。
- 出所は、同梱のOpenBLAS 0.3.34のCに変換されたLAPACK、`lapack-netlib/SRC/sgelsd.c`にある`fprintf(stdout, ...)`である。DFloat（`dgelsd.c`）では出ない。
- 影響: `script/bench.rb`は最後の行（タブ区切り8列）だけを読むので、集計には影響しない。ただし、CPUのfp32の`lstsq`の計測区間で、fprintfを呼んでいることになる。

## 2. `svd(a, driver: 'sdd', job: 'S')`が横長の行列で落ちる。そのため`orth`も落ちる

- 再現: `Numo::Linalg.svd(Numo::DFloat.new(3, 5).rand, driver: 'sdd', job: 'S')`は、`LapackError: the 11-th argument had illegal value`（`LAPACKE_dgesdd_work`の11番目の引数）になる。行列が連続でも転置ビューでも同じ。
- `driver: 'svd'`や`job: 'A'`なら通る。縦長（m ≥ n）なら通る。
- `orth`は内部でこの組み合わせを使うので、横長の行列に`orth`を呼ぶと落ちる。Cumoの`orth`は横長でも通る。
- Dの関門では、行空間を`orth(Xᵀ)`で求めるのをやめ、列空間で比べる形に組み直した（[results/procrustes_pca.md](results/procrustes_pca.md)）。

## 3. `matrix_power(a, n)`が二乗法ではなく、n − 1回の積を順に取る

- `lib/numo/linalg.rb`の`matrix_power`は、`(n - 1).times { r = Numo::Linalg.matmul(r, a) }`である。Cumo、numpy、CuPyは二乗法を使う。
- n = 64では、63回のGEMMと、二乗法の6回ほどの差になる。単発の値で、n = 4096のfp32はNumo 6367 ms対numpy 494 ms（13倍）、n = 1024のfp64は348 ms対18.5 ms（19倍）だった。
- アルゴリズムが違うので、CPUの表（Numo / numpy）からは`matrix_power`を外した。GPUの表（Cumo / CuPy）は、両方とも二乗法なので測っている。

## 4. `lu_solve`が、右辺1本でもO(n²)より重い（疑い）

- 単発の値で、n = 4096のfp32の`lu_solve`（右辺1本）が、Numo 76 ms対numpy 1.9 ms（40倍）だった。n = 1024のfp32でも2.9 ms対0.076 ms。
- `lu_solve`は`getrs`を1回呼ぶだけで、アルゴリズムはnumpy（scipy）と同じである。行優先のn × nのLUを、呼ぶたびに列優先へ並べ替えているのではないかと疑っているが、確かめていない。
- 10ラウンドの計測はCPUの表で取る（[results/markov_diffusion.md](results/markov_diffusion.md)）。

## 5. `eigh`の既定は`syev`で、`turbo: true`を付けたときだけ`syevd`になる

- numpy、CuPy、Cumoの`eigh`は`syevd`である。そろえるため、Dの計測の1ステップでは`turbo: true`を付けている。
- そろえても、CPUの`eigh`はnumpyのほうが1.2〜2.8倍速かった（[results/procrustes_pca.md](results/procrustes_pca.md)）。BLAS / LAPACKのビルドの差として読んでいる。

## 6. `svd`の既定は`gesvd`、numpyの`svd`は常に`gesdd`

- アルゴリズムが違うので、そろえる前の単発の値では、d = 2048のfp32の縦長SVDで、Numoがnumpyの11倍遅く見えた。CPUのPython版を`scipy.linalg.svd(lapack_driver='gesvd')`にそろえると、差は1.2倍になった。

## 7. SFloatの一般化`eigh(a, b)`が、n ≥ 2895で例外を出さずに誤った固有値を返す

2026-10-03、numo-linalg-alt 0.10.1（同梱OpenBLAS 0.3.34）。A題材のCPUの計測で、`eigh_general`のfp32のn = 4096だけが、n = 2048の1.3倍の時間（302 ms対233 ms）しかかからないことから気づいた。

```ruby
x = Numo::SFloat.new(n, n).rand - 0.5
a = (x + x.transpose) / 2
w = Numo::Linalg.eigvalsh(a)
wh, = Numo::Linalg.eigh(a, Numo::SFloat.eye(n), turbo: true)
(w - wh).abs.max
```

| n | max\|eigvalsh − eigh(a, I)\| |
|---|---|
| 2048 | 6.1e-05 |
| 2894 | 7.6e-05 |
| 2895 | 3.7e+11 |
| 2896 | 3.7e+11 |

- n = 2895から、OpenBLASが`On entry to SSYGVD parameter number 11 had an illegal value`を標準エラーに書く。それでも`eigh`は例外を出さず、0や1e30のような値を返す。
- 11番目の引数は作業領域の長さLWORKである。`ssygvd`が要求する長さは1 + 6n + 2n² で、n = 2895で初めて2²⁴ を超える（16,779,421）。これをfloat32に丸めると16,779,420になり、1足りない。失敗の始まるnが、この計算とぴったり一致する。作業領域の問い合わせの結果がfloat32で返され、丸めで足りなくなっている疑いがある。丸めがLAPACKとLAPACKEのどちらで起きるかは、ソースで確かめていない。
- numo-linalg-altは`LAPACKE_ssygvd`を呼ぶ（`ext/numo/linalg/lapack/sygvd.c`）。LAPACKEが返す負のinfoを、例外にしていないように見える。
- DFloatでは起きない（n = 4096で残差4.5e-17）。SFloatでも、標準の`eigvalsh`と`eigh`はn = 4096で正しい。scipy（`eigh(a, b, driver='gvd')`）とCumo（cuSOLVERの`sygvd`）は、n = 2895と4096のfloat32でも正しい値を返した。
- A題材のCPUの表の、fp32のn = 4096の`eigh_general`は、この不具合に当たるので無効とした（[results/hartree_fock.md](results/hartree_fock.md)）。関門はfp64のsmallで回しているので、これを捕まえられなかった。
