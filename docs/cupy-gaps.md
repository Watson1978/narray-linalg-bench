# CuPyに無いもの

numo-linalg-altの関数のうち、CuPyに同じものが無いものと、Python双子での埋め方を記録する。埋めたら、その題材の行に1行足す。numpyに落とした場合は、その処理を計測区間に入れない。

## 確かめた版

2026-10-01にcupy-cuda13x 14.2.0の`dir()`で確かめた。プラン §5の一覧と一致している。

- `cupy.linalg`: `cholesky`, `cond`, `det`, `eig`, `eigh`, `eigvals`, `eigvalsh`, `inv`, `lstsq`, `matmul`, `matrix_power`, `matrix_rank`, `norm`, `pinv`, `qr`, `slogdet`, `solve`, `svd`, `tensorinv`, `tensorsolve`（ほかに`cross`, `matrix_transpose`）
- `cupyx.scipy.linalg`: `expm`, `lu`, `lu_factor`, `lu_solve`, `solve_triangular`（ほかは特殊行列の生成関数）
- 無いもの: `cho_factor` / `cho_solve`, `ldl`, `null_space`, `orth`, `svdvals`, 一般化固有値の`eigh(a, b)`

## 埋め方

Python側の代替は`python/compat.py`にまとめてある。numpyも同じ関数を通るので、Pythonの2実装は同じアルゴリズムで動く。

| numo-linalg-altの関数 | CuPy | 埋め方 | 題材 | 状態 |
|---|---|---|---|---|
| `eigh(a, b)`一般化 | 無い | `cholesky(b)` + `solve_triangular` ×3 + `eigh`（`compat.eigh_general`）。numpyは`scipy.linalg.eigh(a, b, driver='gvd')`。一般化の`eigvalsh`の関門は× | A | Aで埋めた |
| `eigh(vals_range:)` | 無い | 全固有値を出して切る | | 未着手 |
| `ldl` | 無い | 無し（関門は×）。numpyは`scipy.linalg.ldl(lower=False)` | G | Gで記録した |
| `solve(driver: 'sym' / 'pos')` | 無い（LU 1種） | 関門は×（altもdriverを無視して常に`gesv`） | G | Gで記録した |
| `cho_fact` / `cho_solve` / `cho_inv` | 無い | `cholesky` + `solve_triangular` ×2。`cho_inv`は単位行列に`cho_solve`。`cho_fact`の関門は× | A, B | Bで埋めた（numpyも同じ綴り） |
| `lu_inv` | 無い | `lu_solve(I)` | F | Fで埋めた |
| `null_space` / `orth` | 無い | `svd`から自前。閾値はmax(m,n)·eps·s_max（altと同じ） | D, E, F, G | Eで`null_space`、Dで`orth`を埋めた（numpyも同じ綴り） |
| `svdvals` | 無い | `svd(compute_uv=False)` | D, E | DとEで埋めた |
| `lstsq`のdriver | SVD 1種 | 関門は×（altもdriverを無視して常にgelsd） | E | Eで記録した |
| `eig(left:)` | 右のみ | 左は`eig(aᵀ)` | | 未着手 |
| `pinv`の既定の閾値 | 1e-15（altとnumpyはmax(m,n)·eps） | 閾値を明示して渡す | E | Eで記録した |
| `expm(ord)` | 次数指定なし | 既定で比べる。Ruby側と同じ式かは読んで確かめる | C, F | 未着手 |
