# narray-linalg-bench実装プラン

作成日: 2026-09-29
目的: `Cumo::Linalg`(cuSOLVER実装、別プラン`cumo-linalg-plan.md`)を **numo-linalgと互換か(計算結果を含む)**、**CuPyと比べて速いか** の2点で検証する。narray-llmがLLMで果たした役割を、線形代数で果たすプロジェクト。

進め方はnarray-llmの作法をそのまま継ぐ。**前の段階の受け入れテストが通るまで次の段階のコードを書かない。** 関門(正しさ)と計測(速さ)は別のコマンドに分ける。

---

## 0. 結論(先に読むところ)

- **A〜Gの7題材**で、`Numo::Linalg`の公開関数31個をすべて踏む(§3のカバー表)。各題材は物理的・解析的に決まる性質を関門にし、「Numoと一致した」ではなく「正しい答えを出した」で判定する。
- **3実装を並べる**: Ruby(`XM`の切り替えでNumo::Linalg / Cumo::Linalg)、Python(numpy+scipy / CuPy)。**同じ題材を同じアルゴリズムで、同じfixtureを読んで**動かす。
- **CuPyに無い関数がある**(一般化`eigh(a, b)`、`ldl`、`cho_solve`系、`null_space` / `orth`、`lstsq`のdriver選択)。`eig` / `cond` / `expm`はある。CuPy双子は無いところを **同じアルゴリズムで自前に組む** か、それも無理ならnumpyに落とすかを題材ごとに決めて記録する(§5)。これはcumo側の成果にもなる。
- dtypeは **関門はDFloat、計測はSFloatとDFloatの両方**。コンシューマGPUのfp64は1/64レートなので、GPU表はSFloatが主、DFloatは「精度が要る用途でどれだけ遅いか」を示す副。
- fixtureは **Numoで生成して`data/`にバイナリで置き、3実装が同じバイトを読む**。NumoとCumoとnumpyの乱数は一致しないので、生成は1か所。

---

## 1. リポジトリの形

```
narray-linalg-bench/
├── AGENTS.md                 # 作法(narray-llm のものを線形代数向けに書き直す)
├── README.md                 # 結果表(GPU: Cumo / CuPy、CPU: Numo / numpy+scipy)
├── Gemfile                   # numo-narray-alt, numo-linalg-alt, test-unit; group :gpu { cumo }
├── Rakefile                  # rake test / rake gate / rake bench
├── lib/
│   ├── linalg_bench.rb
│   └── linalg_bench/
│       ├── backend.rb        # XM / XL / XF、GPU=1、DTYPE=、scalar 読み戻し
│       ├── fixtures.rb       # data/ の読み書き(to_binary / from_binary)
│       ├── gates.rb          # 残差・直交性・許容誤差の共通関数
│       ├── timer.rb          # best-of-N、ウォームアップ、区間末の同期
│       └── problems/
│           ├── hartree_fock.rb        # A
│           ├── gaussian_process.rb    # B
│           ├── kalman_riccati.rb      # C
│           ├── procrustes_pca.rb      # D
│           ├── vandermonde_lstsq.rb   # E
│           ├── markov_diffusion.rb    # F
│           └── kkt.rb                 # G
├── script/
│   ├── make_fixtures.rb      # data/ を生成(Numo、DFloat、決定的)
│   ├── gate.rb               # 関門だけを全題材で回す(GPU=1 で Cumo)
│   └── bench.rb              # 1 題材 × 1 サイズ × 1 dtype を計測して 1 行出す
├── python/
│   ├── requirements.txt      # numpy, scipy, cupy-cuda13x
│   ├── backend.py            # xp / xl の切り替え(GPU=1 で cupy)、同期
│   ├── fixtures.py
│   ├── gates.py
│   ├── problems/             # Ruby と 1 対 1 の 7 ファイル
│   ├── gate.py
│   └── bench.py
├── bench/
│   ├── run.sh                # narray-llm の run.sh を継ぐ(インターリーブ、対照系列)
│   ├── conditions.tsv        # 題材 × サイズ × dtype × 実装コマンド
│   └── aggregate.py          # 比の中央値、n/10
├── data/                     # fixture(git には入れない。make_fixtures.rb で再生成)
├── test/                     # 受け入れテスト(両バックエンドで通す、GPU 無しでも Numo で全部通す)
└── docs/
    ├── plans/PLAN-linalg-bench.md   # このファイル
    ├── method.md                    # 比較の条件と手順
    ├── cupy-gaps.md                 # CuPy に無い関数とその埋め方
    ├── cumo-issues.md               # 見つけた cumo の問題、棄却した仮説
    └── results/<題材>.md            # 題材ごとの数字と経過
```

### backend.rbの要点

```ruby
require 'numo/narray'
require 'numo/linalg'          # numo-linalg-alt。無ければ LoadError で止める(CPU 側の前提)
HM = Numo

if ENV['GPU'].to_s =~ /\A(1|on|true)\z/i
  require 'cumo/narray'
  require 'cumo/linalg'
  XM = Cumo
else
  XM = Numo
end
XL = XM::Linalg                 # 題材コードは XL.solve / XL.eigh とだけ書く
XF = ENV['DTYPE'] == 'fp64' ? XM::DFloat : XM::SFloat   # 既定 SFloat。関門は fp64 で回す
```

- 題材コードは`Numo::` / `Cumo::`を直接書かない。`XM`、`XL`、`XF`だけ。
- スカラーの読み戻し(Cumoの縮約は0次元NArray)は`LinalgBench.scalar`に1か所。**反復ループの中では読み戻さない**(収束判定は間引く)。narray-llmの作法と同じ。
- Bit / マスク代入 / NArray添字は避ける。

### Python側

```python
if os.environ.get("GPU") in ("1", "on", "true"):
    import cupy as xp
    import cupy.linalg as xl
    import cupyx.scipy.linalg as xsl
    sync = lambda: xp.cuda.Stream.null.synchronize()
else:
    import numpy as xp
    import numpy.linalg as xl
    import scipy.linalg as xsl
    sync = lambda: None
```

- **`CUPY_TF32=0`**、cuBLASのmath modeを起動時に出力して確認(narray-llmのmethod.mdと同じ)。cumo側は`CUMO_ALLOW_TF32`を立てない。
- CuPyに無い関数は`python/problems/`の中で **同じアルゴリズムをxpの基本演算で組む**。numpyに落とす場合は`gaps.py`に集約し、`--allow-cpu-fallback`を付けた時だけ動く(既定は落ちる → CuPy列が「×」になる)。

---

## 2. 共通の約束事

### 関門(正しさ)

- **DFloatで回し、許容誤差は絶対値でなく相対残差**で書く。`solve`なら`‖A x − b‖ / (‖A‖ ‖x‖ + ‖b‖) ≤ tol`、直交性なら`‖QᵀQ − I‖_max ≤ tol`。
- tolは **DFloat 1e-10、SFloat 1e-4** から始め、Numo(OpenBLAS)で測った実際の残差の10倍に **測ってから** 固定する。表にする。
- 分解の結果そのもの(Q、U、固有ベクトル)は符号・順序が一意でないので比べない。**再構成と性質**で比べる。固有値・特異値は降順にソートしてから比べる。
- **NumoとCumoの直接比較**は関門とは別に`test/parity_test.rb`で行う。同じfixtureを両方に入れ、`Cumo::Linalg.f(a).to_a`と`Numo::Linalg.f(a)`の差が型別tol以内。これは「互換性(計算結果含む)」の直接の証拠になる。分解結果は上と同じ理由で再構成で比べる。
- 3実装(Ruby / numpy / CuPy)は **同じfixtureから同じ関門値(エネルギー、RMSD、残差)をJSONに書き出し**、`script/compare_gates.rb`が並べる。ここで3者が揃わなければ、どれかのアルゴリズムが違う。

### 計測(速さ)

- narray-llmの`bench/run.sh`を継ぐ: **Cumo / CuPy / 対照(Cumoをもう一度)の3系列を1ラウンド内でインターリーブ**、順序をラウンドごとに回転、11ラウンドの先頭を捨てて10。比は同じラウンド内の比の中央値、後ろにn/10。**対照が1をまたがない条件は結果に載せない。**
- CPU表(Numo / numpy+scipy)は別に取る。GPUとCPUを同じ表に混ぜない。
- best-of-3、ウォームアップ1回、区間内に同期を挟まない、区間末で同期(CuPyは`Stream.null.synchronize()`、Cumoは結果のスカラー1個を読み戻す)。**`Cumo::Linalg`はdevInfoを読むために各呼び出しで同期する**ので、区間内の同期回数がCuPyと同じかをnsysで一度数えてmethod.mdに書く。
- クロック固定(`nvidia-smi -lgc` / `-lmc`)。fixtureの読み込み、プロセス起動は含めない。
- 1プロセスが2秒以上回るように反復回数を条件ごとに決めて`conditions.tsv`に書く。
- **小さいnでGPUが負けるのは正常。** 各題材で **CPUとGPUが逆転するn** を出し、それをREADMEに載せる。

### fixture

- `script/make_fixtures.rb`がNumoのDFloatで生成し、`data/<題材>_<サイズ>.bin`(`to_binary`、行優先、shapeはファイル名orヘッダ)に置く。Pythonは`np.fromfile(..., dtype='<f8').reshape(...)`。
- 乱数はNumoの`rand`をseed固定で1回だけ。**Cumoでもnumpyでも生成しない。**
- 対称・正定値・直交などの性質は生成時に作り込む(`A = B Bᵀ + n I`、QRのQ、など)。
- サイズは3段: `small`(関門、n ≦ 64)、`mid`(逆転点を探す、n = 128〜1024)、`large`(GPUの本領、n = 2048〜8192、メモリに収まる範囲)。

---

## 3. 題材とカバー表

| 関数 | A | B | C | D | E | F | G |
|---|---|---|---|---|---|---|---|
| `dot` / `matmul` | ● | ● | ● | ● | ● | ● | ● |
| `matrix_power` | | | ● | | | ● | |
| `qr` | | | | | ● | | |
| `svd` / `svdvals` | | | | ● | ● | | |
| `orth` / `null_space` | | | | ● | ● | ● | |
| `lu` / `lu_fact` / `lu_solve` / `lu_inv` | | | | | | ● | |
| `ldl` | | | | | | | ● |
| `cholesky` / `cho_fact` / `cho_solve` / `cho_inv` | ● | ● | | | | | |
| `eig` / `eigvals` | | | ● | | | ● | |
| `eigh` / `eigvalsh` | ● | | | ● | | | ● |
| `norm` / `cond` | | | ● | ● | ● | | |
| `det` / `slogdet` | | ● | | ● | | | |
| `matrix_rank` | | | | ● | ● | | |
| `solve` / `inv` | ● | ● | ● | | | | ● |
| `lstsq` / `pinv` | | | | | ● | | ● |
| `expm` | | | ● | | | ● | |

31関数のうち`blas_char`以外の30がどこかで踏まれる。**AとGはcuSOLVER実装の難所(一般化固有値、`sytrf`のrow/col-major変換)を直撃する**ので、関門として特に重い。

---

## 4. 題材の仕様

各題材は`Problem`クラス1つ。`fixture(size)` / `run(dtype)` / `gate(result)` / `bench_step`の4メソッドを持ち、`gate.rb`と`bench.rb`はそれしか呼ばない。Pythonも同じ4関数。

### A. Hartree-Fock(H₂ / HeH⁺、STO-3G)

- **入力**: 1電子積分H_core、重なりS、2電子積分(μν|λσ)。**積分は自分で計算せず、fixtureに置く**(文献値or PySCFで1回だけ生成し、data/ に固定)。H₂ は基底2個、HeH⁺ も2個。
- **手順**: (1) `X = S^{-1/2}`を`eigh(S)`で作る(または`cholesky`で`X = L^{-T}`)。(2)密度PからFock行列Fを作る。(3) **`eigh(F, S)`の一般化固有値問題**を解いてCとεを得る(X経由の`eigh(XᵀFX)`も別経路として持ち、両方が同じεを出すことを関門に加える)。(4) Pを更新、収束まで。
- **関門**: H₂(R = 1.4 bohr)の全エネルギーが **−1.1167 Haに1e-6以内**(STO-3Gの文献値)。SCFのエネルギーが単調減少。`eigh(F, S)`と`eigh(XᵀFX)`のεが一致。CのS-直交性`‖CᵀSC − I‖_max ≤ tol`。
- **サイズ**: 関門は基底2。計測は **HFの形を保った合成問題**: 基底n = 128〜4096の対称HとSPD Sをfixtureに置き、`eigh(H, S)`と`cholesky(S)` + `solve`を1ステップ分測る(SCFを回さない。2電子積分はn⁴ で持てないので、Fock構築は`H + P`の擬似で代用)。
- **踏む関数**: `eigh(a, b)`, `eigh`, `eigvalsh`, `cholesky`, `cho_solve`, `inv`, `solve`, `dot`.
- **CuPy**: `cupy.linalg.eigh`は一般化に非対応 → `cholesky(S)`で`L⁻¹ H L⁻ᵀ`に落として標準固有値問題にする(`solve_triangular`は`cupyx.scipy.linalg`にある)。cumoの`eigh(a, b)`は`Xsygvd`を直接叩くので、ここはcumoが1呼び出しで済む対CuPy 4呼び出し(cholesky + solve_triangular ×2 + eigh)の比較になる。

### B. ガウス過程回帰(1D → 2D)

- **入力**: 訓練点X(n × d)、観測y、テスト点X*。カーネルはRBF、超パラメータ(ℓ, σ_f, σ_n)を格子で走査。
- **手順**: `K = k(X, X) + σ_n² I`、`L = cholesky(K)`、`α = cho_solve(L, y)`、予測平均`k*ᵀ α`、予測分散、対数周辺尤度`−½ yᵀα − Σ log L_ii − (n/2) log 2π`。`slogdet(K)`でも同じlog detを出し、両者の一致を関門に加える。
- **関門**: (1) fixtureは既知の(ℓ, σ_f, σ_n)で生成した関数 + ノイズなので、**周辺尤度の最大点が生成パラメータの格子の隣接以内**。(2) `cho_solve(L, y)`と`solve(K, y)`と`inv(K).dot(y)`の3経路のαがtol以内で一致。(3) `Σ log L_ii × 2`と`slogdet(K)[1]`が一致。(4)予測平均が訓練点上でyにσ_nの範囲で乗る。
- **サイズ**: n = 64(関門)、512 / 1024 / 2048 / 4096 / 8192(計測)。`cholesky`のO(n³)が主。
- **踏む関数**: `cholesky`, `cho_fact`, `cho_solve`, `cho_inv`, `solve`, `inv`, `slogdet`, `det`, `dot`.
- **CuPy**: `cholesky`あり。`cho_solve` / `cho_inv`は **無い** → `solve_triangular` 2回で組む。`slogdet`あり。

### C. Kalmanフィルタ / 離散Riccati

- **入力**: 連続時間の線形系(A_c, B_c, C, Q, R)、状態次元n、観測次元m。Fixtureは **安定なA_cを作り込む**(固有値の実部を負に)。
- **手順**: (1) `expm(A_c Δt)`で離散化A_d。(2)離散Riccatiを反復`P ← A P Aᵀ − A P Cᵀ (C P Cᵀ + R)⁻¹ C P Aᵀ + Q`(`solve`で逆行列を避ける経路と`inv`の経路の両方)。(3)定常ゲインK、閉ループ`A − K C`。(4)観測列をfixtureから読んでフィルタを走らせる。
- **関門**: (1) Riccatiの定常残差`‖P − f(P)‖ / ‖P‖ ≤ tol`。(2) **閉ループの`eigvals`がすべて`|λ| < 1`**。(3) `expm(A_c Δt)`を`matrix_power(expm(A_c Δt/2^k), 2^k)`と比べて一致。(4) `cond(C P Cᵀ + R)`を出力し、fixture設計時の値と一致。(5)フィルタの推定誤差の分散がPの対角と整合(統計的関門: ±20%)。
- **サイズ**: n = 8(関門)、64 / 256 / 1024(計測、反復200回)。`expm`はn単独でも測る(n = 256〜4096)。
- **踏む関数**: `expm`, `matrix_power`, `solve`, `inv`, `eig`, `eigvals`, `cond`, `norm`, `dot`.
- **CuPy**: `eig` / `eigvals`あり(`cusolverDnXgeev`)。**計測はRiccati反復とexpmを主**にし、`eigvals`は別条件で単独に測る(geevはGPUで遅い可能性が高く、混ぜるとRiccatiの比が読めない)。`expm`は`cupyx.scipy.linalg.expm`にある。numo-linalgの`expm(a, ord=8)`はPadé(8) + scaling-squaring、CuPyの実装が同じ次数かは第一段階で読んでmethod.mdに書く(違えば関門tolに効く)。

### D. Kabsch / Procrustes + 点群PCA

- **入力**: 点群P(N × 3)、それを既知の回転R₀ + 平行移動 + ノイズで動かしたQ。別にPCA用の高次元点群X(N × d)。
- **手順**: (1)重心を引いて`H = PᵀQ`、`U, S, Vᵀ = svd(H)`、`d = sign(det(V Uᵀ))`、`R = V diag(1,1,d) Uᵀ`。(2) RMSD。(3) PCA: `X`の共分散を`eigh`、または`svd(X)`直接。両方の主成分の張る空間が一致することを`orth`で確認。(4) `matrix_rank(X)`、`norm(X, 'fro') = ‖svdvals‖₂`、`norm(X, 2) = svdvals[0]`、`norm(X, 'nuc') = Σ svdvals`。
- **関門**: (1)ノイズ0なら **RMSD ≤ 1e-12(DFloat)、R ≈ R₀**。(2) **`det(R) = +1`**、反転したfixtureではd = −1が立つ。(3) PCAの上位k主成分の張る空間が`eigh`経路と`svd`経路で一致(`‖U_k U_kᵀ − V_k V_kᵀ‖ ≤ tol`)。(4) 3つのノルム等式。(5) `matrix_rank`がfixtureの設計ランクと一致(ランク落ちXを用意)。
- **サイズ**: N = 1000, d = 3(関門)。計測は **svd(H)は3×3で意味が無い**ので、PCA側でN = 8192、d = 256 / 512 / 1024 / 2048の`svd(X)`(縦長)と`eigh(XᵀX)`を測る。ここが **m > nのSVD**、cumoの転置コピー経路を踏む。
- **踏む関数**: `svd`, `svdvals`, `det`, `eigh`, `orth`, `matrix_rank`, `norm`(fro / 2 / nuc), `dot`.
- **CuPy**: `svd`, `eigh`, `det`, `matrix_rank`, `norm`あり。`svdvals`は`svd(compute_uv=False)`。`orth`は無い → `svd`から自前。

### E. Vandermonde最小二乗(ランク落ちあり)

- **入力**: 標本点x(m個)、真の多項式係数、ノイズ付きy。Vandermonde V(m × n)。**列を複製してランク落ちさせたV'** も用意。
- **手順**: (1) `lstsq(V, y)`の3 driver(lsd / lss / lsy)。(2) `qr(V)`で`R⁻¹ Qᵀ y`(mode reduce / r / rawの3つ)。(3) `pinv(V).dot(y)`。(4)正規方程式`solve(VᵀV, Vᵀy)`(条件数が悪いので誤差が大きい、それをcondで説明)。(5)ランク落ちV' で`lstsq`と`pinv`の **最小ノルム解**、`null_space(V')`。
- **関門**: (1) 4経路(lstsq / qr / pinv / 正規方程式)の解が、condに応じたtolで一致(正規方程式は`cond²`倍の緩いtol)。(2) `‖QᵀQ − I‖ ≤ tol`、`Q R = V`。(3) **次数を上げると`cond(V)`が指数的に増える**(隣接次数の比が1より大きい単調性)。(4)ランク落ちで`matrix_rank(V') = n − 複製数`、`lstsq`のrank出力が一致、`null_space(V')`の各列で`V' z ≈ 0`、解が`lstsq`と`pinv`で一致し **そのノルムが他の解より小さい**(`x + z`と比べる)。
- **サイズ**: m = 256, n = 8(関門)。計測は縦長m = 4096 / 16384 / 65536、n = 16 / 64 / 256。`qr`と`lstsq`のtall-skinny。
- **踏む関数**: `lstsq`(3 driver), `qr`(3 mode), `pinv`, `cond`, `matrix_rank`, `null_space`, `svd`, `solve`, `norm`.
- **CuPy**: `lstsq`, `qr`, `pinv`, `matrix_rank`, `cond`あり(`lstsq`はSVDベース1種類のみ、driver指定なし)。`null_space`は無い → `svd`から自前。

### F. マルコフ連鎖 / 拡散方程式の陰解法

- **入力**: 既約な確率行列P(n × n、fixtureで既約性と非周期性を作り込む)、連続時間の生成行列Q(行和0)。拡散: 1D格子n点の陰解法行列`M = I − Δt D L`(Lは離散ラプラシアン、**密行列として持つ**)。
- **手順**: (1) `eig(Pᵀ)`の固有値1の固有ベクトル → 定常分布π。(2) `null_space(Pᵀ − I)`でもπを出す。(3) `matrix_power(P, k)`の行がπに収束。(4) `expm(Q t)`が確率行列(行和1、非負)、`expm(Q t)`の定常分布が(1)と一致。(5)拡散: `lu_fact(M)`を1回、`lu_solve`を時間ステップごとに。`lu(M)`のP L U = M、`lu_inv`と`inv`の一致。
- **関門**: (1) π ≥ 0、Σπ = 1、`πᵀ P = πᵀ`(残差 ≤ tol)。(2) `eigvals(P)`の最大絶対値が1、他は < 1(非周期性)。(3) `matrix_power(P, k)`の各行とπの差が幾何収束(第2固有値の比で説明できる)。(4) `expm(Qt)`の行和1、非負、`expm(Q t₁) expm(Q t₂) = expm(Q (t₁+t₂))`。(5)拡散の総量保存(Σuが一定)、最大値原理(uの最大が減少)、`P L U = M`再構成、`lu_inv(M) = inv(M)`。
- **サイズ**: n = 16(関門)。計測: `eig`はn = 128 / 512 / 2048(geevはGPUで遅い、それ自体が結果)、`matrix_power`と`expm`はn = 256〜4096、拡散はn = 1024 / 4096 / 8192で`lu_solve`を1000ステップ。
- **踏む関数**: `eig`, `eigvals`, `null_space`, `matrix_power`, `expm`, `lu`, `lu_fact`, `lu_solve`, `lu_inv`, `inv`, `dot`.
- **CuPy**: `lu`, `lu_factor`, `lu_solve`, `expm`は`cupyx.scipy.linalg`に、`eig` / `eigvals`は`cupy.linalg`にある。`lu_inv`は`lu_solve(I)`、`null_space`は`svd`から自前。

### G. KKT系(等式制約付き最小二乗)

- **入力**: `min ‖A x − b‖²  s.t.  B x = c`(A: m × n、B: p × n)。KKT行列`K = [[AᵀA, Bᵀ], [B, 0]]`((n+p) × (n+p)、対称不定値)。
- **手順**: (1) `ldl(K)`でL、D、perm。(2) `solve(K, [Aᵀb; c], driver: 'sym')`(sytrf/sytrs経路)と`driver: 'gen'`(LU経路)。(3) `inv(K, driver: 'sysv')`。(4) `eigvalsh(K)`で慣性(正・負・零の個数)。(5)制約消去の別経路: `null_space(B)`の基底Zで`x = x₀ + Z y`、`y`を`lstsq(A Z, b − A x₀)`。
- **関門**: (1)解が制約を満たす`‖B x − c‖ ≤ tol`、Lagrange条件`‖AᵀA x − Aᵀb + Bᵀλ‖ ≤ tol`。(2) sym経路とgen経路と制約消去経路のxが一致。(3) **慣性が(n, p, 0)**(Aがフルランク、Bが行フルランクなら理論値)。(4) `L D Lᵀ`の再構成がpermを戻してKと一致。(5) `inv(K)`と`solve(K, I)`の一致。
- **サイズ**: n = 16, p = 4(関門)。計測: n + p = 512 / 1024 / 2048 / 4096。
- **踏む関数**: `ldl`, `solve`(sym / gen), `inv`(sysv), `eigvalsh`, `null_space`, `lstsq`, `dot`.
- **CuPy**: **`ldl`が無い**(cuSOLVERの`Xsytrf`をCuPyは包んでいない)。`solve`もLU 1種で`driver: 'sym'`に当たる経路が無い。KKTは`solve`(LU)で解いて計測し、`ldl`の再構成関門はCuPy列を「×」にする。慣性は`eigvalsh`で取れる。

---

## 5. CuPyに無いもの(docs/cupy-gaps.mdに育てる)

CuPy main(2026-09-26時点、v14系)の`cupy/linalg/`と`cupyx/scipy/linalg/`を読んで確認したもの。

**CuPyにあるもの**: `dot` / `matmul` / `matrix_power`、`cholesky` / `qr`(reduced / complete / r / raw)/ `svd`、`eig` / `eigvals`(`cusolverDnXgeev`、CUDA 11.7以降でゲート)/ `eigh` / `eigvalsh`、`norm` / `cond` / `det` / `slogdet` / `matrix_rank`、`solve` / `inv` / `lstsq` / `pinv`、`cupyx.scipy.linalg`の`lu` / `lu_factor` / `lu_solve` / `expm` / `solve_triangular`。

| numo-linalgの関数 | CuPy | 双子での埋め方 | 意味 |
|---|---|---|---|
| `eigh(a, b)`一般化 | 無い(`eigh(a, UPLO)`のみ) | `cholesky(b)` + `solve_triangular` ×2 + `eigh` | cumoは`Xsygvd` 1発。呼び出し回数の差が測れる |
| `eigh(vals_range:)` | 無い | 全固有値を出して切る | |
| `ldl` | 無い | 無し(関門は×) | cumoが`Xsytrf`を持てばCuPyに無い機能 |
| `solve(driver: 'sym' / 'pos')` | 無い(LU 1種) | `cholesky` + `solve_triangular` | |
| `cho_fact` / `cho_solve` / `cho_inv` | 無い | `cholesky` + `solve_triangular` ×2 | |
| `lu_inv` | 無い | `lu_solve(I)` | |
| `null_space` / `orth` | 無い | `svd`から自前 | |
| `svdvals` | 無い | `svd(compute_uv=False)` | |
| `lstsq`のdriver(lsd / lss / lsy) | SVD 1種 | lsd相当だけ | |
| `qr(mode: 'reduce' / 'r' / 'raw')` | `reduced` / `r` / `raw`あり | 名前を写す | |
| `eig(left:)`左固有ベクトル | 無い(右のみ) | 左は`eig(aᵀ)`で | |
| `expm(ord)` | `cupyx.scipy.linalg.expm`(ord指定なし) | 既定で比べ、Ruby側と同じ式かは確認 | |

**CuPy側だけにあってcumoに無いもの**(benchには出ないが、cumo-linalg-planの「あとで」に書き戻す): バッチ(`[..., M, M]`)、`gesvdj` / `syevj`(Jacobi系)、`gesvda`、`potrfBatched` / `potrsBatched`、`gesv` / `gels`のIRS混合精度経路(cupyはCC 7.0以上でこれを使う)、`tensorsolve` / `tensorinv`、`solve_triangular`。**特にバッチと`solve_triangular`は、cumoの`cho_solve`を組む時に欲しくなる。**

**これはcumo側の宣伝材料でもある。** READMEに「CuPyには無くCumo::Linalgにはある関数」の節を作る(一般化固有値、LDL、driver選択)。

## 6. 段階

### 第0段階: 計画 — このファイル

### 第一段階: 足場(Numoだけで通す)

- `backend.rb`、`fixtures.rb`、`gates.rb`、`timer.rb`、`Problem`の基底クラス、`gate.rb` / `bench.rb`の骨組み
- `make_fixtures.rb`が7題材の`small` fixtureを作る
- Python側の`backend.py` / `fixtures.py` / `gates.py`と、fixtureを読んで **バイト一致**することのテスト
- `bench/run.sh`をnarray-llmから移し、`conditions.tsv`の形式を決める

**受け入れテスト**: 7題材の`small` fixtureがRubyとPythonで同じ値に読める。`gate.rb`が「題材なし」で空のJSONを出す。**GPU無しで全部通る。**

### 第二〜第八段階: 題材B → E → D → F → C → G → Aの順に1題材ずつ

順序の理由: **B(cholesky)とE(qr / lstsq / svd)はcuSOLVERプランのStage 2 / 4 / 5に対応して早く使えるようになる**。A(一般化固有値)とG(ldl)はcumo側の実装が最後になるので、bench側も最後。CとFはcumo側の`Xgeev`(Stage 7)に依存する。

各題材で:
1. Ruby実装(`XM` / `XL` / `XF`だけで書く)と関門。**Numoで関門が通る** ことが最初の受け入れ
2. Python実装(numpy / scipy)。**Rubyと同じ関門値がJSONで一致**
3. CuPyで動かす(`GPU=1 python`)。無い関数は §5に従って埋め、`cupy-gaps.md`に書く
4. Cumoで動かす(`GPU=1 ruby`)。**cumo側の該当Stageが未実装なら`NotImplementedError`でskip** し、そのことを結果ファイルに書く(bench側を止めない)
5. `parity_test.rb`にNumo vs Cumoの直接比較を足す
6. `mid` / `large` fixture、`conditions.tsv`に条件を足す、`bench.rb`で1条件だけ測って所要時間が2秒以上になる反復回数を決める
7. `docs/results/<題材>.md`に関門の値と最初の数字

**受け入れテスト(各題材共通)**: Numoで関門が通る。numpyで同じ関門値。CuPyで関門が通る(またはgapsに記録された×)。Cumoで関門が通る(または未実装skip)。parityが通る。

### 第九段階: 計測の本番

- 7題材×サイズ× dtype(SFloat / DFloat)を`run.sh`で回す(Cumo / CuPy / 対照)
- CPU表(Numo / numpy+scipy)を別に
- 逆転点の表
- **対照が1をまたがない条件を除いて** READMEに載せる
- nsysで各題材の代表条件について **区間内のカーネル本数と同期回数** をCumoとCuPyで数え、method.mdに書く(cumoのdevInfo同期がどれだけ効いているかはここで分かる)

### 第十段階: 見つけたものをcumoに戻す

- `cumo-issues.md`に、性能で負けた条件と原因の仮説、棄却した仮説
- cumo側のissue / PRに繋げる。narray-llmでcumo #245〜#284が出てきたのと同じ流れ

---

## 7. 数字の出し方(READMEの表の形)

題材ごとに1表。行はサイズ、列はCumo / CuPy / Cumo÷CuPy(n/10)、SFloatとDFloatで表を分ける。

```
### B. ガウス過程回帰 (cholesky + cho_solve + slogdet、1 反復あたり ms、小さいほど速い)

| n | Cumo | CuPy | Cumo / CuPy |
|---|---|---|---|
| 512  | ... | ... | 1.xx 10/10 |
| 2048 | ... | ... | ... |
| 8192 | ... | ... | ... |
```

比は「CuPy ÷ Cumo」で **1を超えればCumoが速い**(narray-llmと同じ向き)。CPU表はNumo / numpy+scipyで同じ形。

---

## 8. リスク・先に確認すること

| リスク | 対応 |
|---|---|
| cumo側の`Cumo::Linalg`が題材より遅れる | bench側はNumo / numpy / CuPyの3者で先に完成させる。Cumoは`NotImplementedError`でskipして進む。**benchはcumoを待たない** |
| CuPyの関数の有無 | §5はCuPy mainのソースで確認済み。第一段階で入れた版(`cupy-cuda13x`)の`dir()`で再確認して §5の版を書く |
| Hartree-Fockの積分fixture | PySCFで1回だけ生成してdata/ に固定し、生成スクリプトを`script/`に残す(PySCFは実行時依存にしない)。文献値 −1.1167 HaはSTO-3G / R=1.4 bohrのもの |
| fp64のGPU計測が遅すぎる | DFloat表は`mid`までにして`large`はSFloatだけ。「fp64は1/64レート」をmethod.mdに書く |
| geev(`eig`)がGPUでCPUより遅い | それ自体が結果。Fの`eig`行にCPUの数字を並べて載せる。CuPyも同じ`Xgeev`を使うので、cumo対CuPyの比は同じ土俵になる |
| 分解結果の非一意性でparityが落ちる | 再構成と性質で比べる(§2)。直接比較は固有値・特異値(ソート済み)と`solve`系の解だけ |
| 同期回数の差(cumoは各呼び出しでdevInfoを読む) | nsysで数えてmethod.mdに書く。差が大きければcumo側のStage 8(遅延devInfo)の根拠になる |
| numo-linalg-altとnumo-linalgの差 | Gemfileはalt。API差があれば`cumo-linalg-plan.md`の §7に書き戻す |

---

## 9. 実装セッションへの依頼事項

1. 第一段階をNumoだけで通す。**GPU無しでも`rake test`が全部通る状態を最後まで保つ。**
2. 題材はB → E → D → F → C → G → Aの順。**前の題材の受け入れテストが通るまで次を書かない。**
3. 題材コードに`Numo::` / `Cumo::` / `numpy.` / `cupy.`を直接書かない(`XM` / `XL` / `XF`、`xp` / `xl` / `xsl`だけ)。
4. 関門のtolは「測ってから固定」。最初のNumoの残差を`docs/results/<題材>.md`に書いてから決める。
5. CuPyに無い関数を埋めた時は必ず`cupy-gaps.md`に1行。numpyに落とした時は計測区間に入れない。
6. 計測の主張はrun.shの10ラウンド + 対照系列付きでのみ。単発の数字をREADMEに書かない。
7. cumoで見つけた問題は`cumo-issues.md`に、棄却した仮説も含めて残す。
