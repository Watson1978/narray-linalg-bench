# AGENTS.md

このリポジトリは `Cumo::Linalg`（cuSOLVER 実装）を、**numo-linalg-alt と互換か（計算結果を含む）** と **CuPy と比べて速いか** の 2 点で検証するプロジェクト。進め方は [docs/plans/PLAN-linalg-bench.md](docs/plans/PLAN-linalg-bench.md) に従う。**前の段階の受け入れテストが通るまで、次の段階のコードを書かないこと。** 作法の多くは narray-llm の AGENTS.md から引き継いでいる。ここには線形代数で効くものだけを置く。

## 段階

| 段階 | 内容 | 状態 |
|---|---|---|
| 第一段階 | 足場（backend / fixtures / gates / timer、7 題材の small fixture、Ruby と Python のバイト一致） | 完了 |
| 第二〜第八段階 | 題材を B → E → D → F → C → G → A の順に 1 つずつ | 7 題材すべて、関門・parity・GPU と CPU の計測まで終わった |
| 第九段階 | 計測の本番 | 題材ごとに第二〜第八段階の中で行った。nsys による同期回数の比較は E の `qr` だけ |
| 第十段階 | 見つけたものを cumo に戻す | `docs/cumo-issues.md` に 5 件。cumo part3 のセッションに共有済み。1 と 2 は master で直り（#584、#585、#586）、`1c561d70` で確かめた。3（`cholesky` の `uplo`）も #587 で直り、`50e07e88` で確かめた。5（0.13.0 で小さい n の `expm` が遅くなった）は #618 で直り、リリースの 0.14.0 で確かめた。4（小さい n の同期）は、cumo 側で対応しないと決めた |

## バックエンドの切り替え

既定は Numo（CPU）で、`GPU=1` を付けると Cumo になる。dtype は `DTYPE=fp32`（既定）か `DTYPE=fp64` で選ぶ。`script/gate.rb`、`python/gate.py`、テストは、`DTYPE` が無ければ fp64 で動く。

```ruby
XM = Cumo or Numo          # lib/linalg_bench/backend.rb
XL = XM::Linalg
XF = XM::SFloat or XM::DFloat
HM = Numo                  # fixture の読み書きと生成だけが使う
```

- **題材コード（`lib/linalg_bench/problems/`）には `Numo::` / `Cumo::` を直接書かない。** 使うのは `XM`、`XL`、`XF` だけ。Python 側（`python/problems/`）も同じで、使うのは `xp` / `xl` / `xsl`（`python/backend.py`）だけにする。`numpy.` / `cupy.` は書かない。
- **`lib/linalg_bench.rb` は `bundler/setup` を読む。** この環境には numo-narray 0.9.2.1 と numo-narray-alt 0.11.2 の両方が入っている。bundler を通さずに `require 'numo/narray'` すると 0.9.2.1 が読まれ、続く `numo/linalg` が alt を読むので、定数の再定義警告が大量に出る。
- CuPy は `CUPY_TF32=0` で動かす（`python/backend.py` が既定で設定する）。`describe()` が cuBLAS の math mode を出力するので、0 であることを確かめる。cumo 側には `CUMO_ALLOW_TF32` を立てない。

## cumo の版

**`Cumo::Linalg` が入った最初のリリースは cumo 0.12.0（2026-10-04、タグ `v0.12.0` = `80d8ce0a`）である。** 0.11.0 には入っていない。このリポジトリは、既定で 0.15.0（2026-11-02）を読む。それより前に測った master のビルドは、版の文字列がどれも 0.11.0 なので、**`Cumo::VERSION` では区別できない。** master のビルドで測った結果には、版の文字列ではなくコミットを書くこと。

- **cumo は Gemfile に入れず、共有の gem 置き場にも入れない。** Gemfile に入れると、CUDA の無い環境で `bundle install` が通らなくなる。master を共有の置き場に入れると、同じ版の文字列でリリース版を上書きする。2026-10-01 に一度これをやってしまい、リリース版で測っている narray-llm の環境を壊した。
- ビルドは `vendor/`（git には入れない）に置く。既定は `vendor/cumo-0.15.0/` で、master の各コミットのビルドは `vendor/cumo-<コミット>/` に残してある。`GPU=1` のときは `LinalgBench.add_cumo_load_path` がその lib を `$LOAD_PATH` の先頭に足す。別のビルドと比べるときは `CUMO_LIB=<lib のパス>` で差し替える。読んだ `cumo.so` は、`script/bench.rb` の 2 列目に出る。
- リリース版の入れ方: `CUMO_NVCC_GENERATE_CODE=arch=compute_120,code=sm_120 gem install cumo -v 0.15.0 --no-document --ignore-dependencies --install-dir vendor/cumo-0.15.0`。
- master のビルドの入れ方: cumo を scratchpad に `git clone --no-hardlinks` し、目的のコミットで `rake build` を実行する（作業ツリーに別の作業が残っていることがあるため）。そのあと `CUMO_NVCC_GENERATE_CODE=arch=compute_120,code=sm_120 gem install --local --no-document --ignore-dependencies --install-dir vendor/cumo-<コミット> pkg/cumo-<版>.gem` を実行する。
- 2026-10-01: master `1c831278`（#581 `expm` まで）を入れた。`Cumo::Linalg` の公開関数は、`blas_char` を含めて 32 個。`nm -C cumo.so | grep -ci cusolver` は 77。
- 2026-10-03: master `aaa3162b`（#585 `triu` / `tril` のカーネル化まで）に入れ替えた。#585 の記号 `zero_triangle` は、`nm -C cumo.so | grep -c` で 27 個（`1c831278` では 0 個）。旧ビルドは `vendor/cumo-1c831278/` に残してあり、`CUMO_LIB` で同じバッチに並べられる。
- 2026-10-03: master `1c561d70`（#586 geqrf、orgqr、getrs、potrs の devInfo を読まない）に入れ替えた。`aaa3162b` は `vendor/cumo-aaa3162b/` に残した。#586 で増えた C の関数はインライン化されて記号に出ないので、`lib/cumo/linalg.rb` の差分（`getrs` の後の `LapackError` の検査が消えた）と、`script/bench.rb` の 2 列目の cumo.so のパスで区別する。
- 2026-10-03: master `50e07e88`（#587 `potrf` の fill を型と n で選ぶ）に入れ替えた。`1c561d70` は `vendor/cumo-1c561d70/` に残した。#587 は `lib/cumo/linalg.rb` だけの変更で、定数 `POTRF_UPPER_FILL_BELOW` の有無で区別する。
- 2026-10-04: rubygems の 0.12.0 を `vendor/cumo-0.12.0/` に入れ、既定をこれに切り替えた。`50e07e88` のビルドは `vendor/cumo-50e07e88/` に移した。0.12.0 は `50e07e88` に #588〜#594 を足したもの。
- 2026-10-12: rubygems の 0.13.0 を `vendor/cumo-0.13.0/` に入れた。既定は 0.12.0 のまま。0.12.0 との A/B は `CUMO_LIB` で並べた。`ext/cumo/narray/gen/kernel_parts.rb` の `KERNEL_PARTS` は、0.12.0 が 4 つ、0.13.0 が 9 つ（#610）。旧側を取り違えていないかは、これで確かめる。
- 2026-10-18: rubygems の 0.14.0 を `vendor/cumo-0.14.0/` に入れ、既定をこれに切り替えた。GPU の全条件を CuPy と並べて取り直し、README の表を更新した。`KERNEL_PARTS` は 0.13.0 と同じ 9 つなので、0.13.0 との区別には使えない。0.14.0 は `nm -C cumo.so | grep -ci cufft` で区別する（0.14.0 は 19、0.13.0 は 0）。gem の `cumo_revision` は `v0.13.0-11-gafd8ead9` で、cumo part3 によればこれが 0.14.0 の正しい文字列である（次のリリースからは #626 で直る）。`-I` で読むと `Cumo::REVISION` は nil になる。
- 2026-11-03: rubygems の 0.15.0 を `vendor/cumo-0.15.0/` に入れ、既定をこれに切り替えた。GPU の全条件を CuPy と並べて取り直し、README の表を更新した。0.15.0 は `nm -C cumo.so | grep -c cumo_cuda_cub_prime_ptx_version` で区別する（0.15.0 は 2、0.14.0 は 0）。cufft の記号はどちらも 19 で、`KERNEL_PARTS` は 9 つから 11 に増えた。gem の `cumo_revision` は `v0.15.0` で、#626 が効いている。ただし `Cumo::REVISION` は `Gem.loaded_specs` から読むので、lib を `$LOAD_PATH` に足すこのリポジトリの読み方では nil になる。確かめるときは、bundler を通さずに `GEM_HOME=vendor/cumo-0.15.0` で `gem "cumo"` してから読む。

## fixture

- **乱数を引くのは `script/make_fixtures.rb` の 1 か所だけ。** Numo の DFloat で、題材ごとに固定した seed で作る。Numo、Cumo、numpy の乱数は同じ seed でも一致しないので、3 実装とも `data/` の同じバイトを読む。
- 形式は `data/<題材>_<サイズ>.bin`（リトルエンディアンの fp64、行優先、配列を順に連結）と `data/<題材>_<サイズ>.json`（各配列の offset、shape、sha256、スカラーの属性）の 2 ファイル。
- `data/` は git に入れない。`rake fixtures` で作り直す。
- Hartree-Fock の積分だけは PySCF で作る。PySCF を実行時の依存にしないため、出力の `script/hartree_fock_sto3g.json` を追跡している。作り直すときは `script/hartree_fock_integrals.py` の先頭にある手順に従う。

## 関門（正しさ）

- **fp64 で回す。** tol は fp64 で 1e-10、fp32 で 1e-4 から始める。**Numo で実際の残差を測り、`docs/results/<題材>.md` に書いてから** その 10 倍に固定する。
- **関門の量は、検査対象の Linalg 関数を使わずに計算する。** `lib/linalg_bench/gates.rb` と `python/gates.py` の中身は、要素演算と `dot` だけで書いてある。
- 分解の結果そのもの（Q、U、固有ベクトル）は、符号や順序が一意に決まらないので直接は比べない。**再構成と性質で比べる。** 固有値と特異値は、降順に並べてから比べる。
- **既知の失敗は `known_failures.json` に記録する。** 題材、関門、実装（numo / cumo / numpy / cupy）、理由の組で書く。記録した関門だけが落ちた題材は `known_failure` で、`rake test` は通る。記録していない関門が落ちたら `fail`。記録した関門が通ったら、それも `fail` にする（直ったのに記録が残らないようにするため）。
- 「Numo と一致した」ではなく、**「正しい答えを出した」で判定する。** Numo と Cumo の直接比較は、関門とは別に `test/test_parity.rb` で行う（cumo が読めない環境では omit）。4 実装の物理量（`values`）が揃っているかは、`script/compare_gates.rb` で並べて確かめる。

## Numo と Cumo の差異

- **Cumo の縮約は 0 次元 NArray を返す。** 読み戻しは `LinalgBench.scalar` の 1 か所でだけ行う。**反復ループの中では読み戻さない。** 収束判定は間引く。
- `0d_narray > x` は Bit を返し、偽でも真として扱われる。テストで比べるときは、`scalar` で Ruby の数値にしてから比べる。
- Bit 配列、マスク代入、NArray を使った添字は避ける。Cumo は NArray の添字を同期して読み戻す。
- `Cumo::Linalg` は devInfo を読むために、呼び出しごとに同期する。cumo 0.10.0 以降、読み戻しは 1 回あたり約 18 µs かかる。小さい n（64〜256）では、これが 1 呼び出しの時間を支配しうる。計測区間内の同期回数が CuPy と揃っているかは、第九段階で nsys を使って数える。同期は `CUMO_SHOW_WARNING=ON CUMO_SHOW_WARNING_ONCE=OFF` で見える。
- `a.gemm(b, c.inplace, beta: 1)` の戻り値は、c とバッファを共有したまま inplace のフラグを持って返る。続けて `r * s` と書くと r が書き換わる。外に返す前に `.view` を通すと、フラグが落ちる。
- 転置ビューを `dot` に渡すと、コピーせずに `CUBLAS_OP_T` へ渡る。速いかどうかは M で決まる。M が小さいと `OP_T` が得で、M が大きいとコピーしたほうが 10〜22% 得になる。常に連続化する関数を挟むと、もともと連続な配列でもコピーを 1 本払う。
- NArray どうしの `==` は、要素ごとではなく真偽を 1 つだけ返す。テストでは `to_a` どうしで比べる。
- `x.inplace.cumsum` は 0.10.0 から x に書き込まない（戻り値は正しい）。

## 計測の作法

- `script/bench.rb` と `python/bench.py` は、1 題材 × 1 サイズ × 1 dtype を測り、TSV を 1 行出す。6 列目が ms/step。
- best-of-3、ウォームアップ 1 回。区間内では同期せず、区間の終わりに 1 回だけ同期する（Cumo は `cudaDeviceSynchronize`、CuPy は `Stream.null.synchronize()`）。fixture の読み込みとプロセスの起動は区間に含めない。
- **README に載せる数字は `bench/run.sh` の 10 ラウンド（対照系列付き）から取ったものだけにする。** 対照が 1 をまたがない条件は載せない。単発の数字は書かない。
- 1 プロセスが 2 秒以上回るように `--inner` を選び、`bench/conditions.tsv` に書く。
- GPU の表と CPU の表を混ぜない。
- CPU の表の Numo と numpy は、どちらも OpenBLAS 0.3.34 だが、ビルドが別物である（numo-linalg-alt 同梱と scipy-openblas）。narray-llm の GEMM では、1 スレッドでほぼ互角、24 スレッドで numpy が 2.0 倍速かった。**Numo / numpy の比は BLAS のビルドの差として読み、表にはビルド名とスレッド数を添える。**
- GPU の系列でも numo-linalg-alt を読むと OpenBLAS のスレッドが立ち、空回りする。GPU の wall にはほぼ出ないが、load average を読むときに紛れる。
- 検証や計測のスクリプトは単一プロセスで回すか、並列数を絞る（`xargs -P2`）。子プロセスにはすべて `timeout` を付ける。ベンチのスクリプトは bash で書く。ログインシェルは zsh で、zsh は変数に入れた引数列を単語に分割しない。

以下は narray-llm で実際に踏んだもの。規則の根拠は narray-llm の AGENTS.md「計測の作法」と docs/measurement-cases.md にある。

### クロックとメモリの段

- **固定は再起動で外れる。計測の直前に、毎回読んで確かめる。** アイドル時の SM が 2790〜2805 MHz なら固定されている。外れていると 180〜1500 MHz あたりになる。GPU 条件では、`bench/run.sh` が読んで、外れていれば止まる。前の計測の直後に読むと、固定されていても 2655 MHz のような値が出て止まることがある（2026-11-03）。読み直して 2790〜2805 MHz なら、そのまま回し直してよい。
- メモリクロックは 9001 / 11001 / 14001 MHz の 3 段で、読み出し帯域は 413 / 508 / 600 GB/s と 1.45 倍違う。`-lmc` で固定しても、段は電力に応じて動く。アイドルの読みでは判定できない。**帯域が効く題材（tall-skinny の `qr` や `lstsq`）では特に効く。**
- 同じバッチの中で、一部のプロセスだけが計測区間のあいだ 9001 に張り付き、値が 2 群に割れることがある。その場合、10 点の中央値は 2 群の間に落ち、どちらでもない数字になる。`bench/aggregate.py` は系列ごとの幅（max ÷ min）を出す。**幅が 5% を超えたら、`$out/<条件>.tsv` の生の系列をラウンド順に見る。** 並べ替えた形で記録しない。値の大小で外れ値を捨てない。段で選り分けるなら、計測区間の時刻を出し、10 ms 間隔の `nvidia-smi --query-gpu=timestamp,clocks.mem` と突き合わせる。サンプラの間隔は、区間の 1/5 以下にする。
- `bench/run.sh` は計測のあいだ、load average と CPU を最も使っている 3 プロセスを、5 秒おきに `$out/monitor.log` に書く。GPU 条件では、クロックと GPU を使うプロセス数も書く。別の作業がラウンドに重なっていないかを、後から確かめるためのもの。重い作業（cumo のビルドなど）を回す前後には、他のセッションに一声かける。
- **CPU の計測は、マシン全体のコアを使う。** 別のセッションのビルドや nsys が重なると、値が数倍に跳ねる。2026-10-03 には、G と A の CPU 計測に cumo のビルドが重なり、`g_solve` n = 4096 の後半のラウンドが 5 倍前後になった。CPU の計測を始める前に、他のセッションに重い作業を控えてもらう。

### 帰属

- **cumo の 2 つのコミットを比べるとき、数 % の差はコードの配置だけで出る。** narray-llm では、関数の番地を 64 / 128 バイトずらしただけのビルドが 1.5〜2.6% 速く出た（12/12）。配置の揺れと同じ大きさの差を cumo の変更のせいにするなら、配置だけをずらしたビルドを同じバッチに並べる。小さい n で投入が支配する条件ほど、この影響が出やすい。
- **どの cumo を読んでいるかは、記号の数で確かめる。** 系列ごとに `$LOADED_FEATURES.grep(/cumo\.so/)` を出し、`nm -C cumo.so | grep -c <その変更で増えた記号>` を数える。今回の変更の記号だけでなく、**直前の PR の記号も、両方のビルドで数える。**
- **帰属を決める前に、予算を机上で出す。** 線形代数では、O(n³) の FLOP 数から GFLOP/s を出し、ピークの何割かを表に添える。そうすると、遅さが実装の差なのか土俵の差なのかを読み分けられる。観測した差が予算より大きければ、別の何かが混ざっている。
- 中くらいの n の行列は L2 に収まる。同じ配列を読み回す単体ベンチでは、帯域の天井を超えた数字が出る。帯域を主張するなら、毎回別の配列を読む形も 1 本並べる。出力を毎回確保する書き方は、確保の費用を両系列に乗せる。実際のコードが毎回確保するなら、確保込みの値も併記する。

### 読み方

- 比の範囲が 1 をまたぐのは「大きさが決まらない」という判定で、「向きが決まらない」という意味ではない。`bench/aggregate.py` は、比ごとに符号検定の両側 p を出す。帰無分布は対照ペアで見る。
- 2 つの測定の差から、機構を書かない。説明は、独立に測ってから書く。
- 「カーネル数と出力が同じだから、違いは同期だけ」とは言えない。同期だけを足した対照で確かめる。
- best-of-N の値に回数を掛けて、単価にしない。
- Python の移植が Ruby と同じアルゴリズムの綴りになっているかは、測る前に確かめる。narray-llm では、CuPy だけヘッドごとのループで走らせて、8.27 倍と出したことがある（正しくは 3.41 倍）。
- 実装に入る前に、その経路が実際に呼ばれるかを確かめる。

### nsys

- `nsys stats` には必ず `--force-export=true` を付ける。付けないと、失敗した回の古い .sqlite が黙って使われる。
- 回数は 1 組の profile で確定する。時間は確定しないので、自分を 3 組測ってから相手と比べる。

### 記録

- `/tmp` は tmpfs で、再起動すると scratchpad ごと消える。残す数字は、その日のうちに `docs/results/` に書く。

## テストの作法

- **GPU が無くても `rake test` が全部通る状態を保つ。** テストは Numo と numpy で動く。
- Python の venv（`python/.venv`）が無いときは、Python 側のテストは omit になる。
- 再現用の一時スクリプトは scratchpad に置く。リグレッションテストとしてコミットするかどうかは、その都度確認する。

## 環境

```
bundle install
python3 -m venv python/.venv
python/.venv/bin/pip install -r python/requirements.txt
rake fixtures
rake test
```
